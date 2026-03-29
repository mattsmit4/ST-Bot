"""
LLM-Based Product Narrowing Analyzer

When a search returns too many tied products, this module analyzes the product pool
to find the most differentiating attribute and generates a smart narrowing question.

Like "21 questions" — it finds the attribute that splits the pool most evenly
and asks the user about it, filtering the pool until <= 3 products remain.
"""

import os
import re
import json
import logging
from typing import Optional


def _clean_value(val) -> str:
    """Convert value to display string, stripping trailing .0 from floats and newlines."""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val).replace('\n', ' ').replace('\r', ' ').strip()


def _normalize_for_grouping(val: str) -> str:
    """Canonical form for grouping: lowercase, normalize speed/resolution units, strip non-alphanumeric."""
    s = str(val).lower()
    # Normalize speed units
    s = re.sub(r'gbit/s', 'gbps', s)
    s = re.sub(r'mbit/s', 'mbps', s)
    # Remove parenthetical resolution labels like "(4K)", "(UHD)", "(Ultra HD 4K)"
    s = re.sub(r'\s*\([^)]*(?:4k|uhd|hd|fhd|qhd)[^)]*\)', '', s, flags=re.IGNORECASE)
    # Strip refresh rate suffixes so "4K 60Hz" and "4K" group together
    s = re.sub(r'@?\s*\d+\s*hz', '', s)
    # Normalize resolution: map pixel counts to standard labels
    s = re.sub(r'3840\s*x\s*2160p?', '4k', s)
    s = re.sub(r'4096\s*x\s*2160p?', '4k', s)
    s = re.sub(r'4k\s*x?\s*2k', '4k', s)
    # Strip all non-alphanumeric
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def _merge_similar_values(value_counts: dict) -> tuple[dict, dict]:
    """Group values by normalized form. Returns (merged_counts, filter_overrides).

    merged_counts: {display_label: total_count} — most common variant as key, summed counts.
    filter_overrides: {display_label: {"op": "contains", "value": common_prefix}} for merged entries.
    """
    groups = {}  # normalized_form -> [(original_value, count), ...]
    for val, count in value_counts.items():
        norm = _normalize_for_grouping(val)
        groups.setdefault(norm, []).append((val, count))

    merged_counts = {}
    filter_overrides = {}
    for norm, variants in groups.items():
        variants.sort(key=lambda x: -x[1])
        label = variants[0][0]
        total = sum(c for _, c in variants)
        merged_counts[label] = total

        if len(variants) > 1:
            raw_vals = [v for v, _ in variants]
            prefix = os.path.commonprefix(raw_vals).rstrip()
            if prefix:
                filter_overrides[label] = {"op": "contains", "value": prefix}

    return merged_counts, filter_overrides


from core.api_retry import RetryHandler, RetryConfig
from core.openai_client import get_openai_client

_logger = logging.getLogger(__name__)


NARROWING_ANALYSIS_PROMPT = """You are a product assistant for StarTech.com helping narrow down search results.

The user searched for products and got too many equally-matched results. Your job is to find the SINGLE attribute that best differentiates the products and ask ONE smart question.

## Strategy (like 20 questions)
1. Look at ALL product metadata below
2. Find the attribute with the MOST VARIATION — the one that would eliminate the most products with a single answer
3. Generate a friendly question with CONCRETE options based on the ACTUAL values in the data. For numeric attributes (lengths, wattages, sizes), group nearby values into ranges (e.g., "3-4 ft" not "3.0 ft, 3.3 ft") and use the "in_range" filter op. For categorical attributes (connectors, color), list each distinct value.
4. Options should use REAL values from the products, not made-up ones

## Product Metadata Fields Available (only fields with data for this product pool):
{field_descriptions}

## Rules
1. Pick the attribute that SPLITS the pool most evenly — not one where 70% or more share the same value. Fields marked with ⚠️ in the value distribution are poor choices. Your options should collectively cover as many products as possible. You may use up to 6 options (plus "Not sure / show me all") to minimize the "Other" bucket. A large "Other" bucket means you picked poorly — show more distinct values rather than lumping them into "Other".
2. Options must use EXACT values from the value distribution summary below. Copy the value strings precisely into your option_filters — do not paraphrase, abbreviate, or reformat them. ("HDMI or VGA?" not "digital or analog?") EXCEPTION: If multiple values clearly represent the same spec with different formatting (e.g., "10Gbps" vs "10 Gbit/s"), group them into one option and use a `contains` filter with the shared prefix.
3. Include product counts in each option so the user knows the pool size
4. Always include "Not sure / show me all" as the LAST option
5. Keep the question text conversational and short. Do NOT include option values, counts, or bullet points in the "question" field — those belong only in the "options" array
6. NEVER mention SKUs or prices
7. If this is a follow-up question (questions_asked > 0), acknowledge progress ("Great choice! Now...")
8. Generate a natural intro that DIRECTLY responds to the user's query. If they asked a question ("Do you have X?"), answer it ("Yes, we have X!") — UNLESS the user prompt says no exact matches were found, in which case be honest: "We don't have an exact match for [X], but here are N similar options." Use their words naturally. 1-2 sentences max. Include the product count.
9. Option counts MUST add up to the pool size. NEVER pick an attribute with 0% coverage — if a field shows 0/N in the coverage summary, no products have data for it and ALL will end up in "Other". STRONGLY PREFER attributes where ALL or most products have data. Only use attributes with < 100% coverage if no better option exists.
10. If the user prompt says no exact matches were found, these are fallback alternatives. Frame your question to help find the closest match — e.g., "To find the best alternative, which feature matters most?" Don't imply products fully match what the user asked for.

## Response Format (JSON only, no markdown):
{{
    "intro": "Natural intro that responds to the user's query and mentions the count",
    "attribute": "the_metadata_field_name",
    "reasoning": "Brief explanation of why this splits the pool best",
    "question": "Your friendly narrowing question",
    "options": ["Option A (N products)", "Option B (N products)", "Not sure / show me all"],
    "option_filters": [
        {{"field": "field_name", "op": "eq|contains|gte|lte|in_range", "value": "filter_value"}},
        {{"field": "field_name", "op": "eq|contains|gte|lte|in_range", "value": "filter_value"}},
        null
    ]
}}

The option_filters array maps 1:1 to the options array. Each entry defines how to filter the product pool if that option is picked. The last option ("Not sure") maps to null.

Supported ops: "eq" (exact match), "contains" (substring/list membership), "gte" (>=), "lte" (<=), "in_range" (between two values, value="min,max").
"""

# Human-readable descriptions for each metadata field, used to build dynamic system prompt
FIELD_DESCRIPTIONS = {
    'connectors': 'Connector types (USB-C, HDMI, DisplayPort, VGA, etc.)',
    'features': 'Product features (4K, HDR, Power Delivery, etc.)',
    'length_ft': 'Cable/adapter length (feet)',
    'length_m': 'Cable/adapter length (meters)',
    'port_count': 'Number of ports',
    'hub_ports': 'Number of ports',
    'usb_version': 'USB version (2.0, 3.0, 3.2 Gen 2)',
    'hub_usb_version': 'USB version',
    'power_delivery_watts': 'Power delivery wattage',
    'dock_num_displays': 'Number of displays supported',
    'max_refresh_rate': 'Maximum refresh rate (Hz)',
    'color': 'Product color',
    'network_rating': 'Network cable rating (Cat5e, Cat6, Cat6a)',
    'network_max_speed': 'Maximum network speed',
    'kvm_ports': 'KVM switch port count',
    'kvm_video_type': 'KVM video interface',
    'mount_type': 'Mount style (wall, desk, pole)',
    'mount_display_range': 'Display size range for mounts',
    'bay_count': 'Drive bay count',
    'num_drives': 'Number of drive bays',
    'drive_size': 'Drive size (2.5", 3.5", M.2)',
    'rack_height_u': 'Rack height in U',
    'u_height': 'Rack unit height',
    'Frame_Type': 'Rack frame type (Open Frame, Enclosed Cabinet, etc.)',
    'rack_type': 'Rack type (2-Post, 4-Post, etc.)',
    'Wallmountable': 'Wall mountable (Yes or No)',
    'screen_size_inches': 'Screen size for privacy screens',
    'fiber_type': 'Fiber mode (Single-mode or Multimode)',
    'Fiber_Classificiation': 'Fiber classification (OM3, OM4, OS2, etc.)',
    'supported_resolutions': 'Supported video resolutions (4K@60Hz, 1080p, etc.)',
    'max_dvi_resolution': 'Maximum digital resolution supported',
}


NARROWING_RESPONSE_PROMPT = """Match the user's response to one of the provided options.

Options that were presented:
{options_text}

User's response: "{response}"

Return JSON only:
{{
    "selected_index": <0-based index of the matching option, or -1 if unclear>,
    "confidence": <0.0-1.0>
}}
"""


# Metadata fields to extract from products for the LLM prompt
DIFF_FIELDS = [
    'connectors', 'features', 'length_ft', 'length_m',
    'hub_ports', 'port_count', 'hub_usb_version', 'usb_version',
    'hub_power_delivery', 'power_delivery_watts',
    'dock_num_displays', 'max_refresh_rate', 'color',
    'network_rating', 'network_max_speed',
    'kvm_ports', 'kvm_video_type',
    'mount_type', 'mount_display_range',
    'drive_size', 'bay_count', 'num_drives',
    'u_height', 'rack_height_u',
    'Frame_Type', 'rack_type', 'Wallmountable',
    'screen_size_inches',
    'fiber_type', 'Fiber_Classificiation',
    'supported_resolutions', 'max_dvi_resolution',
]

# SearchFilters field → DIFF_FIELDS to hide from narrowing LLM when that filter is active.
# Prevents narrowing from re-asking about dimensions the user already specified.
FILTER_TO_DIFF_FIELDS = {
    'port_count':               ['port_count', 'hub_ports', 'kvm_ports'],
    'requested_network_speed':  ['network_max_speed', 'network_rating', 'nw_cable_type', 'network_speed'],
    'screen_size':              ['screen_size_inches'],
    'requested_refresh_rate':   ['max_refresh_rate'],
    'requested_power_wattage':  ['power_delivery_watts', 'hub_power_delivery'],
    'length':                   ['length_ft', 'length_m'],
    'connector_from':           ['connectors', 'kvm_video_type'],
    'connector_to':             ['connectors', 'kvm_video_type'],
    'rack_height':              ['u_height', 'rack_height_u'],
    'bay_count':                ['bay_count', 'num_drives', 'num_drives_raw'],
    'drive_size':               ['drive_size', 'drive_size_raw'],
    'kvm_video_type':           ['kvm_video_type'],
    'usb_version':              ['usb_version', 'hub_usb_version', 'io_interface_raw'],
    'color':                    ['color'],
}

class LLMNarrowingAnalyzer:
    """Analyzes product pools and generates narrowing questions using LLM."""

    def __init__(self):
        self._retry_handler = RetryHandler(RetryConfig(
            max_attempts=2,
            base_delay=1.0,
            max_delay=5.0,
        ))

    def analyze_and_question(
        self,
        products: list,
        original_query: str,
        questions_asked: int = 0,
        exclude_attributes: list[str] = None,
        original_filters: dict = None,
        is_fallback: bool = False,
    ) -> Optional[dict]:
        """
        Analyze a product pool and generate a narrowing question.

        Args:
            products: List of Product objects in the pool
            original_query: The user's original search query
            questions_asked: Number of narrowing questions asked so far
            exclude_attributes: Attributes already asked about (skip these)
            original_filters: Filters already applied from the user's search (skip these)

        Returns:
            dict with 'attribute', 'question', 'options', 'option_filters'
            or None if LLM fails
        """
        client = get_openai_client()
        if not client:
            return None

        user_prompt = self._build_analysis_prompt(products, original_query, questions_asked, exclude_attributes, original_filters, is_fallback=is_fallback)

        # Build dynamic system prompt — only include fields that have data in this pool
        excluded_fields = set()
        if exclude_attributes:
            for attr in exclude_attributes:
                excluded_fields.add(attr)
                excluded_fields.update(LLMNarrowingAnalyzer.FIELD_ALIASES.get(attr, []))
        if original_filters:
            for k in original_filters:
                excluded_fields.update(FILTER_TO_DIFF_FIELDS.get(k, []))

        sample = products  # Use ALL products for accurate distributions
        valid_fields = set()
        for f in DIFF_FIELDS:
            if f in excluded_fields:
                continue
            for p in sample:
                val = p.metadata.get(f)
                if val is not None and str(val).lower() not in ('nan', 'none', ''):
                    valid_fields.add(f)
                    break  # At least one product has data — field is valid

        # Auto-discover tier1 fields from category_columns for this product pool
        from config.category_columns import get_tier1_config
        pool_categories = set(p.metadata.get('category', '') for p in products)
        for cat in pool_categories:
            for field_name, display_label in get_tier1_config(cat).items():
                if field_name in excluded_fields:
                    continue
                for p in products:
                    val = p.metadata.get(field_name)
                    if val is not None and str(val).lower() not in ('nan', 'none', ''):
                        valid_fields.add(field_name)
                        if field_name not in FIELD_DESCRIPTIONS:
                            FIELD_DESCRIPTIONS[field_name] = display_label
                        break

        field_lines = []
        for f in list(DIFF_FIELDS) + [f for f in valid_fields if f not in DIFF_FIELDS]:
            if f in valid_fields and f in FIELD_DESCRIPTIONS:
                field_lines.append(f"- {f}: {FIELD_DESCRIPTIONS[f]}")
        field_desc = "\n".join(field_lines) if field_lines else "- No differentiating fields available"
        system_prompt = NARROWING_ANALYSIS_PROMPT.format(field_descriptions=field_desc)

        try:
            response = self._retry_handler.execute(
                lambda: client.chat.completions.create(
                    model=os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano'),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_completion_tokens=500,
                    timeout=30.0
                )
            )

            content = response.choices[0].message.content.strip()
            result = self._parse_analysis_response(content)
            if result:
                result = self._validate_option_counts(result, products)
            if result is None:
                result = self._try_code_fallback(products, exclude_attributes, original_filters, is_fallback)
            return result

        except Exception as e:
            _logger.warning(f"Narrowing analysis failed: {e}")
            return self._try_code_fallback(products, exclude_attributes, original_filters, is_fallback)

    def _try_code_fallback(self, products, exclude_attributes, original_filters, is_fallback):
        """Code-based fallback with strict then lenient validation."""
        fallback = self._compute_fallback_question(
            products, exclude_attributes, original_filters, is_fallback=is_fallback
        )
        if not fallback:
            return None
        result = self._validate_option_counts(fallback, products)
        if result is None:
            result = self._validate_option_counts(fallback, products, lenient=True)
        return result

    def parse_response_and_filter(
        self,
        user_response: str,
        product_pool: list,
        last_options: list[str],
        option_filters: list,
    ) -> Optional[dict]:
        """
        Parse user's narrowing response and filter the product pool.

        Args:
            user_response: The user's response text
            product_pool: Current product pool
            last_options: Options that were presented
            option_filters: Filter specs from the question generation

        Returns:
            dict with 'filtered_pool' and 'selected' or None if parsing fails
        """
        # Fast path: direct string matching
        selected_idx = self._fast_match(user_response, last_options)

        if selected_idx is None:
            # LLM fallback for ambiguous responses
            selected_idx = self._llm_match(user_response, last_options)

        if selected_idx is None or selected_idx < 0:
            return None

        # "Not sure / show me all" is always the last option
        if selected_idx >= len(last_options) - 1:
            # Only return show_all if the user actually asked to see all.
            # If the LLM picked show_all because nothing else matched (user gave
            # a different filter like "1-2 meters" during a connector question),
            # return None so the handler falls back to "can't parse" instead of
            # dumping the entire pool.
            show_all_words = {'show', 'all', 'not sure', 'everything', 'unsure', "don't know"}
            response_lower = user_response.lower()
            if any(w in response_lower for w in show_all_words):
                return {'filtered_pool': product_pool, 'selected': 'show_all'}
            return None

        # Apply the filter
        if selected_idx < len(option_filters) and option_filters[selected_idx]:
            filter_spec = option_filters[selected_idx]
            filtered = self._apply_filter(product_pool, filter_spec)

            # Exclude products that match earlier options (matches validation counting)
            earlier_ids = set()
            for i in range(selected_idx):
                if i < len(option_filters) and option_filters[i]:
                    earlier = self._apply_filter(product_pool, option_filters[i])
                    earlier_ids.update(id(p) for p in earlier)
            if earlier_ids:
                filtered = [p for p in filtered if id(p) not in earlier_ids]

            # If filtering removed everything, return the full pool
            if not filtered:
                return {'filtered_pool': product_pool, 'selected': 'filter_empty'}

            return {'filtered_pool': filtered, 'selected': last_options[selected_idx]}

        return {'filtered_pool': product_pool, 'selected': 'no_filter'}

    def _build_analysis_prompt(self, products, original_query, questions_asked, exclude_attributes=None, original_filters=None, is_fallback=False):
        """Build the user prompt with product metadata summary."""
        parts = [f'User searched: "{original_query}"']
        parts.append(f"Pool size: {len(products)} products")
        parts.append(f"Questions asked so far: {questions_asked}")

        if exclude_attributes:
            parts.append(f"DO NOT ask about these attributes (already asked): {', '.join(exclude_attributes)}")
        if original_filters:
            filter_parts = []
            for k, v in original_filters.items():
                aliases = LLMNarrowingAnalyzer.FIELD_ALIASES.get(k, [])
                if aliases:
                    names = '/'.join([k] + aliases)
                    filter_parts.append(f"{names}={v}")
                else:
                    filter_parts.append(f"{k}={v}")
            filter_desc = ', '.join(filter_parts)
            if is_fallback:
                parts.append(f"IMPORTANT: No exact matches found. User wanted: {filter_desc}, but some requirements could NOT be met. The products below are the closest alternatives, NOT exact matches. DO NOT ask about these attributes — but your intro MUST be honest about unmet requirements.")
            else:
                parts.append(f"User's search already filtered on: {filter_desc}. DO NOT ask about these attributes.")
        parts.append("")
        parts.append("Product metadata:")

        # Build set of fields to exclude from metadata (already filtered + already asked + all aliases)
        excluded_fields = set()
        if original_filters:
            for k in original_filters:
                excluded_fields.update(FILTER_TO_DIFF_FIELDS.get(k, []))
        if exclude_attributes:
            for attr in exclude_attributes:
                excluded_fields.add(attr)
                excluded_fields.update(LLMNarrowingAnalyzer.FIELD_ALIASES.get(attr, []))

        # Add field coverage summary so LLM can prefer attributes with complete data
        sample = products  # Use ALL products for accurate coverage stats
        sample_size = len(sample)
        coverage_lines = []
        for f in DIFF_FIELDS:
            if f in excluded_fields:
                continue
            count = sum(1 for p in sample if p.metadata.get(f) is not None
                        and str(p.metadata.get(f)).lower() not in ('nan', 'none', ''))
            coverage_lines.append(f"  {f}: {count}/{sample_size}")
        if coverage_lines:
            parts.append("Field coverage (prefer fields with full coverage):")
            parts.extend(coverage_lines)
            parts.append("")

        # Add value distribution so LLM knows exactly which values exist
        parts.append("Value distribution per field (ONLY pick attributes listed here; use EXACT values in option_filters):")
        for f in DIFF_FIELDS:
            if f in excluded_fields:
                continue
            value_counts = {}
            for p in sample:
                val = p.metadata.get(f)
                if val is None or str(val).lower() in ('nan', 'none', ''):
                    continue
                if isinstance(val, list):
                    for v in val:
                        v_str = _clean_value(v)
                        if v_str:
                            value_counts[v_str] = value_counts.get(v_str, 0) + 1
                else:
                    v_str = _clean_value(val)
                    if v_str:
                        value_counts[v_str] = value_counts.get(v_str, 0) + 1
            if value_counts:
                # Merge near-duplicate values before showing to LLM
                value_counts, _ = _merge_similar_values(value_counts)
                top = sorted(value_counts.items(), key=lambda x: -x[1])[:6]
                total = sum(value_counts.values())
                max_pct = max(value_counts.values()) / total * 100
                balance_note = f" ⚠️ top value covers {max_pct:.0f}%" if max_pct > 70 else ""
                parts.append(f"  {f}: " + ", ".join(f'"{v}" ({c})' for v, c in top) + balance_note)
        parts.append("")

        # Cap at 20 products to keep prompt manageable
        for i, p in enumerate(products[:20], 1):
            fields = {}
            for f in DIFF_FIELDS:
                if f in excluded_fields:
                    continue
                val = p.metadata.get(f)
                if val is not None and str(val).lower() not in ('nan', 'none', ''):
                    # Truncate long lists
                    if isinstance(val, list) and len(val) > 5:
                        val = val[:5]
                    # Clean float values (3.0 → 3) before sending to LLM
                    if isinstance(val, float) and val == int(val):
                        val = int(val)
                    fields[f] = val

            name = p.metadata.get('name', p.product_number)
            parts.append(f"{i}. {p.product_number} ({name}): {json.dumps(fields, default=str)}")

        if len(products) > 20:
            parts.append(f"... and {len(products) - 20} more products with similar attributes")

        return "\n".join(parts)

    def _parse_analysis_response(self, content: str) -> Optional[dict]:
        """Parse the LLM's JSON response."""
        try:
            # Strip markdown code fences if present
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Validate required fields
            if not all(k in data for k in ('question', 'options', 'option_filters')):
                _logger.warning(f"Narrowing response missing required fields: {list(data.keys())}")
                return None

            if len(data['options']) < 2:
                _logger.warning("Narrowing response has fewer than 2 options")
                return None

            # Ensure option_filters matches options length
            while len(data.get('option_filters', [])) < len(data['options']):
                data['option_filters'].append(None)

            return {
                'intro': data.get('intro', ''),
                'attribute': data.get('attribute', 'unknown'),
                'question': data['question'],
                'options': data['options'],
                'option_filters': data['option_filters'],
            }

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            _logger.warning(f"Failed to parse narrowing response: {e}")
            return None

    def _compute_adaptive_tolerance(self, filters: list) -> None:
        """Set adaptive tolerance on numeric eq filters targeting the same field."""
        numeric_eq = []
        for i, filt in enumerate(filters):
            if filt and filt.get('op') == 'eq':
                try:
                    float(filt['value'])
                    numeric_eq.append((i, filt))
                except (ValueError, TypeError):
                    pass
        if len(numeric_eq) > 1:
            fields = set(f['field'] for _, f in numeric_eq)
            if len(fields) == 1:  # All target the same field
                values = sorted(float(f['value']) for _, f in numeric_eq)
                min_gap = min(values[j+1] - values[j] for j in range(len(values)-1))
                adaptive_tolerance = min_gap / 2
                for _, filt in numeric_eq:
                    filt['tolerance'] = adaptive_tolerance

    def _validate_option_counts(self, result: dict, products: list, lenient: bool = False) -> dict:
        """Pre-apply filters, replace LLM counts, and drop empty options."""
        options = result['options']
        filters = result['option_filters']

        self._compute_adaptive_tolerance(filters)

        validated_options = []
        validated_filters = []
        all_matched_ids = set()
        for opt, filt in zip(options, filters):
            if filt is None:
                # Only keep the "Not sure / show me all" option as-is
                if 'not sure' in opt.lower() or 'show me all' in opt.lower():
                    validated_options.append(opt)
                    validated_filters.append(filt)
                # Other null-filter options are dropped (products caught by "Other" catch-all)
                continue
            matched = self._apply_filter(products, filt)
            # Only count products not already claimed by a previous option
            unique_matched = [p for p in matched if id(p) not in all_matched_ids]
            actual_count = len(unique_matched)
            if actual_count == 0:
                continue  # Drop empty/redundant options
            all_matched_ids.update(id(p) for p in unique_matched)
            validated_options.append(re.sub(
                r'\(\d+ products?\)',
                f'({actual_count} product{"s" if actual_count != 1 else ""})',
                opt
            ))
            validated_filters.append(filt)

        # Add "Other" option for products that don't match any filter
        unmatched = [p for p in products if id(p) not in all_matched_ids]
        if unmatched:
            gap = len(unmatched)
            # Reject attribute if "Other" covers too much of the pool (low coverage field)
            if not lenient and gap > len(products) * 0.4:
                _logger.warning(f"Narrowing 'Other' covers {gap}/{len(products)} products — attribute has low coverage, skipping")
                return None
            other_label = f"Other ({gap} product{'s' if gap != 1 else ''})"
            other_skus = ','.join(p.product_number for p in unmatched)
            other_filter = {"field": "product_number", "op": "in_list", "value": other_skus}
            # Insert before "Not sure / show me all" (last option with filter=None)
            if validated_filters and validated_filters[-1] is None:
                validated_options.insert(-1, other_label)
                validated_filters.insert(-1, other_filter)
            else:
                validated_options.append(other_label)
                validated_filters.append(other_filter)
            all_matched_ids.update(id(p) for p in unmatched)

        # Safety net: if no useful options remain (only Other + Not sure), skip narrowing
        non_other = [o for o in validated_options
                     if 'other' not in o.lower() and 'not sure' not in o.lower()]
        if not non_other:
            _logger.warning("Narrowing produced no useful options — all filters matched 0 products")
            return None

        # Safety net: if any single option covers the entire pool, question can't narrow
        for opt in non_other:
            count_match = re.search(r'\((\d+) products?\)', opt)
            if count_match and int(count_match.group(1)) >= len(products):
                _logger.warning(f"Narrowing option '{opt}' covers entire pool — skipping attribute")
                return None

        result['options'] = validated_options
        result['option_filters'] = validated_filters
        result['total_accounted'] = len(all_matched_ids)
        return result

    def _fast_match(self, user_response: str, options: list[str]) -> Optional[int]:
        """Try to match user response to an option via string matching."""
        response_lower = user_response.strip().lower()

        # Check for ordinal references: "2nd option", "option 2", "the second one"
        ordinal_map = {
            r'\b(?:1st|first)\b': 0,
            r'\b(?:2nd|second)\b': 1,
            r'\b(?:3rd|third)\b': 2,
            r'\b(?:4th|fourth)\b': 3,
        }
        for pattern, idx in ordinal_map.items():
            if re.search(pattern, response_lower):
                real_options = [i for i, o in enumerate(options)
                                if 'not sure' not in o.lower() and 'show me all' not in o.lower()]
                if idx < len(real_options):
                    return real_options[idx]
                return None

        # Check for bare number: "2", "option 2", "product 3", "#3"
        bare_num = re.match(r'^(?:(?:option|product)\s*#?\s*)?(\d+)$', response_lower)
        if not bare_num:
            bare_num = re.match(r'^#?\s*(\d+)$', response_lower)
        if bare_num:
            num = int(bare_num.group(1))
            # First: check if the number matches an option VALUE (e.g., "4" matches "4 (7 products)")
            # Also matches "6" against "6 ports" by checking the leading number
            for i, option in enumerate(options):
                option_value = option.split('(')[0].strip()
                if option_value == str(num):
                    return i
                option_leading = option_value.split()[0] if option_value else ''
                if option_leading == str(num):
                    return i
            # Then: fall back to positional index (e.g., "4" = 4th option)
            real_options = [i for i, o in enumerate(options)
                            if 'not sure' not in o.lower() and 'show me all' not in o.lower()]
            if 1 <= num <= len(real_options):
                return real_options[num - 1]
            return None

        # Exact match (ignoring case and product counts)
        for i, option in enumerate(options):
            # Strip product count like "(8 products)" for matching
            option_clean = option.split('(')[0].strip().lower()
            if response_lower == option_clean:
                return i

        # Substring match — user's response is contained in an option
        for i, option in enumerate(options):
            option_clean = option.split('(')[0].strip().lower()
            if response_lower in option_clean or option_clean in response_lower:
                return i

        # Check for "show all", "not sure", "all of them" etc.
        show_all_patterns = ['not sure', 'show all', 'show me all', 'all of them',
                             "don't care", 'dont care', 'any', 'all']
        if response_lower in show_all_patterns:
            return len(options) - 1

        return None

    def _llm_match(self, user_response: str, options: list[str]) -> Optional[int]:
        """Use LLM to match ambiguous responses to options."""
        client = get_openai_client()
        if not client:
            return None

        options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
        prompt = NARROWING_RESPONSE_PROMPT.format(
            options_text=options_text,
            response=user_response
        )

        try:
            response = client.chat.completions.create(
                model=os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=50,
                timeout=15.0
            )

            content = response.choices[0].message.content.strip()
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            idx = data.get('selected_index', -1)
            confidence = data.get('confidence', 0.0)

            if idx >= 0 and confidence >= 0.5:
                return idx

        except Exception as e:
            _logger.warning(f"LLM match failed: {e}")

        return None

    # Field aliases — some metadata fields have different names depending on product type
    FIELD_ALIASES = {
        'usb_version': ['hub_usb_version', 'storage_interface'],
        'hub_usb_version': ['usb_version', 'storage_interface'],
        'port_count': ['hub_ports', 'kvm_ports'],
        'hub_ports': ['port_count', 'kvm_ports'],
        'kvm_ports': ['port_count', 'hub_ports'],
        'power_delivery_watts': ['hub_power_delivery', 'power_delivery'],
        'hub_power_delivery': ['power_delivery_watts', 'power_delivery'],
        'length_ft': ['length_m'],
        'length_m': ['length_ft'],
    }

    def _compute_fallback_question(self, products, exclude_attributes=None, original_filters=None, is_fallback=False):
        """Code-based fallback: pick the field with most variation, generate options from data."""
        excluded_fields = set()
        if exclude_attributes:
            for attr in exclude_attributes:
                excluded_fields.add(attr)
                excluded_fields.update(self.FIELD_ALIASES.get(attr, []))
        if original_filters:
            for k in original_filters:
                excluded_fields.update(FILTER_TO_DIFF_FIELDS.get(k, []))

        sample = products  # Use ALL products for accurate fallback analysis
        best_field = None
        best_score = -1
        best_values = {}
        best_filter_overrides = {}

        # Augment DIFF_FIELDS with category-specific tier1 fields
        from config.category_columns import get_tier1_config
        all_fields = list(DIFF_FIELDS)
        pool_categories = set(p.metadata.get('category', '') for p in products)
        tier1_fields = set()
        for cat in pool_categories:
            for field_name, display_label in get_tier1_config(cat).items():
                tier1_fields.add(field_name)
                if field_name not in all_fields:
                    all_fields.append(field_name)
                if field_name not in FIELD_DESCRIPTIONS:
                    FIELD_DESCRIPTIONS[field_name] = display_label

        # Fields that are never useful for narrowing
        always_exclude = {
            'cable_length_raw',  # Raw field — use length_ft/length_m instead
            'standards',         # Long multi-line IEEE strings, always unique per product
            'color',             # Cosmetic — rarely helps buying decisions
            'material',          # Cosmetic
            'Accent_Color',      # Cosmetic
            'connector_plating', # Cosmetic
        }

        for f in all_fields:
            if f in excluded_fields or f in always_exclude:
                continue
            value_counts = {}
            for p in sample:
                val = p.metadata.get(f)
                if val is None or str(val).lower() in ('nan', 'none', ''):
                    continue
                if isinstance(val, list) and len(val) >= 2:
                    # Treat list as a pair (e.g., connectors ['USB-C', 'HDMI'] → "USB-C to HDMI")
                    v_str = f"{_clean_value(str(val[0]))} to {_clean_value(str(val[1]))}"
                    if v_str:
                        value_counts[v_str] = value_counts.get(v_str, 0) + 1
                elif isinstance(val, list):
                    for v in val:
                        v_str = _clean_value(v)
                        if v_str:
                            value_counts[v_str] = value_counts.get(v_str, 0) + 1
                else:
                    v_str = _clean_value(val)
                    if v_str:
                        value_counts[v_str] = value_counts.get(v_str, 0) + 1

            # Merge near-duplicate values (e.g. "10Gbps" vs "10 Gbit/s")
            value_counts, filter_overrides = _merge_similar_values(value_counts)

            if len(value_counts) < 2:
                continue  # Need at least 2 distinct values to differentiate

            # Score: distinct values * coverage * balance * grouping quality
            # Balance penalizes fields where one value dominates (e.g., Audio on 10/12 HDMI cables)
            total = sum(value_counts.values())
            coverage = total / len(sample)
            max_share = max(value_counts.values()) / total
            balance = 1.0 - max_share  # 0.0 if one value = 100%, 0.5 if 50/50
            # Cap distinct count to prevent free-text fields from dominating
            effective_distinct = min(len(value_counts), 6)
            # Penalize fields where most values are singletons (poor for narrowing)
            avg_per_group = total / len(value_counts)
            grouping_factor = min(avg_per_group / 3.0, 1.0)
            tier1_boost = 1.3 if f in tier1_fields else 1.0
            score = effective_distinct * coverage * (balance + 0.1) * tier1_boost * grouping_factor
            if score > best_score:
                best_score = score
                best_field = f
                best_values = value_counts
                best_filter_overrides = filter_overrides

        if not best_field:
            return None  # No differentiating field found

        # Build options from top 6 values to minimize "Other" bucket
        top_values = sorted(best_values.items(), key=lambda x: -x[1])[:6]
        # Only use pair_eq when the field actually stores list values (e.g., connectors)
        first_meta = next((p.metadata.get(best_field) for p in products
                           if p.metadata.get(best_field) is not None
                           and str(p.metadata.get(best_field)).lower() not in ('nan', 'none', '')), None)
        is_paired = isinstance(first_meta, list) and len(first_meta) >= 2
        # Use eq for numeric fields to avoid substring matching (e.g., "16.4" matching "164.0")
        NUMERIC_FIELDS = {
            'length_ft', 'length_m', 'port_count', 'hub_ports', 'kvm_ports',
            'power_delivery_watts', 'max_refresh_rate', 'screen_size_inches',
            'dock_num_displays', 'bay_count', 'num_drives', 'u_height',
        }
        if is_paired:
            filter_op = "pair_eq"
        elif best_field in NUMERIC_FIELDS:
            filter_op = "eq"
        else:
            filter_op = "contains"
        options = []
        option_filters = []
        for val, count in top_values:
            label = f"{val} ({count} product{'s' if count != 1 else ''})"
            options.append(label)
            if val in best_filter_overrides:
                override = best_filter_overrides[val]
                option_filters.append({"field": best_field, "op": override["op"], "value": override["value"]})
            else:
                option_filters.append({"field": best_field, "op": filter_op, "value": val})

        options.append("Not sure / show me all")
        option_filters.append(None)

        desc = FIELD_DESCRIPTIONS.get(best_field, best_field)
        if is_fallback:
            intro = (f"We don't have an exact match, but here are {len(products)} "
                     f"similar options. Let me help find the closest fit.")
        else:
            intro = f"I found {len(products)} products. Let me help narrow it down."
        return {
            'attribute': best_field,
            'question': f"Which {desc.lower()} are you looking for?",
            'intro': intro,
            'options': options,
            'option_filters': option_filters,
        }

    def _resolve_meta_value(self, product, field: str, value: str):
        """
        Get metadata value for a field, trying aliases as fallback.
        """
        # Try primary field
        meta_val = product.metadata.get(field)
        if meta_val is not None:
            return meta_val

        # Try known aliases
        for alias in self.FIELD_ALIASES.get(field, []):
            meta_val = product.metadata.get(alias)
            if meta_val is not None:
                return meta_val

        return None

    def _apply_filter(self, products: list, filter_spec: dict) -> list:
        """Apply a filter spec to a product pool."""
        field = filter_spec.get('field', '')
        op = filter_spec.get('op', 'eq')
        value = filter_spec.get('value', '')

        # in_list uses product_number attribute directly, not metadata
        if op == 'in_list':
            allowed = set(str(value).split(','))
            return [p for p in products if hasattr(p, 'product_number') and p.product_number in allowed]

        result = []
        for p in products:

            meta_val = self._resolve_meta_value(p, field, value)
            if meta_val is None:
                continue

            if op == 'eq':
                # Try numeric comparison first (handles "10" vs "10.0", "9.84" ≈ "10")
                try:
                    tolerance = filter_spec.get('tolerance', 0.5)
                    if abs(float(meta_val) - float(value)) < tolerance:
                        result.append(p)
                        continue
                except (ValueError, TypeError):
                    pass
                # Fall back to string comparison for non-numeric fields
                if str(meta_val).lower() == str(value).lower():
                    result.append(p)
            elif op == 'pair_eq':
                # Match paired list values (e.g., connectors ['USB-C', 'HDMI'] → "USB-C to HDMI")
                if isinstance(meta_val, list) and len(meta_val) >= 2:
                    pair = f"{_clean_value(str(meta_val[0]))} to {_clean_value(str(meta_val[1]))}"
                    if pair.lower() == str(value).lower():
                        result.append(p)
            elif op == 'contains':
                if isinstance(meta_val, list):
                    if any(str(value).lower() in str(v).lower() for v in meta_val):
                        result.append(p)
                elif str(value).lower() in str(meta_val).lower():
                    result.append(p)
            elif op == 'gte':
                try:
                    if float(meta_val) >= float(value):
                        result.append(p)
                except (ValueError, TypeError):
                    pass
            elif op == 'lte':
                try:
                    if float(meta_val) <= float(value):
                        result.append(p)
                except (ValueError, TypeError):
                    pass
            elif op == 'in_range':
                try:
                    parts = str(value).split(',')
                    low, high = float(parts[0]), float(parts[1])
                    if low <= float(meta_val) <= high:
                        result.append(p)
                except (ValueError, TypeError, IndexError):
                    pass

        return result
