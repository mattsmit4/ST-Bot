"""
Consultative Response Builder - Generate personalized, question-answering responses.

Instead of just listing products, this generates responses that:
1. Directly answer any questions the user asked
2. Make clear recommendations with reasoning
3. Explain how products meet their specific requirements
4. Note any limitations or caveats

Used for complex queries where users need consultation, not just a product list.
"""

import os
import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from core.models import Product, SearchFilters
from core.openai_client import get_openai_client
from core.api_retry import RetryHandler, DEFAULT_OPENAI_RETRY
from llm.requirements_analyzer import UserRequirements
from config.category_columns import get_tier2_columns
from config.unit_converter import format_measurement


# Cable type to speed mapping (Gbps)
CABLE_SPEED_MAP = {
    'Cat5': 0.1,    # 100 Mbps
    'Cat5e': 1,     # 1 Gbps
    'Cat6': 1,      # 1 Gbps (10Gbps at short distances)
    'Cat6a': 10,    # 10 Gbps
    'Cat7': 10,     # 10 Gbps
    'Cat8': 25,     # 25-40 Gbps
}


def _parse_length_to_feet(length_str: str) -> Optional[float]:
    """Parse a length string to feet.

    Args:
        length_str: Length string like "100 ft", "30m", "25 feet"

    Returns:
        Length in feet, or None if unparseable
    """
    if not length_str:
        return None

    length_str = str(length_str).lower().strip()

    # Try to extract number and unit
    match = re.match(r'([\d.]+)\s*(ft|feet|foot|m|meters?|cm|centimeters?|in|inch|inches)?', length_str)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2) or 'ft'  # Default to feet

    # Convert to feet
    if unit in ('m', 'meter', 'meters'):
        return value * 3.28084
    elif unit in ('cm', 'centimeter', 'centimeters'):
        return value * 0.0328084
    elif unit in ('in', 'inch', 'inches'):
        return value / 12
    else:
        return value  # Already in feet


# System prompt for consultative response generation
CONSULTATIVE_PROMPT = """You are a helpful product consultant for StarTech.com. Generate a consultative response that directly addresses the customer's needs.

## Response Structure

1. **Answer Questions First** - If the user asked direct questions (like "will one dock handle all of this?"), answer with YES/NO plus brief explanation at the start.

2. **Recommend with Reasoning** - Present products using this EXACT format:

   **Product 1:** SKU (Category) — Recommended
   - 4-5 bullet points showing how it meets requirements

   **Product 2:** SKU (Category) — One sentence on how it differs from Product 1.

   **Product 3:** SKU (Category) — One sentence on how it differs from Product 1.

   If only ONE product is provided, skip Products 2-3. Do NOT create an "Alternative Options" section.

3. **Address Limitations** - If there are caveats (like downstream charging vs host charging), explain briefly.

4. **Close with Follow-up** - End with "Want full specs on any of these?" or similar.

## Example Output

**Product 1:** DK31C3MNCR (USB-C Dock) — Recommended
- Supports dual 4K monitors at 60Hz via HDMI and DisplayPort
- 60W Power Delivery to charge your MacBook Pro
- 7 USB ports for connecting peripherals
- Gigabit Ethernet for wired connectivity

**Product 2:** 129N-USBC-KVM-DOCK (KVM Dock) — Dual 4K with 90W PD, plus KVM switching between two computers.

**Product 3:** DK31C4DPPD (USB-C Dock) — Supports up to 4 monitors but only has DisplayPort outputs.

## Formatting Rules

- Use **bold** for product SKUs and section headers
- Use bullet points (- ) for Product 1 features only
- Products 2-3 must be ONE line each — no bullet points
- ALWAYS include the product Category in parentheses after the SKU using readable names (e.g., "USB-C Dock", "Hub", "Adapter", "Cable", "Rack"). Match the category from product data — do NOT call everything a "dock".
- Be concise but helpful - this is customer support, not a sales pitch
- Match requirements to product specs explicitly (e.g., "4 USB-A ports covers your hard drives, SD reader, keyboard")

## Important Context

- If host_type is "desktop", they DON'T need charging for their computer (desktops have their own power)
- "Power Delivery" on docks typically charges the HOST laptop - for downstream device charging (iPad, phone), mention they can use the dock's USB-C ports
- Mac compatibility: Most USB-C/Thunderbolt docks work with M1/M2 Macs via USB-C or Thunderbolt

## CRITICAL - No Hallucinations

You can ONLY describe product features that are explicitly listed in the product data above.
- Use EXACT port/connector names from the data. If "Video outputs: HDMI" → say HDMI, NOT DVI.
- If "Ethernet: Yes" is NOT listed for a product, do NOT claim it has Ethernet.
- If a product's "video_outputs" shows VGA only, do NOT claim it supports HDMI.
- Cross-check EVERY spec you write against that product's data section above.
- If no products match the user's requirements (e.g., they want HDMI but products are VGA), be honest:
  "Unfortunately, the products I found support VGA, not the HDMI you requested."
- NEVER make up capabilities, ports, or features that aren't in the product data.

## Privacy Screens - Size Matching is CRITICAL

For privacy screen queries, you MUST match the product's screen_size to what the user requested:
- The "screen_size" field in product data shows the ACTUAL screen size (e.g., "13.5 inch", "15.6 inch")
- SKU patterns: 135CT = 13.5", 156W = 15.6", 24MAM = 24"
- If no products match the requested screen size exactly, be HONEST: "I found privacy screens, but none in your requested 15.6" size. The closest options are..."
- NEVER claim a 13.5" screen fits a 15.6" laptop - this is physically impossible

## Handling Product Limitations

When products don't fully meet requirements:

1. **Length Mismatch** - If a "LENGTH LIMITATION" section appears in the context:
   - Acknowledge the limitation upfront: "Our longest [type] cables are X ft"
   - Suggest practical alternatives:
     - Use cable couplers/joiners to connect multiple cables
     - Consider bulk cable with field-terminated ends
     - For ethernet: up to 328ft (100m) is within spec for Cat6a
   - Still recommend the longest available option
   - DO NOT pretend the products meet the length requirement

2. **Speed/Feature Mismatch** - If products don't match speed requirements:
   - Be clear about what speed the products support
   - Explain trade-offs if any

## Multi-Product Solutions (Complete Kits)

When the context shows "COMPLETE SOLUTION - Multiple Products Needed", the customer needs products from different categories working together. Structure your response as:

1. **Solution Overview** - Start by explaining what they need and why (2-3 sentences)

2. **For each component role:**
   - What function it serves in their setup
   - Recommended product with SKU
   - One sentence explaining why it's the right choice

3. **Complete Kit Summary** - At the end, list all recommended products together.

   ⚠️ QUANTITY IS CRITICAL - READ CAREFULLY:
   - Look at each component's "Quantity needed:" value in the context above
   - If it says "Quantity needed: 2", you MUST write "2x [SKU]"
   - If it says "Quantity needed: 1", you MUST write "1x [SKU]"
   - DO NOT default to 1x for cables - always check the Quantity needed value

   Example: If context shows "Connectivity" with "Quantity needed: 2"
   Then write: "2x HDMM15 (15ft HDMI cables)" NOT "1x HDMM15"

   ```
   **Complete Kit:**
   - 1x [Switch SKU] - [brief purpose]
   - 1x [Adapter SKU] - [brief purpose]
   - 2x [Cable SKU] - [brief purpose]  ← quantity matches "Quantity needed: 2"
   ```

4. If any component has no matching products, suggest what to look for elsewhere

Example for a video switching setup:
```
To switch between your laptop and PS5 on a single 4K monitor, you'll need three components working together: a video switch for source selection, an adapter to convert to DisplayPort, and cables for connectivity.

**Video Switching**
→ **VS221HD20** - 2-port HDMI switch lets you toggle between devices with a button

**Signal Conversion**
→ **HD2DP** - Active HDMI to DisplayPort adapter for your monitor's input

**Connectivity**
→ **HDMM15** - 15ft HDMI cables to reach from your devices to the switch

**Complete Kit:**
- 1x VS221HD20 (HDMI switch)
- 1x HD2DP (HDMI→DP adapter)
- 2x HDMM15 (15ft HDMI cables)
```

## Response Format

Respond with ONLY the response text (no JSON wrapping). Use markdown formatting."""


@dataclass
class ProductSummary:
    """Simplified product info for LLM context."""
    sku: str
    name: str
    monitor_count: Optional[int]
    usb_a_ports: int
    usb_c_ports: int
    power_delivery: Optional[str]
    has_ethernet: bool
    video_outputs: List[str]
    features: List[str]
    # Cable-specific fields
    cable_length_ft: Optional[float] = None
    cable_type: Optional[str] = None  # "Cat6a", "Cat5e", etc.
    is_shielded: bool = False
    network_speed_gbps: Optional[float] = None
    # Product category (hub, dock, adapter, cable, etc.)
    category: str = ""
    # Category-specific fields from config
    category_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sku": self.sku,
            "name": self.name,
            "monitor_count": self.monitor_count,
            "usb_a_ports": self.usb_a_ports,
            "usb_c_ports": self.usb_c_ports,
            "power_delivery": self.power_delivery,
            "has_ethernet": self.has_ethernet,
            "video_outputs": self.video_outputs,
            "features": self.features,
            "cable_length_ft": self.cable_length_ft,
            "cable_type": self.cable_type,
            "is_shielded": self.is_shielded,
            "network_speed_gbps": self.network_speed_gbps,
        }
        if self.category_data:
            d["category_specs"] = self.category_data
        return d


def extract_product_summary(product: Product) -> ProductSummary:
    """Extract key specs from a product for LLM context."""
    meta = product.metadata
    conntype = meta.get('conn_type', '')

    # Monitor count
    num_displays = meta.get('dock_num_displays')
    if num_displays:
        try:
            num_displays = int(float(num_displays))
        except (ValueError, TypeError):
            num_displays = None

    # USB port counts from conn_type
    usb_a_count = 0
    usb_c_count = 0
    if conntype:
        # Count USB-A ports
        usb_a_matches = re.findall(r'(\d+)\s*x?\s*USB[\s-]?(?:3\.[012]|2\.0)?\s*(?:Type[\s-]?)?A', conntype, re.I)
        if usb_a_matches:
            usb_a_count = sum(int(m) for m in usb_a_matches)
        elif 'USB-A' in conntype or 'USB Type-A' in conntype:
            # Count occurrences
            usb_a_count = conntype.upper().count('USB-A') + conntype.count('Type-A')

        # Count USB-C ports (excluding host connection)
        usb_c_matches = re.findall(r'(\d+)\s*x?\s*USB[\s-]?(?:3\.[012]|2\.0)?\s*(?:Type[\s-]?)?C', conntype, re.I)
        if usb_c_matches:
            usb_c_count = sum(int(m) for m in usb_c_matches)

    # Power Delivery
    pd_wattage = meta.get('power_delivery') or meta.get('hub_power_delivery')

    # Ethernet
    network_speed = meta.get('network_speed')
    has_ethernet = bool(network_speed) or 'RJ-45' in conntype or 'Ethernet' in conntype

    # Video outputs
    video_outputs = []
    for video_type in ['HDMI', 'DisplayPort', 'VGA', 'DVI']:
        if video_type.lower() in conntype.lower():
            # Try to count them
            count_match = re.search(rf'(\d+)\s*x?\s*{video_type}', conntype, re.I)
            if count_match:
                count = int(count_match.group(1))
                video_outputs.append(f"{count}x {video_type}")
            else:
                video_outputs.append(video_type)

    # Features
    features = meta.get('features', [])
    if isinstance(features, str):
        features = [features]

    # Cable-specific extraction
    cable_length_ft = None
    length_str = meta.get('length') or meta.get('cable_length_raw')
    if length_str:
        cable_length_ft = _parse_length_to_feet(length_str)

    cable_type = meta.get('network_rating')  # "Cat6a", "Cat5e", etc.

    # Check if shielded
    cable_shield = str(meta.get('cable_shield', '')).lower()
    is_shielded = 'shielded' in cable_shield or 'stp' in cable_shield or 's/ftp' in cable_shield

    # Network speed from cable type
    network_speed_gbps = CABLE_SPEED_MAP.get(cable_type) if cable_type else None

    # Extract category-specific tier2 data from config
    category = meta.get('category', '')
    category_data = {}
    for field in get_tier2_columns(category):
        value = meta.get(field)
        if value and str(value).lower() not in ('nan', 'none', ''):
            # Use readable field name (replace underscores, limit to key specs)
            category_data[field] = format_measurement(str(value), field)
    # Limit to avoid token bloat
    if len(category_data) > 20:
        category_data = dict(list(category_data.items())[:20])

    return ProductSummary(
        sku=product.product_number,
        name=meta.get('name', product.product_number),
        category=category,
        monitor_count=num_displays,
        usb_a_ports=usb_a_count,
        usb_c_ports=usb_c_count,
        power_delivery=str(pd_wattage) if pd_wattage else None,
        has_ethernet=has_ethernet,
        video_outputs=video_outputs,
        features=features[:5] if features else [],  # Limit features
        cable_length_ft=cable_length_ft,
        cable_type=cable_type,
        is_shielded=is_shielded,
        category_data=category_data if category_data else None,
        network_speed_gbps=network_speed_gbps,
    )


class ConsultativeResponseBuilder:
    """
    Builds personalized, consultative responses using LLM.

    Takes user requirements + matched products and generates a response
    that directly addresses what the user needs.
    """

    def __init__(self):
        self.client = get_openai_client()
        self.retry_handler = RetryHandler(DEFAULT_OPENAI_RETRY)

    def build_response(
        self,
        query: str,
        requirements: UserRequirements,
        products: List[Product],
        filters: Optional[SearchFilters] = None,
        component_products: Optional[Dict[str, List[Product]]] = None,
        debug_lines: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Build a consultative response for the user's query.

        Args:
            query: Original user query
            requirements: Extracted user requirements
            products: Top products to recommend (1-3)
            filters: Search filters used (optional context)
            component_products: Products organized by solution component role
            debug_lines: Optional debug output

        Returns:
            Response string or None on failure
        """
        if not self.client:
            if debug_lines is not None:
                debug_lines.append("⚠️ CONSULTATIVE: No OpenAI client")
            return None

        # For multi-product solutions, we may have no "primary" products but have component products
        if not products and not component_products:
            if debug_lines is not None:
                debug_lines.append("⚠️ CONSULTATIVE: No products to recommend")
            return None

        try:
            # Extract product summaries for primary products
            product_summaries = [extract_product_summary(p) for p in products[:3]] if products else []

            # Extract product summaries for component products
            component_summaries = None
            if component_products:
                component_summaries = {}
                for role, prods in component_products.items():
                    component_summaries[role] = [extract_product_summary(p) for p in prods[:2]]

            # Build context for LLM (pass filters for length mismatch detection)
            context = self._build_context(
                query, requirements, product_summaries, filters, component_summaries
            )

            if debug_lines is not None:
                if component_products:
                    total_prods = sum(len(p) for p in component_products.values())
                    debug_lines.append(f"🤖 CONSULTATIVE: Building multi-product response ({total_prods} products across {len(component_products)} components)")
                else:
                    debug_lines.append(f"🤖 CONSULTATIVE: Building response for {len(products)} products")

            result = self.retry_handler.execute(
                lambda: self._call_openai(context),
                fallback=None
            )

            return result

        except Exception as e:
            if debug_lines is not None:
                debug_lines.append(f"⚠️ CONSULTATIVE ERROR: {type(e).__name__}: {str(e)[:50]}")
            return None

    def _build_context(
        self,
        query: str,
        requirements: UserRequirements,
        products: List[ProductSummary],
        filters: Optional[SearchFilters] = None,
        component_summaries: Optional[Dict[str, List[ProductSummary]]] = None
    ) -> str:
        """Build the context string for the LLM."""
        parts = []

        # Original query
        parts.append(f"## Customer Query\n{query}")

        # Requirements
        parts.append("\n## Extracted Requirements")
        if requirements.host_device:
            parts.append(f"- Host device: {requirements.host_device} ({requirements.host_type})")
        if requirements.monitor_count:
            res_info = f" at {requirements.resolution}" if requirements.resolution else ""
            hz_info = f"/{requirements.refresh_rate}Hz" if requirements.refresh_rate else ""
            parts.append(f"- Monitors needed: {requirements.monitor_count}{res_info}{hz_info}")
        if requirements.peripherals:
            parts.append(f"- Peripherals to connect: {', '.join(requirements.peripherals)}")
        if requirements.charging_device:
            parts.append(f"- Charging needed: {requirements.charging_device} ({requirements.charging_type} charging)")
        if requirements.min_usb_a_ports:
            parts.append(f"- Minimum USB-A ports: {requirements.min_usb_a_ports}")
        if requirements.questions:
            parts.append(f"- Questions to answer: {'; '.join(requirements.questions)}")

        # Check for length mismatch (cable products)
        requested_length_ft = None
        if filters and filters.length:
            requested_length_ft = _parse_length_to_feet(filters.length)

        # Find max available length from products
        cable_products = [p for p in products if p.cable_length_ft]
        max_available_ft = max((p.cable_length_ft for p in cable_products), default=None) if cable_products else None

        # Detect length mismatch
        if requested_length_ft and max_available_ft and requested_length_ft > max_available_ft:
            parts.append(f"\n## ⚠️ LENGTH LIMITATION")
            parts.append(f"- User requested: {int(requested_length_ft)} ft")
            parts.append(f"- Longest available: {int(max_available_ft)} ft")
            parts.append(f"- Gap: {int(requested_length_ft - max_available_ft)} ft")
            parts.append("- MUST acknowledge this limitation in your response and suggest alternatives (couplers, bulk cable)")

        # Multi-product solution components
        if component_summaries and requirements.is_multi_product_solution:
            parts.append("\n## 🔧 COMPLETE SOLUTION - Multiple Products Needed")
            parts.append("This setup requires products from different categories working together.")
            parts.append("Present these as a complete kit the customer should purchase.\n")

            for role, role_products in component_summaries.items():
                # Find the matching component definition
                component = next(
                    (c for c in requirements.solution_components if c.role == role),
                    None
                )

                parts.append(f"\n### {role}")
                if component:
                    parts.append(f"**Purpose:** {component.reason}")
                    priority_label = {1: "Essential", 2: "Recommended", 3: "Optional"}.get(component.priority, "Recommended")
                    parts.append(f"**Priority:** {priority_label}")
                    parts.append(f"**Quantity needed:** {component.quantity}")

                if role_products:
                    for i, prod in enumerate(role_products, 1):
                        parts.append(f"\n**Option {i}: {prod.sku}**")
                        parts.append(f"- {prod.name}")
                        # Add key specs
                        if prod.cable_length_ft:
                            parts.append(f"- Length: {int(prod.cable_length_ft)} ft")
                        if prod.cable_type:
                            parts.append(f"- Type: {prod.cable_type}")
                        if prod.video_outputs:
                            parts.append(f"- Video: {', '.join(prod.video_outputs)}")
                else:
                    parts.append("*No matching products found for this component*")

            return "\n".join(parts)

        # Standard single-category products
        if products:
            parts.append("\n## Products to Recommend (ranked best to worst)")
            for i, prod in enumerate(products, 1):
                parts.append(f"\n### Product {i}: {prod.sku}")
                parts.append(f"- Name: {prod.name}")
                if prod.category:
                    parts.append(f"- Category: {prod.category.replace('_', ' ')}")

                # Cable-specific info
                if prod.cable_length_ft:
                    parts.append(f"- Length: {int(prod.cable_length_ft)} ft")
                if prod.cable_type:
                    parts.append(f"- Cable type: {prod.cable_type}")
                if prod.network_speed_gbps:
                    parts.append(f"- Speed: {prod.network_speed_gbps} Gbps")
                if prod.is_shielded:
                    parts.append(f"- Shielded: Yes (good for EMI protection)")

                # Dock/hub-specific info
                if prod.monitor_count:
                    parts.append(f"- Monitor support: {prod.monitor_count}")
                if prod.video_outputs:
                    parts.append(f"- Video outputs: {', '.join(prod.video_outputs)}")
                if prod.usb_a_ports:
                    parts.append(f"- USB-A ports: {prod.usb_a_ports}")
                if prod.usb_c_ports:
                    parts.append(f"- USB-C ports: {prod.usb_c_ports}")
                if prod.power_delivery:
                    parts.append(f"- Power Delivery: {prod.power_delivery}")
                if prod.has_ethernet:
                    parts.append(f"- Ethernet: Yes")
                if prod.features:
                    parts.append(f"- Features: {', '.join(prod.features)}")

                # Add category-specific specs from config
                if prod.category_data:
                    # Skip fields already shown above to avoid duplication
                    shown = {'dock_num_displays', 'power_delivery', 'hub_power_delivery',
                             'network_speed', 'cable_length_raw', 'network_rating',
                             'features', 'conn_type'}
                    extra = {k: v for k, v in prod.category_data.items() if k not in shown}
                    if extra:
                        specs_str = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in list(extra.items())[:10])
                        parts.append(f"- Specs: {specs_str}")

        return "\n".join(parts)

    def _call_openai(self, context: str) -> Optional[str]:
        """Make the OpenAI API call."""
        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-nano"),
            messages=[
                {"role": "system", "content": CONSULTATIVE_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.3,  # Slightly higher for more natural responses
            max_completion_tokens=1000,  # Increased for multi-product solutions
        )

        content = response.choices[0].message.content.strip()
        return content


# Module-level instance
_builder: Optional[ConsultativeResponseBuilder] = None


def build_consultative_response(
    query: str,
    requirements: UserRequirements,
    products: List[Product],
    filters: Optional[SearchFilters] = None,
    component_products: Optional[Dict[str, List[Product]]] = None,
    debug_lines: Optional[List[str]] = None
) -> Optional[str]:
    """
    Convenience function to build a consultative response.

    Args:
        query: Original user query
        requirements: Extracted user requirements
        products: Products to recommend
        filters: Optional search filters
        component_products: Products organized by solution component role
        debug_lines: Optional debug output

    Returns:
        Response string or None
    """
    global _builder
    if _builder is None:
        _builder = ConsultativeResponseBuilder()
    return _builder.build_response(
        query, requirements, products, filters, component_products, debug_lines
    )
