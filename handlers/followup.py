"""
Follow-up intent handlers.

Handles follow-up questions about products in context:
- Routing: dispatch to sub-handlers based on LLM interpretation
- Specs/details: answer specific product questions
- Compatibility: check feature support across products
- Educational: general technology questions
- Recommendations: suggest best product for use case

Refinement and search helpers are in followup_refinement.py.
"""

import os
import re
import logging
from handlers.base import BaseHandler, HandlerContext, HandlerResult
from handlers.followup_refinement import RefinementMixin

_logger = logging.getLogger(__name__)

from core.search.resolution import supports_4k
from config.patterns import PRICING_PATTERNS, WARRANTY_PATTERNS


class FollowupHandler(RefinementMixin, BaseHandler):
    """Handle follow-up questions about products in context."""

    def _get_targeted_products(self, ctx, interpretation, fallback_to_all=True):
        """Get products targeted by interpretation's product_indices."""
        if interpretation.product_indices:
            # Filter out None values (LLM sometimes returns [None] for unresolved refs)
            valid_indices = [i for i in interpretation.product_indices if isinstance(i, int)]
            if valid_indices:
                return [
                    ctx.context.current_products[i - 1]
                    for i in valid_indices
                    if 1 <= i <= len(ctx.context.current_products)
                ]
        return ctx.context.current_products if fallback_to_all else []

    def _generate_llm_response(self, ctx, interpretation, response_type, extra_context=None, fallback_to_all=True, query_override=None):
        """Generate an LLM response for targeted products with normalized query."""
        from llm.llm_response_generator import generate_response, ResponseType

        products = self._get_targeted_products(ctx, interpretation, fallback_to_all=fallback_to_all)
        if not products:
            return None

        # Use resolved query from interpreter when available (e.g., "Yes" → "USB port counts")
        query = query_override or ctx.query
        if interpretation.product_indices and products:
            for idx, prod in zip(interpretation.product_indices, products):
                query = re.sub(rf'#\s*{idx}\b', f'**{prod.product_number}**', query)
            if response_type == ResponseType.PRODUCT_DETAILS and len(products) == 1:
                query = f"Tell me about {products[0].product_number}"

        # Build context
        context = None
        if interpretation.product_indices:
            context = {
                'selected_from_total': len(ctx.context.current_products),
                'user_product_refs': [f'#{i}' for i in interpretation.product_indices],
            }
        if extra_context:
            context = {**(context or {}), **extra_context}

        return generate_response(products=products, query=query, response_type=response_type, context=context)

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Sub-intent dispatch for narrowing/clarification responses
        meta = ctx.intent.meta_info or {}
        if meta.get('narrowing_response'):
            from handlers.narrowing import NarrowingResponseHandler
            return NarrowingResponseHandler().handle(ctx)
        if meta.get('clarification_response'):
            from handlers.clarification import ClarificationResponseHandler
            return ClarificationResponseHandler().handle(ctx)

        # Check for order/customer service requests -- redirect to support
        if self._is_order_request(ctx.query.lower()):
            return HandlerResult(
                response="I can't help with orders, returns, or account questions.\n\n"
                         "You can reach StarTech.com support at:\n\n"
                         "**Website:** www.startech.com/support\n"
                         "**Phone:** 1-800-265-1844"
            )

        # Check for pricing queries early -- not something we can help with
        if self._is_pricing_request(ctx.query.lower()):
            return HandlerResult(
                response="I don't have access to pricing information. "
                         "For product pricing and availability, please visit www.startech.com."
            )

        # Check for warranty queries — answer from product data if available,
        # redirect only when no products or no warranty data in context
        if self._is_warranty_request(ctx.query.lower()):
            has_warranty_data = ctx.context.current_products and any(
                p.metadata.get('warranty') for p in ctx.context.current_products
            )
            if not has_warranty_data:
                return HandlerResult(
                    response="For warranty details, return policies, and RMA requests, "
                             "please visit www.startech.com/warranty or contact StarTech.com support directly. "
                             "Is there anything else I can help you find?"
                )

        # Check for refinement (e.g., "I need 3 foot cables")
        if ctx.intent.meta_info and ctx.intent.meta_info.get('refinement'):
            return self._handle_refinement(ctx)

        # Check if we have products in context
        if not ctx.context.current_products:
            # Check if user referenced a product that doesn't exist
            if ctx.intent.meta_info and ctx.intent.meta_info.get('needs_product_clarification'):
                return HandlerResult(
                    response="I don't have a specific product in our conversation yet. "
                             "Which product are you asking about? You can:\n\n"
                             "- Search for products (e.g., \"HDMI cables\" or \"USB-C dock\")\n"
                             "- Enter a product SKU if you have one"
                )
            return HandlerResult(
                response="I can help you find StarTech.com products. What are you looking for?"
            )

        # Use LLM interpreter to understand what user wants
        llm_result = self._try_llm_interpretation(ctx)
        if llm_result:
            return llm_result

        # Fallback: show product specs
        return self._show_product_specs(ctx)

    ORDER_PATTERNS = [
        r'\b(?:order|purchase|buy|buying)\b',
        r'\badd\s+to\s+cart\b',
        r'\bship(?:ping|ped|s)?\s+(?:to|it|this|them)\b',
        r'\bdeliver(?:y|ed)?\s+(?:to|it|this|them|the|my|our)\b',
        r'\bplace\s+(?:a\s+)?(?:order|purchase)\b',
        r'\binvoice\b',
        r'\bquote\s+(?:for|on|me)\b',
        r'\bemail\s+(?:this|the|a)\b',
        r'\bsend\s+(?:this|the|a)\s+(?:list|quote|info)\b',
    ]

    def _is_order_request(self, query: str) -> bool:
        """Check if query is an order/purchase/shipping request."""
        for pattern in self.ORDER_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _is_pricing_request(self, query: str) -> bool:
        """Check if query contains pricing-related terms."""
        for pattern in PRICING_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _is_warranty_request(self, query: str) -> bool:
        """Check if query contains warranty-related terms."""
        for pattern in WARRANTY_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    # _handle_refinement and all _refine_by_* methods are inherited from
    # RefinementMixin (handlers/followup_refinement.py)

    def _try_llm_interpretation(self, ctx: HandlerContext) -> HandlerResult | None:
        """
        Use LLM to interpret what user wants from a followup question.

        Before showing all product specs, use LLM to understand if user is:
        - Filtering (e.g., "2 meter ones" -> show only 2m products)
        - Asking about specific product (e.g., "the first one")
        - Comparing (e.g., "which is better")
        - etc.

        Args:
            ctx: Handler context

        Returns:
            HandlerResult if LLM interpretation succeeds, None otherwise
        """
        try:
            from llm.llm_followup_interpreter import (
                LLMFollowupInterpreter,
                FollowupAction,
                filter_products_by_criteria,
                format_filtered_products
            )

            interpreter = LLMFollowupInterpreter()

            last_msg = ctx.context.get_last_message(role='assistant')
            last_bot_response = last_msg.content if last_msg else None

            # Gather recent bot responses for pronoun resolution across turns
            recent_bot_responses = None
            history = ctx.context.get_conversation_history(limit=4, role='assistant')
            if len(history) > 1:
                recent_bot_responses = [m.content for m in history[1:] if m.content]

            interpretation = interpreter.interpret(
                ctx.query,
                ctx.context.current_products,
                ctx.debug_lines if ctx.debug_mode else None,
                last_bot_response=last_bot_response,
                recent_bot_responses=recent_bot_responses,
            )

            if not interpretation:
                return None

            ctx.add_debug(f"LLM FOLLOWUP: action={interpretation.action.value}")

            # Hardware-agnostic OS compatibility — intercept before action dispatch
            # so it catches both SPECS and COMPATIBILITY actions
            HARDWARE_AGNOSTIC = {
                'privacy_screen', 'cable', 'network_cable', 'fiber_cable',
                'mount', 'display_mount', 'rack', 'cable_organizer',
                'laptop_lock', 'video_splitter', 'video_switch',
                'ethernet_switch', 'kvm_switch', 'kvm_extender',
            }
            OS_TERMS = {'mac', 'macos', 'apple', 'windows', 'linux', 'ubuntu', 'chromeos'}
            if interpretation.action in (FollowupAction.SPECS, FollowupAction.COMPATIBILITY):
                crit = interpretation.criteria or {}
                question = str(crit.get('question', '') or crit.get('feature', '')).lower()
                # Also check non-standard keys
                if not question:
                    question = ' '.join(str(v) for v in crit.values()).lower()
                if any(t in question for t in OS_TERMS):
                    products_check = self._get_targeted_products(ctx, interpretation)
                    if products_check:
                        categories = {p.metadata.get('category', '') for p in products_check}
                        if categories & HARDWARE_AGNOSTIC:
                            cat_raw = list(categories & HARDWARE_AGNOSTIC)[0]
                            cat_name = cat_raw.replace('_', ' ')
                            # Proper pluralization
                            cat_plural = cat_name + 'es' if cat_name.endswith('ch') or cat_name.endswith('sh') else cat_name + 's'
                            ctx.add_debug(f"HARDWARE-AGNOSTIC: {cat_name} + OS question")
                            return HandlerResult(
                                response=f"Yes — {cat_plural} are hardware accessories that work with any operating system "
                                         f"(Mac, Windows, Linux). They connect to the hardware, not the OS."
                            )

            # Handle FILTER action - this is the key fix for "2 meter ones"
            if interpretation.action == FollowupAction.FILTER:
                # For relative length (shorter/longer), ALWAYS do a new search
                # User wants NEW products, not just filtered existing ones
                # "I need them longer" means show products longer than ALL current ones
                if interpretation.criteria and 'length_comparison' in interpretation.criteria:
                    direction = interpretation.criteria['length_comparison']
                    ctx.add_debug(f"RELATIVE LENGTH: '{direction}' - searching for new products")
                    return self._refine_by_relative_length(ctx, direction)

                # Comparative requests with no extractable criteria ("more ports", "faster")
                # can't be filtered — give a helpful response instead of crashing
                if not interpretation.criteria:
                    comparative_words = ['more', 'faster', 'bigger', 'better', 'higher', 'greater', 'extra', 'fewer', 'less']
                    if any(w in ctx.query.lower() for w in comparative_words):
                        count = len(ctx.context.current_products)
                        if count == 1:
                            msg = "This is the only product matching your search."
                        else:
                            msg = f"The {count} products shown are all we have matching your search."
                        return HandlerResult(
                            response=f"{msg} Would you like to broaden the search or try different specs?"
                        )

                # For absolute filters (like "6ft"), save and filter existing products
                criteria = interpretation.criteria or {}
                ctx.context.save_products_before_filter()

                filtered = filter_products_by_criteria(
                    ctx.context.current_products,
                    criteria
                )

                ctx.add_debug(f"FILTER: {len(filtered)}/{len(ctx.context.current_products)} match")

                # If all products match a feature filter, tell the user they already have it
                if filtered and len(filtered) == len(ctx.context.current_products):
                    feature = criteria.get('feature', '')
                    if feature:
                        ctx.add_debug(f"FILTER: All products already match '{feature}'")
                        count = len(filtered)
                        if count == 1:
                            msg = f"The product shown already supports **{feature}**."
                        else:
                            msg = f"All {count} products shown already support **{feature}**."
                        return HandlerResult(
                            response=f"{msg} Would you like to narrow by a different spec?"
                        )

                # If no products match, try new search
                if not filtered:
                    search_result = self._auto_search_for_criteria(ctx, criteria, filter_reason="0 matches")
                    if search_result:
                        return search_result
                # If all match but not a feature filter, also try new search
                elif len(filtered) == len(ctx.context.current_products):
                    reason = f"all {len(filtered)} matched (filter ineffective)"
                    search_result = self._auto_search_for_criteria(ctx, criteria, filter_reason=reason)
                    if search_result:
                        return search_result

                response = format_filtered_products(
                    filtered,
                    criteria,
                    len(ctx.context.current_products)
                )

                if response:
                    return HandlerResult(
                        response=response,
                        products_to_set=filtered if filtered else None
                    )
                # If all products match, fall through to normal handling

            # Handle out-of-range product references (e.g., "the second one" when only 1 exists)
            if interpretation.product_indices:
                max_idx = len(ctx.context.current_products)
                out_of_range = [i for i in interpretation.product_indices if isinstance(i, int) and i > max_idx]
                if out_of_range and max_idx > 0:
                    if max_idx == 1:
                        return HandlerResult(
                            response=f"There's only 1 product currently shown (**{ctx.context.current_products[0].product_number}**). "
                                     f"Would you like me to search for more options?"
                        )
                    else:
                        return HandlerResult(
                            response=f"I only have {max_idx} products shown (numbered 1-{max_idx}). "
                                     f"Which one are you asking about?"
                        )

            # Handle SPECS action with length question - "which one is longest?"
            if interpretation.action == FollowupAction.SPECS:
                crit = interpretation.criteria or {}
                question_type = crit.get('question', '') or crit.get('feature', '')
                # Fallback: if LLM put the question as a key (e.g., {'operating temperature': True})
                # instead of {'question': 'operating temperature'}, use the first key
                if not question_type and crit:
                    first_key = next(iter(crit))
                    if first_key not in ('question', 'feature', 'length_ft', 'length_m',
                                         'length_comparison', 'color', 'connector', 'category_hint',
                                         'use_case', 'topic'):
                        question_type = first_key
                if question_type == 'length':
                    ctx.add_debug(f"LENGTH QUESTION: Answering 'which is longest/shortest?'")

                    # Analyze current products for length
                    products_with_length = []
                    for i, prod in enumerate(ctx.context.current_products, 1):
                        length_ft = prod.metadata.get('length_ft', 0)
                        length_m = prod.metadata.get('length_m', 0)
                        # Normalize to feet for comparison
                        length = length_ft if length_ft else (length_m * 3.28084 if length_m else 0)
                        name = prod.metadata.get('name', prod.product_number)
                        products_with_length.append((i, prod.product_number, name, length))

                    if not any(p[3] for p in products_with_length):
                        ctx.add_debug("LENGTH QUESTION: No cable length data, falling through to LLM specs")
                        # Fall through to normal specs handling — LLM can use other dimensional data
                    else:
                        # Sort by length descending to find longest
                        products_with_length.sort(key=lambda x: x[3], reverse=True)

                        # Check if all same length
                        lengths = [p[3] for p in products_with_length if p[3] > 0]
                        if len(set(lengths)) == 1:
                            length_str = f"{lengths[0]:.1f} ft" if lengths[0] else "unknown"
                            return HandlerResult(
                                response=f"All {len(products_with_length)} products are the same length ({length_str})."
                            )

                        # Report longest
                        longest = products_with_length[0]
                        shortest = products_with_length[-1]
                        response_parts = [f"**#{longest[0]} {longest[1]}** is the longest at {longest[3]:.1f} ft."]
                        if shortest[3] > 0 and shortest[3] != longest[3]:
                            response_parts.append(f"**#{shortest[0]} {shortest[1]}** is the shortest at {shortest[3]:.1f} ft.")
                        response_parts.append("\nWould you like more details on any of these?")

                        return HandlerResult(response="\n".join(response_parts))

                # General specs question - use LLM to answer
                # SPECS action always uses SPECIFIC_QUESTION (tier2).
                # DETAILS action has its own handler for full product details (tier3).
                from llm.llm_response_generator import ResponseType
                rtype = ResponseType.SPECIFIC_QUESTION
                # Pass resolved question so response generator knows what to answer
                # (e.g., user said "Yes" but interpreter resolved it to "USB port counts")
                resolved_query = f"What are the {question_type} for each product?" if question_type and question_type not in ('all', 'length') else None
                response = self._generate_llm_response(ctx, interpretation, rtype, query_override=resolved_query)
                if response:
                    return HandlerResult(response=response)

            # Handle RECALL action - retrieve products from earlier in conversation
            if interpretation.action == FollowupAction.RECALL:
                category_hint = (interpretation.criteria or {}).get('category_hint', '')
                ctx.add_debug(f"RECALL: Looking for '{category_hint}' in history")

                # Search history for matching products
                history_entry = ctx.context.find_in_history(category_hint)

                if history_entry:
                    ctx.add_debug(f"RECALL: Found {len(history_entry.products)} products from '{history_entry.query}'")

                    # Build response
                    parts = [f"Here are the {category_hint}s I showed you earlier:", ""]
                    for i, prod in enumerate(history_entry.products, 1):
                        name = prod.metadata.get('name', prod.product_number)
                        parts.append(f"{i}. **{prod.product_number}** - {name}")
                    parts.append("")
                    parts.append("Would you like more details on any of these?")

                    return HandlerResult(
                        response="\n".join(parts),
                        products_to_set=history_entry.products
                    )
                else:
                    # History not found - offer to search fresh
                    ctx.add_debug(f"RECALL: Not found in history, offering new search")

                    # Get available categories for helpful message
                    available = ctx.context.get_history_categories()
                    if available:
                        available_str = ", ".join(set(available))
                        return HandlerResult(
                            response=f"I don't have {category_hint}s in my recent history. "
                                     f"I remember showing you: {available_str}. "
                                     f"Would you like me to search for {category_hint}s?"
                        )
                    else:
                        return HandlerResult(
                            response=f"I don't have a record of showing you {category_hint}s earlier. "
                                     f"Would you like me to search for them now?"
                        )

            # Handle COMPATIBILITY action - check feature for ALL products
            # (Hardware-agnostic OS check already handled above before action dispatch)
            if interpretation.action == FollowupAction.COMPATIBILITY:
                feature = (interpretation.criteria or {}).get('feature', 'the requested feature')
                ctx.add_debug(f"COMPATIBILITY CHECK: {feature}")

                from llm.llm_response_generator import generate_response, ResponseType
                products_for_check = self._get_targeted_products(ctx, interpretation)

                response = generate_response(
                    products=products_for_check,
                    query=ctx.query,
                    response_type=ResponseType.COMPATIBILITY,
                    context={'feature': feature}
                )
                if response:
                    return HandlerResult(response=response)

            # Handle EDUCATIONAL action - general knowledge questions
            if interpretation.action == FollowupAction.EDUCATIONAL:
                topic = (interpretation.criteria or {}).get('topic', 'your question')
                ctx.add_debug(f"EDUCATIONAL: {topic}")

                # Use LLM to answer the educational question
                answer = self._answer_educational_question(ctx.query, ctx.context.current_products)
                if answer:
                    return HandlerResult(response=answer)
                # Fall through to other handlers if LLM fails

            # Handle CLARIFY action - ask for clarification
            if interpretation.action == FollowupAction.CLARIFY:
                question = interpretation.clarification_question or (
                    "I'm not sure which product you're referring to. "
                    "Could you be more specific about what you'd like to know?"
                )
                return HandlerResult(response=question)

            # Handle DETAILS action - show details for specific product(s)
            if interpretation.action == FollowupAction.DETAILS:
                if interpretation.product_indices:
                    from llm.llm_response_generator import ResponseType
                    response = self._generate_llm_response(ctx, interpretation, ResponseType.PRODUCT_DETAILS, fallback_to_all=False)
                    if response:
                        return HandlerResult(response=response)

            # Handle COMPARE action - compare specific products
            # The LLM interpreter extracts product_indices from queries like "the first two"
            # which the pattern-based handler can't parse
            if interpretation.action == FollowupAction.COMPARE:
                if interpretation.product_indices and len(interpretation.product_indices) >= 2:
                    ctx.add_debug(f"COMPARE: Using LLM indices {interpretation.product_indices}")

                    from llm.llm_response_generator import ResponseType
                    response = self._generate_llm_response(ctx, interpretation, ResponseType.COMPARISON, fallback_to_all=False)
                    if response:
                        ctx.context.set_comparison_context(interpretation.product_indices)
                        return HandlerResult(response=response)

            # Handle RECOMMEND action - make a product recommendation
            if interpretation.action == FollowupAction.RECOMMEND:
                ctx.add_debug("RECOMMEND: Making product recommendation")
                use_case = (interpretation.criteria or {}).get('use_case')

                from llm.llm_response_generator import ResponseType
                response = self._generate_llm_response(
                    ctx, interpretation, ResponseType.RECOMMENDATION,
                    extra_context={'use_case': use_case} if use_case else None,
                )
                if response:
                    return HandlerResult(response=response)

            # Let other actions (SPECS) fall through to existing handlers
            return None

        except ImportError:
            ctx.add_debug("LLM FOLLOWUP: Import error")
            return None
        except Exception as e:
            ctx.add_debug(f"LLM FOLLOWUP: {type(e).__name__}: {str(e)[:50]}")
            return None

    def _show_product_specs(self, ctx: HandlerContext) -> HandlerResult:
        """Show specs for products in context using LLM with tier2 data."""
        from llm.llm_response_generator import generate_response, ResponseType
        products = ctx.context.current_products[:5]
        response = generate_response(
            products=products,
            query=ctx.query,
            response_type=ResponseType.SEARCH_RESULTS,
        )
        if response:
            return HandlerResult(response=response)
        # Minimal fallback if LLM fails
        lines = []
        for i, prod in enumerate(products, 1):
            lines.append(f"{i}. **{prod.product_number}** - {prod.metadata.get('name', prod.product_number)}")
        return HandlerResult(response="\n".join(lines))

    def _check_product_compatibility(self, product, feature: str) -> bool:
        """
        Check if a product supports a given feature.

        Args:
            product: Product to check
            feature: Feature name (e.g., "Thunderbolt 3", "4K", "Mac")

        Returns:
            True if product supports the feature
        """
        feature_lower = feature.lower()
        content_lower = (product.content or '').lower()
        features = [f.lower() for f in product.metadata.get('features', [])]
        meta = product.metadata

        # Thunderbolt compatibility
        if 'thunderbolt' in feature_lower:
            # USB-C cables with USB 3.1+ are generally Thunderbolt 3 compatible
            connectors = [str(c).lower() for c in meta.get('connectors', [])]
            has_usbc = any('usb-c' in c or 'usb c' in c or 'type-c' in c for c in connectors)

            # Check for explicit Thunderbolt mention
            if 'thunderbolt' in content_lower:
                return True
            # USB-C with USB 3.1/3.2 Gen 2 speeds are TB3 compatible
            if has_usbc and any(kw in content_lower for kw in ['usb 3.1', 'usb 3.2', '10gbps', '10 gbps']):
                return True
            if any('thunderbolt' in f for f in features):
                return True

            return False

        # Mac compatibility
        if feature_lower in ('mac', 'macbook', 'macos', 'apple'):
            # USB-C products work with Mac
            connectors = [str(c).lower() for c in meta.get('connectors', [])]
            has_usbc = any('usb-c' in c or 'usb c' in c or 'type-c' in c for c in connectors)
            if has_usbc:
                return True
            # Check explicit mentions
            if any(kw in content_lower for kw in ['mac', 'macos', 'macbook', 'apple']):
                return True

            return False

        # 4K support
        if '4k' in feature_lower:
            return supports_4k(product)

        # Power delivery
        if 'power delivery' in feature_lower or feature_lower == 'pd':
            pd_watts = meta.get('power_delivery') or meta.get('hub_power_delivery')
            if pd_watts:
                return True
            if 'power delivery' in content_lower or ' pd ' in content_lower:
                return True

            return False

        # Generic feature check
        if feature_lower in content_lower:
            return True
        if any(feature_lower in f for f in features):
            return True

        return False

    def _get_compatibility_explanations(self, products, feature: str) -> list:
        """
        Get brief explanations for why products support a feature.

        Args:
            products: List of products
            feature: Feature name

        Returns:
            List of explanation strings
        """
        feature_lower = feature.lower()
        explanations = []

        for prod in products:
            sku = prod.product_number
            meta = prod.metadata
            content_lower = (prod.content or '').lower()

            # Thunderbolt explanation
            if 'thunderbolt' in feature_lower:
                connectors = [str(c).lower() for c in meta.get('connectors', [])]
                has_usbc = any('usb-c' in c or 'usb c' in c for c in connectors)
                if has_usbc:
                    # Find speed info
                    if '10gbps' in content_lower or '10 gbps' in content_lower:
                        explanations.append(f"**{sku}**: USB-C with 10Gbps data rate (TB3 compatible)")
                    elif 'usb 3.2' in content_lower:
                        explanations.append(f"**{sku}**: USB 3.2 spec supports Thunderbolt 3")
                    else:
                        explanations.append(f"**{sku}**: USB-C connector works with Thunderbolt 3 ports")
                continue

            # Mac explanation
            if feature_lower in ('mac', 'macbook', 'macos', 'apple'):
                connectors = [str(c).lower() for c in meta.get('connectors', [])]
                has_usbc = any('usb-c' in c or 'usb c' in c for c in connectors)
                if has_usbc:
                    explanations.append(f"**{sku}**: USB-C works natively with Mac")
                continue

            # 4K explanation
            if '4k' in feature_lower:
                if supports_4k(prod):
                    features = meta.get('features', [])
                    res_feature = next((f for f in features if '4k' in f.lower()), None)
                    if res_feature:
                        explanations.append(f"**{sku}**: {res_feature}")

        return explanations

    def _answer_educational_question(self, query: str, products: list) -> str | None:
        """
        Answer an educational/physics question using LLM.

        These are general knowledge questions about technology, not about
        specific products. Examples:
        - "Does cable length affect charging speed?"
        - "How does USB-C deliver power?"
        - "What's the difference between Cat6 and Cat6a?"

        Args:
            query: User's educational question
            products: Current products for context (may inform the answer)

        Returns:
            Answer string or None on failure
        """
        import os
        from core.openai_client import get_openai_client

        client = get_openai_client()
        if not client:
            return None

        # Build context about current products for relevant answers
        product_context = ""
        if products:
            product_types = set()
            for p in products:
                cat = p.metadata.get('category', '')
                if cat:
                    product_types.add(cat)
            if product_types:
                product_context = f"\nContext: User is looking at {', '.join(product_types)} products."

        prompt = f"""You are a helpful technical expert for StarTech.com products.
Answer this educational/technical question concisely (2-4 sentences).
Focus on practical implications for the user.
Be factual and helpful - don't hedge unnecessarily.
{product_context}

Question: {query}"""

        try:
            response = client.chat.completions.create(
                model=os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano'),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_completion_tokens=300,
                timeout=30.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            _logger.warning(f"Educational question LLM call failed: {type(e).__name__}: {e}")
            return None

    # _make_recommendation was removed — replaced by _generate_llm_response
    # with ResponseType.RECOMMENDATION (centralized LLM response generation)
