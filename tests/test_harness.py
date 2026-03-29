"""
ST-Bot v2.0 End-to-End Test Harness

Runs real queries through the full pipeline (real LLM, real products, real search)
and validates intent classification, filter extraction, and product results.

Supports multi-step conversations to test followup flows.

Usage:
    cd "ST-Bot 2"
    python tests/test_harness.py
"""

import sys
import os
import re
import time
import argparse

# Ensure ST-Bot 2 root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# TEST CASES
# =============================================================================
# Each test case has a name and one or more steps.
# Steps share a conversation context (for followup testing).
# Each step has a query and expected outcomes to validate.
#
# Available expect keys:
#   intent            - exact intent type string (e.g., "new_search", "followup")
#   category          - filter category should contain this substring
#   category_any_of   - list of valid category substrings (match any)
#   min_monitors      - extracted min_monitors filter value
#   port_count        - extracted port_count filter value
#   rack_height       - extracted rack_height filter value
#   thunderbolt       - extracted thunderbolt_version filter value
#   power_wattage     - extracted requested_power_wattage filter value
#   refresh_rate      - extracted requested_refresh_rate filter value
#   network_speed     - extracted requested_network_speed filter value
#   has_length        - True if a length was extracted
#   products_min      - minimum number of products returned
#   products_category - all returned products should have this category
#   response_contains - list of substrings that should appear in response
#   response_not_contains - list of substrings that should NOT appear in response
# =============================================================================

TEST_CASES = [
    # -----------------------------------------------------------------
    # Intent classification basics
    # -----------------------------------------------------------------
    {
        "name": "Greeting",
        "steps": [
            {
                "query": "Hello!",
                "expect": {"intent": "greeting"}
            }
        ]
    },
    {
        "name": "Farewell",
        "steps": [
            {
                "query": "Thanks, goodbye!",
                "expect": {"intent": "farewell"}
            }
        ]
    },
    {
        "name": "Educational question",
        "steps": [
            {
                "query": "What's the difference between Cat6 and Cat6a?",
                "expect": {"intent": "educational"}
            }
        ]
    },
    {
        "name": "Out of scope - return policy",
        "steps": [
            {
                "query": "What's your return policy?",
                "expect": {"intent": "out_of_scope"}
            }
        ]
    },

    # -----------------------------------------------------------------
    # Search + filter extraction
    # -----------------------------------------------------------------
    {
        "name": "HDMI cable with length",
        "steps": [
            {
                "query": "I need a 6ft HDMI cable",
                "expect": {
                    "intent": "new_search",
                    "has_length": True,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "USB-C to HDMI adapter",
        "steps": [
            {
                "query": "USB-C to HDMI adapter",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Dock with power and refresh rate",
        "steps": [
            {
                "query": "USB-C docking station with 90W charging and 60Hz display output",
                "expect": {
                    "intent": "new_search",
                    "power_wattage": 90,
                    "refresh_rate": 60,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "KVM switch with HDMI",
        "steps": [
            {
                "query": "4-port KVM switch with HDMI",
                "expect": {
                    "intent": "new_search",
                    "port_count": 4,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Server rack 25U",
        "steps": [
            {
                "query": "server rack enclosure at least 25U",
                "expect": {
                    "intent": "new_search",
                    "rack_height": 25,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Dual monitor mounts",
        "steps": [
            {
                "query": "dual monitor mounts",
                "expect": {
                    "intent": "new_search",
                    "category": "display_mount",
                    "min_monitors": 2,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Cat6a cable with length",
        "steps": [
            {
                "query": "Cat6a ethernet cable 50 feet",
                "expect": {
                    "intent": "new_search",
                    "has_length": True,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Connector + product type extracts category",
        "steps": [
            {
                "query": "Do you sell Thunderbolt cables?",
                "expect": {
                    "intent": "new_search",
                    "category": "cable",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Connector cable correction preserves same-connector",
        "steps": [
            {
                "query": "Do you sell Thunderbolt cables?",
                "expect": {
                    "intent": "new_search",
                    "category": "cable",
                    "products_min": 1,
                }
            },
            {
                "query": "I meant USB-C not Thunderbolt",
                "expect": {
                    "intent": "new_search",
                    "category": "cable",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Thunderbolt 4 dock",
        "steps": [
            {
                "query": "Thunderbolt 4 dock",
                "expect": {
                    "intent": "new_search",
                    "thunderbolt": 4,
                    "products_min": 1,
                }
            }
        ]
    },

    # -----------------------------------------------------------------
    # Followup handling (multi-step conversations)
    # -----------------------------------------------------------------
    {
        "name": "Followup: specific question about product",
        "steps": [
            {
                "query": "USB-C docking stations",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "How many ports does the second one have?",
                "expect": {
                    "intent": "followup",
                }
            }
        ]
    },
    {
        "name": "Followup: help me decide",
        "steps": [
            {
                "query": "dual monitor mounts",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "I need help deciding which mount would work best",
                "expect": {
                    "intent": "followup",
                }
            }
        ]
    },
    {
        "name": "Followup: filter current results",
        "steps": [
            {
                "query": "HDMI cables",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "do you have any 10ft or longer?",
                "expect": {
                    "intent": "followup",
                }
            }
        ]
    },
    {
        "name": "Followup: which do you recommend",
        "steps": [
            {
                "query": "USB-C hubs",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "which would you recommend?",
                "expect": {
                    "intent": "followup",
                }
            }
        ]
    },
    {
        "name": "Followup: compare products",
        "steps": [
            {
                "query": "Thunderbolt docks",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "what's the difference between the first two?",
                "expect": {
                    "intent": "followup",
                }
            }
        ]
    },
    {
        "name": "New search after products shown (different category)",
        "steps": [
            {
                "query": "USB-C docking stations",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            {
                "query": "I also need HDMI cables",
                "expect": {
                    "intent_any_of": ["new_search", "followup"],
                }
            }
        ]
    },

    {
        "name": "Narrowing refinement preserves category",
        "steps": [
            {
                "query": "USB-C docking stations",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "products_min": 1,
                }
            },
            {
                "query": "Do you have one with Thunderbolt support?",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "products_min": 1,
                }
            }
        ]
    },

    # -----------------------------------------------------------------
    # Realistic stress tests (complex, multi-requirement queries)
    # -----------------------------------------------------------------
    {
        "name": "USB-C to VGA projector 30ft",
        "steps": [
            {
                "query": "I need to connect my USB-C laptop to a VGA projector that's 30 feet away in a conference room",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "4 computers share monitors + kb/mouse (KVM)",
        "steps": [
            {
                "query": "We have 4 computers that need to share 2 monitors, a keyboard, and a mouse - what do I need?",
                "expect": {
                    "intent": "new_search",
                    "category": "kvm",
                    "port_count": 4,
                    "min_monitors": 2,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Fiber optic media converter 10Gbps",
        "steps": [
            {
                "query": "I'm looking for a fiber optic media converter that supports 10 gigabit SFP+ and single mode fiber",
                "expect": {
                    "intent": "new_search",
                    "network_speed": 10000,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "PoE injector gigabit",
        "steps": [
            {
                "query": "I need a gigabit PoE injector",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Complete rack setup 12U",
        "steps": [
            {
                "query": "I need a complete rack setup for a small server room - enclosure, shelves, power, and cable management for about 12U",
                "expect": {
                    "intent": "new_search",
                    "category": "rack",
                    "rack_height": 12,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Cross-platform dock with dual monitors",
        "steps": [
            {
                "query": "I need a docking station that works with my MacBook and also my colleague's Lenovo ThinkPad, both need dual monitors and ethernet",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "min_monitors": 2,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Single USB-C port needs everything (implicit dock)",
        "steps": [
            {
                "query": "I have a laptop with only one USB-C port and I need dual monitors, ethernet, USB peripherals, and laptop charging all through that single port",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "min_monitors": 2,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "4K HDMI splitter 2 outputs",
        "steps": [
            {
                "query": "I need a 4K HDMI splitter with 2 outputs",
                "expect": {
                    "intent": "new_search",
                    "category_any_of": ["video_splitter", "splitter"],
                    "port_count": 2,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "HDMI splitter to 8 displays",
        "steps": [
            {
                "query": "I need to duplicate a signal from one HDMI source to 8 different displays across a large showroom",
                "expect": {
                    "intent": "new_search",
                    "category_any_of": ["video_splitter", "splitter"],
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Multi-category workstation setup",
        "steps": [
            {
                "query": "We're setting up 6 workstations that all need dual monitor mounts, docking stations, and cable management - what do you recommend?",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Multi-user KVM for 4 servers",
        "steps": [
            {
                "query": "I need a KVM solution where 2 users can independently access 4 different servers from separate locations",
                "expect": {
                    "intent": "new_search",
                    "category_any_of": ["kvm_switch", "kvm_extender"],
                    "port_count": 4,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "TB4 dock full spec (stress test)",
        "steps": [
            {
                "query": "I'm looking for a Thunderbolt 4 dock that supports dual 4K 60Hz, has at least 3 USB-A ports, gigabit ethernet, and 96W charging",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "thunderbolt": 4,
                    "min_monitors": 2,
                    "power_wattage": 96,
                    "refresh_rate": 60,
                    "products_min": 1,
                }
            }
        ]
    },
    {
        "name": "Single mode fiber cable extracts mode keyword",
        "steps": [
            {
                "query": "15 meter single mode fiber optic cable with LC connectors",
                "expect": {
                    "intent": "new_search",
                    "category": "fiber_cable",
                    "products_min": 1,
                }
            }
        ]
    },

    # -----------------------------------------------------------------
    # Round 7: Multi-turn conversation — hub → narrowing → compatibility
    # → recommendation → pronoun resolution → category correction → refinement
    # -----------------------------------------------------------------
    {
        "name": "R7: Hub > narrowing > compat > recommend > pronoun > category correction > refine",
        "steps": [
            # Step 1: Basic hub search — likely triggers narrowing
            {
                "query": "I need a USB-C hub",
                "expect": {
                    "intent": "new_search",
                    "products_min": 1,
                }
            },
            # Step 2: Respond to narrowing (or followup if no narrowing triggered)
            {
                "query": "Show me all of them",
                "expect": {
                    "intent": "followup",
                    "products_min": 1,
                }
            },
            # Step 3: Compatibility check — "Do they all have ethernet?"
            # Tests plural pronoun resolution against ALL current products
            {
                "query": "Do they all have ethernet?",
                "expect": {
                    "intent": "followup",
                }
            },
            # Step 4: Recommendation with use-case context
            {
                "query": "Which one would you recommend for a video editor?",
                "expect": {
                    "intent": "followup",
                }
            },
            # Step 5: Pronoun "it" after recommendation — should target recommended product
            {
                "query": "How much power does it deliver?",
                "expect": {
                    "intent": "followup",
                }
            },
            # Step 6: Category correction — new search for docking station
            {
                "query": "Actually I need a full docking station, not just a hub",
                "expect": {
                    "intent": "new_search",
                    "category": "dock",
                    "products_min": 1,
                }
            },
            # Step 7: Refinement immediately after new search
            {
                "query": "With Thunderbolt and at least 3 USB-A ports",
                "expect": {
                    "intent_any_of": ["new_search", "followup"],
                    "products_min": 1,
                }
            },
        ]
    },
]


# =============================================================================
# DEBUG OUTPUT PARSER
# =============================================================================

def parse_debug_output(response: str) -> dict:
    """Extract structured data from debug output in the response."""
    parsed = {
        'filters_raw': None,
        'category': None,
        'min_monitors': None,
        'port_count': None,
        'rack_height': None,
        'thunderbolt_version': None,
        'power_wattage': None,
        'refresh_rate': None,
        'network_speed': None,
        'has_length': False,
    }

    # Find the FILTERS line and extract everything after it on that line
    # Use greedy match to handle field values containing parentheses (e.g. features=['4K (60Hz)'])
    filters_match = re.search(r'FILTERS: (SearchFilters\(.+)', response)
    if filters_match:
        filters_str = filters_match.group(1)
        parsed['filters_raw'] = filters_str

        # Extract category
        cat_match = re.search(r"product_category='([^']*)'", filters_str)
        if cat_match and cat_match.group(1):
            parsed['category'] = cat_match.group(1).lower().replace(' ', '_')

        # Extract numeric filter values
        for field, key in [
            ('min_monitors', 'min_monitors'),
            ('port_count', 'port_count'),
            ('rack_height', 'rack_height'),
            ('thunderbolt_version', 'thunderbolt_version'),
            ('requested_power_wattage', 'power_wattage'),
            ('requested_refresh_rate', 'refresh_rate'),
            ('requested_network_speed', 'network_speed'),
        ]:
            match = re.search(rf'{field}=(\d+)', filters_str)
            if match:
                parsed[key] = int(match.group(1))

        # Check if length was extracted
        length_match = re.search(r'length=(\d+\.?\d*)', filters_str)
        if length_match and float(length_match.group(1)) > 0:
            parsed['has_length'] = True

    return parsed


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(filter_names=None, last_n=None):
    """Run test cases and report results. Optionally filter by name or count."""
    print("=" * 70)
    print("ST-Bot v2.0 End-to-End Test Harness")
    print("=" * 70)

    # ---- Initialize (same as app.py) ----
    print("\nLoading products...")
    from data.loader import load_startech_products
    from core.orchestrator import process_query, OrchestratorComponents
    from core.models import ConversationContext
    from core.search import SearchEngineWrapper
    from llm.llm_intent_classifier import LLMIntentClassifier
    from llm.llm_filter_extractor import LLMFilterExtractor
    from llm.query_analyzer import QueryAnalyzer

    products = load_startech_products("ProductAttributeValues_Cleaned_Exported.xlsx")
    print(f"Loaded {len(products)} products")

    components = OrchestratorComponents(
        intent_classifier=LLMIntentClassifier(valid_skus={p.product_number.upper() for p in products}),
        filter_extractor=LLMFilterExtractor(),
        search_engine=SearchEngineWrapper(products),
        query_analyzer=QueryAnalyzer(),
    )

    # ---- Filter tests ----
    tests_to_run = TEST_CASES
    if filter_names:
        tests_to_run = [t for t in tests_to_run
                        if any(f.lower() in t['name'].lower() for f in filter_names)]
    if last_n:
        tests_to_run = tests_to_run[-last_n:]

    # ---- Run tests ----
    filter_note = ""
    if filter_names:
        filter_note += f" (filter: {', '.join(filter_names)})"
    if last_n:
        filter_note += f" (last {last_n})"
    print(f"\nRunning {len(tests_to_run)} of {len(TEST_CASES)} test cases{filter_note}...\n")

    results = []  # (name, step_desc, passed, failures)
    total_steps = 0
    total_passed = 0
    total_api_time = 0

    for test in tests_to_run:
        name = test['name']
        # Fresh context for each test case
        context = ConversationContext()

        for step_idx, step in enumerate(test['steps'], 1):
            total_steps += 1
            query = step['query']
            expect = step['expect']
            step_desc = f"{name} (step {step_idx})" if len(test['steps']) > 1 else name

            # Run query
            step_start = time.perf_counter()
            try:
                response, intent_type = process_query(
                    query=query,
                    context=context,
                    components=components,
                    all_products=products,
                    debug_mode=True,
                )
            except Exception as e:
                response = f"ERROR: {e}"
                intent_type = "error"
            step_time = time.perf_counter() - step_start
            total_api_time += step_time

            # Parse debug output
            debug_data = parse_debug_output(response)

            # Get product count from context (includes narrowing pool)
            product_count = 0
            if context.current_products:
                product_count = len(context.current_products)
            elif context.has_pending_narrowing() and context.pending_narrowing.product_pool:
                product_count = len(context.pending_narrowing.product_pool)

            # Validate expectations
            failures = []

            # Intent check
            if 'intent' in expect:
                if intent_type != expect['intent']:
                    failures.append(f"intent: expected={expect['intent']}, got={intent_type}")

            # Intent any-of check (for ambiguous queries where multiple intents are valid)
            if 'intent_any_of' in expect:
                if intent_type not in expect['intent_any_of']:
                    failures.append(f"intent: expected one of {expect['intent_any_of']}, got={intent_type}")

            # Category check (exact or substring match)
            if 'category' in expect:
                actual_cat = (debug_data['category'] or '').lower().replace(' ', '_')
                expected_cat = expect['category'].lower().replace(' ', '_')
                if not actual_cat:
                    failures.append(f"category: expected={expected_cat}, got=None")
                elif expected_cat not in actual_cat and actual_cat not in expected_cat:
                    failures.append(f"category: expected={expected_cat}, got={actual_cat}")

            # Category any-of check (for queries where multiple categories are valid)
            if 'category_any_of' in expect:
                actual_cat = (debug_data['category'] or '').lower().replace(' ', '_')
                valid_cats = [c.lower().replace(' ', '_') for c in expect['category_any_of']]
                if not actual_cat:
                    failures.append(f"category: expected one of {valid_cats}, got=None")
                elif not any(v in actual_cat or actual_cat in v for v in valid_cats):
                    failures.append(f"category: expected one of {valid_cats}, got={actual_cat}")

            # Numeric filter checks
            filter_checks = {
                'min_monitors': 'min_monitors',
                'port_count': 'port_count',
                'rack_height': 'rack_height',
                'thunderbolt': 'thunderbolt_version',
                'power_wattage': 'power_wattage',
                'refresh_rate': 'refresh_rate',
                'network_speed': 'network_speed',
            }
            for expect_key, parsed_key in filter_checks.items():
                if expect_key in expect:
                    actual = debug_data.get(parsed_key)
                    expected = expect[expect_key]
                    if actual != expected:
                        failures.append(f"{expect_key}: expected={expected}, got={actual}")

            # Length check
            if 'has_length' in expect:
                if debug_data['has_length'] != expect['has_length']:
                    failures.append(f"has_length: expected={expect['has_length']}, got={debug_data['has_length']}")

            # Product count check
            if 'products_min' in expect:
                if product_count < expect['products_min']:
                    failures.append(f"products: expected>={expect['products_min']}, got={product_count}")

            # Product category check
            if 'products_category' in expect and context.current_products:
                expected_cat = expect['products_category'].lower().replace('_', ' ')
                for p in context.current_products:
                    p_cat = (p.metadata.get('category', '') or '').lower().replace('_', ' ')
                    if expected_cat not in p_cat and p_cat not in expected_cat:
                        failures.append(f"product {p.product_number} category={p_cat}, expected={expected_cat}")
                        break

            # Response content checks
            if 'response_contains' in expect:
                resp_lower = response.lower()
                for substr in expect['response_contains']:
                    if substr.lower() not in resp_lower:
                        failures.append(f"response missing: '{substr}'")

            if 'response_not_contains' in expect:
                resp_lower = response.lower()
                for substr in expect['response_not_contains']:
                    if substr.lower() in resp_lower:
                        failures.append(f"response should not contain: '{substr}'")

            # Record result
            passed = len(failures) == 0
            if passed:
                total_passed += 1

            status = "[PASS]" if passed else "[FAIL]"

            # Build detail string
            details = []
            details.append(f"intent={intent_type}")
            if debug_data['category']:
                details.append(f"cat={debug_data['category']}")
            if product_count > 0:
                details.append(f"{product_count} products")
            details.append(f"{step_time:.1f}s")
            detail_str = ", ".join(details)

            print(f"  {status} {step_desc}")
            print(f"         query: \"{query}\"")
            print(f"         {detail_str}")
            if failures:
                for f in failures:
                    print(f"         >> {f}")
            print()

            results.append((step_desc, passed, failures))

    # ---- Summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total steps:  {total_steps}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_steps - total_passed}")
    print(f"  Pass rate:    {total_passed/total_steps*100:.0f}%")
    print(f"  Total time:   {total_api_time:.1f}s")
    print()

    if total_passed < total_steps:
        print("FAILURES:")
        for name, passed, failures in results:
            if not passed:
                for f in failures:
                    print(f"  - {name}: {f}")
        print()

    return total_passed == total_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-Bot v2.0 End-to-End Test Harness")
    parser.add_argument('--filter', '-f', action='append', dest='filters',
                        help="Run tests whose name contains this string (case-insensitive, repeatable)")
    parser.add_argument('--last', '-l', type=int, dest='last_n',
                        help="Run only the last N test cases")
    args = parser.parse_args()
    success = run_tests(filter_names=args.filters, last_n=args.last_n)
    sys.exit(0 if success else 1)
