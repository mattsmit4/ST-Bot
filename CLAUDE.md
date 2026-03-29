# ST-Bot 2 — Claude Code Instructions

## Architecture
6-layer pipeline: Storefront → Sorter → Intent Classifier → Filter Extractor → Handlers → Response Generation.

### Directory Structure
- `app.py` — Streamlit entry point, chat UI, session management
- `core/orchestrator.py` — Pipeline orchestration, intent routing, state management
- `core/models/` — Data models (Product, SearchFilters, Intent, ConversationContext)
- `core/search/` — Search engine (scored search, connector matching, filter dispatch)
- `core/search/config.py` — Connector match config, filter dispatch table
- `core/category_config.py` — Single source of truth for category names + descriptions
- `core/openai_client.py` — Centralized OpenAI client with token usage tracking
- `core/vagueness.py` — LLM-based vague query detection
- `core/clarification.py` — Clarification question building + response parsing
- `handlers/` — Intent handlers (search, followup, narrowing, clarification, sku, educational, greeting)
- `handlers/followup.py` — Follow-up routing (~690 lines, uses RefinementMixin)
- `handlers/followup_refinement.py` — Refinement + search helper methods (RefinementMixin)
- `llm/` — All LLM prompts and callers
- `llm/llm_response_generator.py` — Tiered response generation (tier1/2/3)
- `llm/llm_followup_interpreter.py` — Follow-up action classification (filter, specs, compare, etc.)
- `llm/llm_narrowing_analyzer.py` — Product narrowing question generation
- `llm/llm_filter_extractor.py` — Natural language → SearchFilters extraction
- `llm/llm_intent_classifier.py` — Intent classification with fast paths
- `config/` — Configuration (category columns, patterns, device mappings)
- `config/__init__.py` — Shared get_config_value()
- `config/category_columns.py` — Per-category tier1/tier2 display field configs + validation
- `config/patterns.py` — Pricing, warranty, order detection patterns
- `config/device_mappings.py` — Device → connector inference (printer→USB-B, monitor→HDMI)
- `data/` — Data loading and field derivation
- `data/loader.py` — Excel → Product loading with COLUMN_ALIASES
- `data/derived.py` — Field normalization (USB version, cable length, PoE, categories)
- `tests/` — Test suite (84 tests total)

## How to Run
```bash
cd "ST-Bot 2"
streamlit run app.py
```

## Fix Philosophy
- **Scalable over clever** — every fix must work for ALL product types, not just the failing case. Ask: "Will this need updating when a new product category is added?" If yes, find a better approach.
- **Reuse existing data** — prefer leveraging what's already in the system (e.g., `context.last_filters`) over building new keyword lists or pattern matchers.
- **No spaghetti** — if a fix adds complexity, it's the wrong fix. Simpler code is better code.
- **LLM over hard-coded** — prefer LLM-based solutions over regex/keyword matching. API costs don't matter.
- **Generic, not specific** — never hard-code a fix for one failing query.
- **Check obvious connections first** — before assuming the pipeline logic is wrong, verify data is flowing correctly (key mismatches, missing pass-through, case sensitivity).

## Key Design Patterns
- **Tier system**: tier1 (product cards), tier2 (comparisons/specs), tier3 (full detail). `tier_override` in context dict overrides.
- **Category carry-forward**: When products are in context and LLM can't extract a category, the previous category is carried forward. Skipped when >5 products (stale pool) or when bot suggested a different category.
- **Hardware-agnostic compatibility**: Privacy screens, cables, mounts, switches, etc. answer "Yes — works with any OS" for Mac/Windows compatibility questions.
- **Connector override**: Printer/scanner queries get `connector_to='USB-B'` post-processing.
- **Narrowing exclusions**: `standards`, `color`, `material` excluded from narrowing. Network speed fields excluded when already filtered.
- **Vagueness gate**: Queries with category but no qualifying filters (connectors, length, features) trigger clarification instead of dumping products.

## Testing
```bash
# Unit tests (fast, no LLM calls)
pytest tests/test_search_engine.py tests/test_derived_fields.py -v

# Integration tests (fast, mocked)
pytest tests/test_integration.py -v

# Full suite (84 tests)
pytest tests/ -v

# End-to-end with real LLM (slow)
python tests/test_harness.py
```

## User Preferences
- Check in before major changes (plan mode)
- Give honest assessments, not sugar-coating
- High-level explanations of what changed
- Fix issues immediately when found during testing, don't defer to notes
- Check obvious data connection issues first (key mismatches, case sensitivity)
