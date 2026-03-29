"""
ST-Bot v2.0 — StarTech.com Product Assistant

Streamlit entry point. All business logic lives in core/ and handlers/.
This file handles only: UI, session state, and component wiring.

Run with: streamlit run app.py
"""

# Load environment variables FIRST (before any imports that depend on them)
from dotenv import load_dotenv
load_dotenv()

import os
import random
import threading
import uuid

import streamlit as st

from data.loader import load_startech_products, get_product_statistics
from core.models import ConversationContext
from core.search import SearchEngineWrapper
from core.orchestrator import process_query, OrchestratorComponents
from llm.llm_intent_classifier import LLMIntentClassifier
from llm.llm_filter_extractor import LLMFilterExtractor
from llm.query_analyzer import QueryAnalyzer
from app_logging import setup_logging, get_logger
from app_logging.gsheets_logger import init_gsheets_logger, GSPREAD_AVAILABLE


# =============================================================================
# SPINNER MESSAGES
# =============================================================================

SPINNER_MESSAGES = [
    # Serious
    "Searching our catalog...",
    "Looking through our products...",
    "Scanning our full catalog...",
    "Finding the best match...",
    "Searching for relevant products...",
    "Checking our product lineup...",
    "Browsing the catalog for you...",
    "Matching your requirements...",
    "Pulling up product details...",
    "Searching across all categories...",
    "Looking up your options...",
    "Searching for compatible products...",
    "Reviewing our product specs...",
    "Finding products that fit your needs...",
    "Cross-referencing specifications...",
    "Filtering through our inventory...",
    "Narrowing down your options...",
    "Checking product availability...",
    "Analyzing your requirements...",
    "Running a product search...",
    # Goofy
    "Untangling cables to find your answer...",
    "Hold my ethernet cable...",
    "Flipping through datasheets at superhuman speed...",
    "Rummaging through the warehouse...",
    "Asking the servers nicely...",
    "Teaching the chatbot to read datasheets...",
    "Politely asking the database for help...",
    "Speed-reading every product manual...",
    "Warming up the product-finding engines...",
    "One sec, the warehouse is really big...",
    # Patience
    "Thanks for your patience — searching now...",
    "Bear with me, finding the right products...",
    "Give me just a moment to find the best options...",
    "This one's worth the wait — searching now...",
    "Taking a careful look through our catalog...",
    "Appreciate your patience — matching specs now...",
    "Worth the wait, promise...",
    "Brewing up results...",
    "Almost there, no rush right?",
    "Working hard, not hardly working...",
]


# =============================================================================
# CONFIGURATION
# =============================================================================

from config import get_config_value as _get_config_value

DEBUG_MODE = _get_config_value('DEBUG_MODE', 'true') == 'true'

# Initialize structured logging (module level — runs once)
setup_logging(
    log_dir="logs",
    console_level=20,   # INFO
    file_level=10,      # DEBUG
    enable_console=True,
    enable_file=False,
    enable_csv=False,
    enable_error_log=True,
)
_logger = get_logger("app")

# Initialize Google Sheets logging for cloud deployment
if GSPREAD_AVAILABLE:
    try:
        if "gsheets" in st.secrets:
            gsheets_creds = dict(st.secrets["gsheets"])
            spreadsheet_id = st.secrets.get("gsheets_spreadsheet_id", "")
            if spreadsheet_id:
                init_gsheets_logger(spreadsheet_id, gsheets_creds)
                _logger.info("Google Sheets logging initialized")
    except Exception as e:
        _logger.warning(f"Google Sheets logging not configured: {e}")


# =============================================================================
# COMPONENT INITIALIZATION
# =============================================================================

@st.cache_resource(show_spinner="Loading product catalog...")
def load_products(excel_path: str):
    """Load products from Excel (cached across reruns)."""
    try:
        products = load_startech_products(excel_path)
        stats = get_product_statistics(products)
        return products, stats, None
    except FileNotFoundError:
        return [], {}, f"File not found: {excel_path}"
    except Exception as e:
        return [], {}, f"Error loading Excel: {str(e)}"


def get_components(products) -> OrchestratorComponents:
    """Create orchestrator components. Not cached — constructors are lightweight."""
    return OrchestratorComponents(
        intent_classifier=LLMIntentClassifier(valid_skus={p.product_number.upper() for p in products}),
        filter_extractor=LLMFilterExtractor(),
        search_engine=SearchEngineWrapper(products),
        query_analyzer=QueryAnalyzer(),
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="ST-Bot - StarTech.com Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 ST-Bot - StarTech.com Product Assistant")
    st.markdown("*Your dedicated connectivity expert — here to help you find the right product, answer technical questions, and compare options.*")

    # --- Sidebar: Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        excel_path = st.text_input(
            "📁 Excel File Path",
            value="ProductAttributeValues_Cleaned_Exported.xlsx",
            help="Path to your Excel file",
        )
        st.markdown("---")

    # --- Load products ---
    products, stats, error = load_products(excel_path)

    if error:
        st.error(f"❌ {error}")
        st.info("💡 Please ensure ProductAttributeValues_Cleaned_Exported.xlsx is in the same folder as this app")
        st.stop()

    if not products:
        st.warning("⚠️ No products loaded. Check your Excel file.")
        st.stop()

    # --- Sidebar: Product Statistics ---
    with st.sidebar:
        st.header("📦 Product Catalog")
        st.metric("Total Products", stats['total'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("With Length", stats['with_length'])
        with col2:
            st.metric("With Connectors", stats['with_connectors'])

        with st.expander("📊 Categories"):
            for cat, count in sorted(
                stats['by_category'].items(), key=lambda x: x[1], reverse=True
            )[:10]:
                st.write(f"• **{cat.title()}:** {count}")

        st.markdown("---")

    # --- Initialize session state ---
    if "context" not in st.session_state:
        st.session_state.context = ConversationContext(
            session_id=str(uuid.uuid4())
        )
        st.session_state.messages = []

    # --- Components ---
    components = get_components(products)

    # --- Sidebar: Session Stats ---
    with st.sidebar:
        st.header("📊 Session Stats")
        session_id = st.session_state.context.session_id or ""
        st.write(f"**Session ID:** `{session_id[:16]}...`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")

        if st.session_state.context.current_products:
            st.write(
                f"**Products in Context:** "
                f"{len(st.session_state.context.current_products)}"
            )

        if st.button("🔄 New Session"):
            st.session_state.context = ConversationContext(
                session_id=str(uuid.uuid4())
            )
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ Clear Product Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared! Reloading products...")
            st.rerun()

    # --- Display chat history (or welcome message) ---
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Welcome! I'm your StarTech.com product assistant. I can help you:\n\n"
                "- **Find products** — \"I need a USB-C dock with dual 4K\"\n"
                "- **Look up a product** — \"Tell me about DK31C4DPPD\"\n"
                "- **Compare options** — \"How do those two compare on power delivery?\"\n"
                "- **Answer technical questions** — \"What's the difference between Cat6 and Cat6a?\"\n"
                "- **Ask follow-up questions** — \"Does that work with my MacBook?\" or \"How many ports does it have?\"\n\n"
                "What can I help you find today?"
            )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    prompt = st.chat_input("Ask your tech questions here!")

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query in background thread with spinner
        with st.chat_message("assistant"):
            placeholder = st.empty()
            result = {}
            current_msg = random.choice(SPINNER_MESSAGES)
            placeholder.markdown(f"⏳ {current_msg}")

            context = st.session_state.context

            def _run_query():
                try:
                    result['response'], result['intent'] = process_query(
                        query=prompt,
                        context=context,
                        components=components,
                        all_products=products,
                        debug_mode=DEBUG_MODE,
                    )
                except Exception as e:
                    _logger.error(f"Query processing error: {e}", exc_info=True)
                    result['response'] = (
                        "I'm having trouble right now. Please try again."
                    )
                    result['intent'] = None

            thread = threading.Thread(target=_run_query, daemon=True)
            thread.start()

            # Rotate spinner messages while waiting
            while thread.is_alive():
                thread.join(timeout=8.0)
                if thread.is_alive():
                    next_msgs = [m for m in SPINNER_MESSAGES if m != current_msg]
                    current_msg = random.choice(next_msgs)
                    placeholder.markdown(f"⏳ {current_msg}")

            placeholder.empty()

            response = result.get(
                'response', "I'm having trouble right now. Please try again."
            )
            intent_type = result.get('intent')

            st.markdown(response)

            # Debug expander
            if DEBUG_MODE:
                with st.expander("🔍 Debug Info"):
                    st.write(f"**Intent Detected:** {intent_type}")
                    st.write(f"**Total Products Available:** {len(products)}")
                    if st.session_state.context.current_products:
                        st.write(
                            f"**Products in Context:** "
                            f"{len(st.session_state.context.current_products)}"
                        )

        # Save assistant message for display history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
