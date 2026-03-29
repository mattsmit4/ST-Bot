"""
Greeting, farewell, and out-of-scope intent handlers.

Simple handlers for conversational intents that don't require
product search or complex logic.
"""

from handlers.base import BaseHandler, HandlerContext, HandlerResult
from llm.prompts import get_system_prompts


class GreetingHandler(BaseHandler):
    """Handle greeting intent."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        response = get_system_prompts().format_greeting_response()
        return HandlerResult(response=response)


class FarewellHandler(BaseHandler):
    """Handle farewell intent."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        response = get_system_prompts().format_farewell_response()
        return HandlerResult(response=response)


class OutOfScopeHandler(BaseHandler):
    """
    Handle out-of-scope queries.

    Returns appropriate responses for:
    - Gibberish/random text
    - Customer service questions (returns, pricing, warranty)
    - Competitor comparisons
    - Off-topic requests (politics, essays, etc.)
    - Prompt injection attempts
    """

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        # Get the out-of-scope type and reason from intent metadata
        meta = ctx.intent.meta_info or {}
        scope_type = meta.get('out_of_scope_type', 'unknown')
        reason = ctx.intent.reasoning

        # Build response based on type
        if scope_type == 'gibberish':
            response = (
                "I didn't understand that. Could you rephrase your question?\n\n"
                "I can help you find:\n"
                "- Cables (HDMI, DisplayPort, USB, Ethernet)\n"
                "- Adapters and converters\n"
                "- Docking stations and hubs\n"
                "- KVM switches\n"
                "- Mounts and racks"
            )
        elif scope_type == 'customer_service':
            response = (
                "I can't help with orders, returns, or account questions.\n\n"
                "You can reach StarTech.com support at:\n"
                "- Website: www.startech.com/support\n"
                "- Phone: 1-800-265-1844"
            )
        elif scope_type == 'pricing':
            response = (
                "I don't have pricing information.\n\n"
                "For product pricing and availability, please visit www.startech.com"
            )
        elif scope_type == 'competitor':
            response = (
                "I can only provide information about StarTech.com products. "
                "Would you like me to help you find a specific type of product?"
            )
        elif scope_type == 'fictional':
            response = (
                "I couldn't find that product in our catalog.\n\n"
                "What real product can I help you find today?"
            )
        elif scope_type == 'injection':
            response = (
                "I'm a product recommendation assistant for StarTech.com. "
                "How can I help you find the right cable, adapter, dock, or other connectivity product?"
            )
        elif scope_type == 'setup_help':
            response = (
                "I'm a product recommendation assistant — I can help you find the right "
                "StarTech.com product, but I can't help with device setup or configuration.\n\n"
                "If you need help setting up a **StarTech product**, visit "
                "[StarTech.com Technical Support](https://www.startech.com/support).\n\n"
                "Otherwise, try asking me things like:\n"
                "- \"I need to connect my laptop to two monitors\"\n"
                "- \"Show me USB-C docking stations\"\n"
                "- \"What's the difference between HDMI and DisplayPort?\""
            )
        elif scope_type == 'not_our_products':
            response = (
                "StarTech.com specializes in connectivity accessories — we don't sell "
                "computers, monitors, printers, or other devices directly.\n\n"
                "But we carry everything to connect them:\n"
                "- Docking stations and hubs\n"
                "- Cables and adapters\n"
                "- KVM switches\n"
                "- Mounts and racks\n\n"
                "What devices are you trying to connect?"
            )
        elif scope_type == 'unavailable_feature':
            response = reason  # Use the specific reason which explains what's unavailable
        else:
            # Generic off-topic response
            response = (
                "I can only help with product recommendations for StarTech.com.\n\n"
                "Try asking about:\n"
                "- Cables (HDMI, DisplayPort, USB, Ethernet)\n"
                "- Adapters and converters\n"
                "- Docking stations\n"
                "- KVM switches"
            )

        ctx.add_debug(f"OUT OF SCOPE: {scope_type}")
        return HandlerResult(response=response)
