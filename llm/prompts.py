"""
System prompts and response templates for ST-Bot.

Contains all prompts used with Claude API and response formatting templates.
"""

from typing import Optional, List


class SystemPrompts:
    """
    System prompts and templates for LLM interactions.

    Provides prompts for:
    - Query parsing with Claude
    - Response formatting
    - Educational content
    - Error messages

    Example:
        prompts = SystemPrompts()
        system_prompt = prompts.get_query_parser_prompt()
        response = prompts.format_product_results(products)
    """

    def __init__(self):
        """Initialize system prompts."""
        pass

    def get_query_parser_prompt(self) -> str:
        """
        Get system prompt for query parsing with LLM.

        Returns:
            System prompt for Claude API

        Example:
            >>> prompts = SystemPrompts()
            >>> prompt = prompts.get_query_parser_prompt()
            >>> # Use with Claude API
        """
        return """You are a technical product expert helping users find the right cables, adapters, and connectivity products.

Your job is to understand the user's query and extract:
1. What product type they need (e.g., "USB-C to HDMI cable", "DisplayPort adapter")
2. What technical features they need (e.g., "4K", "Thunderbolt", "6ft")
3. Their use case (e.g., "connecting laptop to monitor")

Respond with JSON only:
{
    "product_type": "specific product type",
    "features": ["list", "of", "features"],
    "use_case": "what they're trying to do",
    "confidence": 0.0-1.0
}

Examples:
Query: "I need to connect my MacBook to my 4K TV"
Response: {"product_type": "USB-C to HDMI cable", "features": ["4K"], "use_case": "connect MacBook to 4K TV", "confidence": 0.9}

Query: "cable for my monitor"
Response: {"product_type": "HDMI cable", "features": [], "use_case": "connect to monitor", "confidence": 0.6}

Be specific about product types. Only respond with valid JSON."""

    def get_conceptual_question_prompt(self) -> str:
        """
        Get system prompt for answering conceptual questions.

        Returns:
            System prompt for educational responses

        Example:
            >>> prompt = prompts.get_conceptual_question_prompt()
        """
        return """You are a technical expert helping users understand cables, adapters, and connectivity.

Answer the user's question clearly and concisely:
- Focus on practical information
- Use simple language
- Provide specific examples when helpful
- Keep responses under 200 words

Do not recommend specific products - just explain concepts."""

    def format_blocked_request(
        self,
        reason: str,
        alternatives: Optional[List[str]] = None
    ) -> str:
        """
        Format a blocked request response.

        Args:
            reason: Why the request was blocked
            alternatives: Alternative approaches (optional)

        Returns:
            Formatted response string

        Example:
            >>> response = prompts.format_blocked_request(
            ...     reason="Daisy-chaining not supported",
            ...     alternatives=["docking station", "individual cables"]
            ... )
        """
        response = reason

        if alternatives:
            response += "\n\nAlternatives:\n"
            for alt in alternatives:
                response += f"• {alt}\n"

        return response.strip()

    def format_greeting_response(self) -> str:
        """
        Format a greeting response.

        Tone: Confident, warm, expert - like StarTech's best CSR.
        Not scripted or corporate.

        Returns:
            Greeting message
        """
        return "Hey! I'm your StarTech connectivity expert. What challenge can I help you solve today?"

    def format_farewell_response(self) -> str:
        """
        Format a farewell response.

        Should match the greeting's tone: confident, warm, expert energy.
        Reinforces StarTech expertise and leaves a positive final impression.

        Returns:
            Farewell message
        """
        return "Happy to help! If you need anything else with your connectivity setup, I'm here."

    def format_no_results_response(
        self,
        query: str,
        suggestions: Optional[List[str]] = None
    ) -> str:
        """
        Format a no results response.

        Args:
            query: Original user query
            suggestions: Suggested alternatives

        Returns:
            Formatted no results message

        Example:
            >>> response = prompts.format_no_results_response(
            ...     query="50ft Thunderbolt cable",
            ...     suggestions=["Try a shorter cable", "Consider an active cable"]
            ... )
        """
        response = (
            f"I couldn't find products matching '{query}'. "
            "This might mean:\n"
            "• The product doesn't exist in our catalog\n"
            "• Try different search terms\n"
            "• The specifications might not be available"
        )

        if suggestions:
            response += "\n\nSuggestions:\n"
            for suggestion in suggestions:
                response += f"• {suggestion}\n"

        return response.strip()

    def format_ambiguous_query_response(self) -> str:
        """
        Format response for ambiguous queries.

        Returns:
            Message asking for clarification

        Example:
            >>> response = prompts.format_ambiguous_query_response()
        """
        return (
            "I'd like to help you find the right product, but I need a bit more information. "
            "Could you tell me:\n"
            "• What devices you're connecting?\n"
            "• What you're trying to accomplish?\n"
            "• Any specific requirements (length, features, etc.)?"
        )

    def format_product_summary(
        self,
        product_count: int,
        query: str
    ) -> str:
        """
        Format a product results summary.

        Args:
            product_count: Number of products found
            query: Original query

        Returns:
            Summary message

        Example:
            >>> summary = prompts.format_product_summary(5, "USB-C cable")
        """
        if product_count == 0:
            return f"No products found for '{query}'."
        elif product_count == 1:
            return f"Found 1 product for '{query}':"
        else:
            return f"Found {product_count} products for '{query}':"

    def format_context_note(
        self,
        context_type: str,
        details: Optional[str] = None
    ) -> str:
        """
        Format a context note to add to responses.

        Args:
            context_type: Type of context (e.g., "4k", "long_cable")
            details: Additional details (optional)

        Returns:
            Formatted context note

        Example:
            >>> note = prompts.format_context_note("4k", "60Hz recommended")
        """
        templates = {
            "4k": "💡 Tip: For reliable 4K support, look for cables certified for 4K/60Hz or higher.",
            "long_cable": "💡 Tip: For cables longer than 15ft, consider an active cable or signal booster.",
            "thunderbolt": "💡 Tip: Thunderbolt cables support high-speed data (40Gbps) and video simultaneously.",
            "power_delivery": "💡 Tip: Many USB-C cables support Power Delivery for charging.",
        }

        note = templates.get(context_type, "")

        if details:
            note += f" {details}"

        return note

    def format_error_response(self, error_type: str) -> str:
        """
        Format an error response.

        Args:
            error_type: Type of error

        Returns:
            User-friendly error message

        Example:
            >>> error = prompts.format_error_response("search_failed")
        """
        templates = {
            "search_failed": (
                "I encountered an issue while searching. "
                "Please try again or rephrase your query."
            ),
            "invalid_input": (
                "I didn't quite understand that. "
                "Could you rephrase your question?"
            ),
            "system_error": (
                "Something went wrong on my end. "
                "Please try again in a moment."
            ),
        }

        return templates.get(error_type, "An error occurred. Please try again.")


class ResponseTemplates:
    """
    Additional response templates for common scenarios.

    Provides templates for:
    - Product recommendations
    - Comparison responses
    - Technical explanations
    """

    @staticmethod
    def format_connector_explanation(connector_type: str) -> str:
        """
        Get explanation for a connector type.

        Args:
            connector_type: Type of connector

        Returns:
            Explanation text

        Example:
            >>> explanation = ResponseTemplates.format_connector_explanation("USB-C")
        """
        explanations = {
            "USB-C": (
                "USB-C is a versatile connector that supports data transfer, "
                "video output, and power delivery in a single cable."
            ),
            "HDMI": (
                "HDMI is the standard for video and audio transmission, "
                "commonly used for TVs, monitors, and projectors."
            ),
            "DisplayPort": (
                "DisplayPort is designed for computer displays and supports "
                "high resolutions and refresh rates."
            ),
            "Thunderbolt": (
                "Thunderbolt combines data, video, and power in one connection "
                "with speeds up to 40Gbps."
            ),
        }

        return explanations.get(
            connector_type,
            f"Information about {connector_type} connectors."
        )

    @staticmethod
    def format_feature_explanation(feature: str) -> str:
        """
        Get explanation for a technical feature.

        Args:
            feature: Feature name

        Returns:
            Explanation text

        Example:
            >>> explanation = ResponseTemplates.format_feature_explanation("4K")
        """
        explanations = {
            "4K": (
                "4K (3840×2160) provides four times the resolution of 1080p "
                "for sharper images and more detail."
            ),
            "8K": (
                "8K (7680×4320) offers exceptional detail, ideal for "
                "large displays and professional applications."
            ),
            "HDR": (
                "HDR (High Dynamic Range) expands contrast and color range "
                "for more realistic images."
            ),
            "Power Delivery": (
                "Power Delivery enables USB-C cables to charge devices "
                "at higher wattages (up to 100W)."
            ),
        }

        return explanations.get(
            feature,
            f"Technical feature: {feature}"
        )


# Singleton instance for easy access
_system_prompts = SystemPrompts()
_response_templates = ResponseTemplates()


def get_system_prompts() -> SystemPrompts:
    """
    Get the system prompts instance.

    Returns:
        SystemPrompts instance

    Example:
        >>> prompts = get_system_prompts()
        >>> greeting = prompts.format_greeting_response()
    """
    return _system_prompts


def get_response_templates() -> ResponseTemplates:
    """
    Get the response templates instance.

    Returns:
        ResponseTemplates instance

    Example:
        >>> templates = get_response_templates()
        >>> explanation = templates.format_connector_explanation("HDMI")
    """
    return _response_templates
