"""
LLM-Based Clarification Question Generator

Replaces hard-coded clarification question templates with context-aware,
LLM-generated questions. When a user sends a vague query like "I need a cable",
this module generates a natural follow-up question specific to what the user
mentioned (e.g., asking about connector types) instead of generic use-case options.

Design:
- LLM generates the question text and context-aware options
- Hard-coded templates in core/clarification.py serve as permanent fallback
- Follows same patterns as llm_filter_extractor.py (retry, JSON response, graceful degradation)
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any

from core.models import VagueQueryType, ClarificationMissing, PendingClarification
from core.api_retry import RetryHandler, RetryConfig
from core.openai_client import get_openai_client

_logger = logging.getLogger(__name__)


CLARIFICATION_QUESTION_PROMPT = """You are a conversational product assistant for StarTech.com, a company that sells cables, adapters, docks, hubs, and connectivity products.

Your job is to ask ONE clarifying question when a user's product request is too vague to search.

## Rules

1. **Be specific to what the user mentioned**:
   - "I need a cable" → Ask about what devices they're connecting or what connector types
   - "I need ports" → Ask what KIND of ports (USB-A, USB-C, video outputs)
   - "something for my computer" → Ask what they're trying to DO with their computer
   - "I need an adapter" → Ask what they're converting FROM and TO

2. **Generate 3-4 context-aware options** relevant to the user's specific request:
   - For cables: device pairs, connector types, or common cable scenarios
   - For ports/hubs: port types, quantity needs
   - For generic: common use cases (video output, data transfer, charging, networking)
   - Always include "Something else" as the last option

3. **On follow-up questions (questions_asked >= 1)**:
   - Reference what the user already told you
   - Ask about what's STILL missing, not what you already know
   - Be more specific and direct

4. **On 3rd+ questions (questions_asked >= 2)**:
   - Offer specific, concrete scenarios the user can pick from
   - Make it easy to just choose one
   - Include a "just tell me your devices" fallback

5. **Keep it short**: 1-2 sentences for the question, plus option bullets

6. **Never mention**: product SKUs, prices, specific product names, or technical jargon unless the user used it first

## Response Format
Return ONLY valid JSON:
{
    "acknowledgment": "Brief acknowledgment (only for initial question, empty string for follow-ups)",
    "question": "Your clarifying question",
    "options": ["Option 1", "Option 2", "Option 3", "Something else"]
}

## Examples

User query: "I need a cable"
Missing: use_case, connector_to
{
    "acknowledgment": "I can help you find the right cable!",
    "question": "What are you trying to connect?",
    "options": ["Laptop to external monitor or TV", "Phone or tablet to a display", "Two devices via USB", "Something else"]
}

User query: "I need more ports"
Missing: use_case
{
    "acknowledgment": "I can help you expand your ports!",
    "question": "What type of ports do you need more of?",
    "options": ["USB-A ports (for mice, keyboards, drives)", "USB-C or Thunderbolt ports", "Video outputs (HDMI, DisplayPort)", "Something else"]
}

User query: "I need a cable"
Already known: connector_from=USB-C
Missing: connector_to
Questions asked: 1
{
    "acknowledgment": "",
    "question": "Got it, USB-C on one end. What does the other device have?",
    "options": ["HDMI (monitors, TVs)", "DisplayPort (monitors)", "USB-A (older devices)", "Not sure \\u2014 tell me the device"]
}

User query: "I need a cable"
Already known: nothing
Missing: use_case, connector_to
Questions asked: 2
{
    "acknowledgment": "",
    "question": "Let me try a different approach. Which of these sounds closest?",
    "options": ["Connect my laptop to a monitor", "Connect my phone to a TV", "Extend a USB device across a room", "Just tell me your two devices and I'll figure it out"]
}"""


class LLMClarificationGenerator:
    """
    Generates context-aware clarification questions using LLM.

    Falls back to None on failure (caller uses hard-coded templates).

    Usage:
        generator = LLMClarificationGenerator()
        result = generator.generate_initial_response(
            original_query="I need a cable",
            vague_type=VagueQueryType.CABLE,
            missing_info=[ClarificationMissing.USE_CASE],
            collected_info={},
        )
        # Returns: "I can help you find the right cable! What are you trying to connect?\n- Laptop to ..."
    """

    def __init__(self, model: str = None, temperature: float = 0.4):
        self.model = model or os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano')
        self.temperature = temperature
        self.retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.5,
            max_delay=5.0
        )

    def generate_initial_response(
        self,
        original_query: str,
        vague_type: VagueQueryType,
        missing_info: List[ClarificationMissing],
        collected_info: Dict[str, Any],
    ) -> Optional[str]:
        """Generate the first clarification response (acknowledgment + question).

        Returns formatted string or None on failure.
        """
        result = self._generate(
            original_query=original_query,
            vague_type=vague_type,
            missing_info=missing_info,
            collected_info=collected_info,
            questions_asked=0,
        )
        if result:
            ack = result.get('acknowledgment', '')
            question = self._format_question_with_options(result)
            return f"{ack} {question}".strip() if ack else question
        return None

    def generate_question(
        self,
        original_query: str,
        vague_type: VagueQueryType,
        missing_info: List[ClarificationMissing],
        collected_info: Dict[str, Any],
        questions_asked: int,
        conversation_history: list = None,
    ) -> Optional[str]:
        """Generate a follow-up clarification question.

        Returns formatted string or None on failure.
        """
        result = self._generate(
            original_query=original_query,
            vague_type=vague_type,
            missing_info=missing_info,
            collected_info=collected_info,
            questions_asked=questions_asked,
            conversation_history=conversation_history,
        )
        if result:
            return self._format_question_with_options(result)
        return None

    def _generate(
        self,
        original_query: str,
        vague_type: VagueQueryType,
        missing_info: List[ClarificationMissing],
        collected_info: Dict[str, Any],
        questions_asked: int,
        conversation_history: list = None,
    ) -> Optional[Dict]:
        """Call LLM and return parsed JSON dict, or None on failure."""
        client = get_openai_client()
        if not client:
            return None

        user_prompt = self._build_user_prompt(
            original_query, vague_type, missing_info,
            collected_info, questions_asked, conversation_history,
        )

        try:
            handler = RetryHandler(
                config=self.retry_config,
                operation_name="llm_clarification_question"
            )
            response = handler.execute(
                lambda: self._call_openai(client, user_prompt),
                fallback=None,
            )
            if response is None:
                return None
            return self._parse_response(response)
        except Exception as e:
            _logger.error("LLM clarification error", extra={
                "error": str(e),
                "query": original_query,
            })
            return None

    def _build_user_prompt(
        self,
        original_query: str,
        vague_type: VagueQueryType,
        missing_info: List[ClarificationMissing],
        collected_info: Dict[str, Any],
        questions_asked: int,
        conversation_history: list = None,
    ) -> str:
        """Build the user prompt with all available context."""
        parts = [f'User query: "{original_query}"']
        parts.append(f"Vague type: {vague_type.value}")

        if collected_info:
            readable = ", ".join(f"{k}={v}" for k, v in collected_info.items())
            parts.append(f"Already known: {readable}")
        else:
            parts.append("Already known: nothing yet")

        missing_str = ", ".join(m.value for m in missing_info)
        parts.append(f"Still missing: {missing_str}")
        parts.append(f"Questions asked so far: {questions_asked}")

        if conversation_history:
            history_lines = []
            for msg in conversation_history[-6:]:
                role = "User" if msg.role == "user" else "Bot"
                content = msg.content[:150]
                history_lines.append(f"  {role}: {content}")
            parts.append("Recent conversation:\n" + "\n".join(history_lines))

        return "\n".join(parts)

    def _call_openai(self, client, user_prompt: str) -> str:
        """Make the OpenAI API call."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLARIFICATION_QUESTION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_completion_tokens=300,
            response_format={"type": "json_object"},
            timeout=30.0,
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response. Returns dict or None."""
        try:
            data = json.loads(response)
            if 'question' not in data:
                _logger.warning("LLM clarification missing 'question' key")
                return None
            return data
        except json.JSONDecodeError:
            _logger.warning("LLM clarification returned invalid JSON")
            return None

    def _format_question_with_options(self, result: Dict) -> str:
        """Format question + options into a display string with markdown bullets."""
        question = result.get('question', '')
        options = result.get('options', [])
        if options:
            options_text = "\n".join(f"- {opt}" for opt in options)
            return f"{question}\n{options_text}"
        return question
