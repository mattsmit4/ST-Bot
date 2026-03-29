"""
Educational intent handler for ST-Bot.

Provides informative responses to technical questions like
"What's the difference between Cat6 and Cat6a?"
"""

import os

from handlers.base import BaseHandler, HandlerContext, HandlerResult


EDUCATIONAL_SYSTEM_PROMPT = """You are a StarTech.com product expert helping customers understand technology differences.

Guidelines:
- Use a comparison table (Markdown) when comparing two or more technologies.
- Be factual. If you are unsure about specific specs (e.g., a very new standard), say so rather than guessing.
- Focus on practical differences that matter for buying decisions.
- Keep responses under 400 words.
- End with a brief follow-up question to guide the customer toward products (e.g., "What are you trying to connect?" or "Would you like me to show you some options?").
- Do NOT invent specific StarTech.com product SKUs or prices."""


class EducationalHandler(BaseHandler):
    """Handles educational questions using the LLM."""

    def handle(self, ctx: HandlerContext) -> HandlerResult:
        meta = ctx.intent.meta_info or {}
        topic = meta.get('topic')
        topic1 = meta.get('topic1')
        topic2 = meta.get('topic2')

        ctx.add_debug(f"EDUCATIONAL: type={meta.get('type')}, topic={topic}")

        topic_desc = topic or (f"{topic1} vs {topic2}" if topic1 and topic2 else "this topic")

        llm_response = self._answer_with_llm(ctx.query, topic_desc)
        if llm_response:
            return HandlerResult(response=llm_response)

        # Fallback if LLM call fails entirely
        return HandlerResult(
            response="I'd be happy to help explain that! Could you be more specific about "
                     "what you'd like to know? Or if you're looking for products, "
                     "just tell me what you're trying to connect."
        )

    def _answer_with_llm(self, query: str, topic_desc: str) -> str | None:
        from core.openai_client import get_openai_client

        client = get_openai_client()
        if not client:
            return None

        try:
            response = client.chat.completions.create(
                model=os.environ.get('OPENAI_CHAT_MODEL', 'gpt-5.4-nano'),
                messages=[
                    {"role": "system", "content": EDUCATIONAL_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_completion_tokens=600,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
