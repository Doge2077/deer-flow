"""Middleware for automatic thread title generation."""

from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from src.config.title_config import get_title_config
from src.models import create_chat_model


class TitleMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    state_schema = TitleMiddlewareState

    def _should_generate_title(self, state: TitleMiddlewareState) -> bool:
        """Check if we should generate a title for this thread."""
        config = get_title_config()
        if not config.enabled:
            return False

        # Check if thread already has a title in state
        if state.get("title"):
            return False

        # Check if this is the first turn (has at least one user message and one assistant response)
        messages = state.get("messages", [])
        if len(messages) < 2:
            return False

        # Count user and assistant messages
        user_messages = [m for m in messages if m.type == "human"]
        assistant_messages = [m for m in messages if m.type == "ai"]

        # Generate title after first complete exchange
        return len(user_messages) == 1 and len(assistant_messages) >= 1

    def _get_prompt_context(self, state: TitleMiddlewareState) -> tuple[str, str, str]:
        """Build the prompt and normalized message snippets for title generation."""
        config = get_title_config()
        messages = state.get("messages", [])

        # Get first user message and first assistant response
        user_msg_content = next((m.content for m in messages if m.type == "human"), "")
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        # Ensure content is string (LangChain messages can have list content)
        user_msg = str(user_msg_content) if user_msg_content else ""
        assistant_msg = str(assistant_msg_content) if assistant_msg_content else ""
        prompt = config.prompt_template.format(
            max_words=config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg, assistant_msg

    def _normalize_title(self, content: object) -> str:
        """Normalize model output into a bounded plain-text title."""
        config = get_title_config()
        title_content = str(content) if content else ""
        title = title_content.strip().strip('"').strip("'")
        return title[: config.max_chars] if len(title) > config.max_chars else title

    def _fallback_title(self, user_msg: str) -> str:
        """Fallback title when the title model fails."""
        config = get_title_config()
        fallback_chars = min(config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            return user_msg[:fallback_chars].rstrip() + "..."
        return user_msg if user_msg else "New Conversation"

    async def _generate_title(self, state: TitleMiddlewareState) -> str:
        """Generate a concise title based on the conversation."""
        prompt, user_msg, _assistant_msg = self._get_prompt_context(state)

        # Use a lightweight model to generate title
        model = create_chat_model(thinking_enabled=False)

        try:
            response = await model.ainvoke(prompt)
            return self._normalize_title(response.content)
        except Exception as e:
            print(f"Failed to generate title: {e}")
            return self._fallback_title(user_msg)

    def _generate_title_sync(self, state: TitleMiddlewareState) -> str:
        """Synchronous variant used by embedded client flows."""
        prompt, user_msg, _assistant_msg = self._get_prompt_context(state)

        model = create_chat_model(thinking_enabled=False)

        try:
            response = model.invoke(prompt)
            return self._normalize_title(response.content)
        except Exception as e:
            print(f"Failed to generate title: {e}")
            return self._fallback_title(user_msg)

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        """Generate and set thread title for synchronous agent execution."""
        if self._should_generate_title(state):
            title = self._generate_title_sync(state)
            print(f"Generated thread title: {title}")
            return {"title": title}

        return None

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        """Generate and set thread title after the first agent response."""
        if self._should_generate_title(state):
            title = await self._generate_title(state)
            print(f"Generated thread title: {title}")

            # Store title in state (will be persisted by checkpointer if configured)
            return {"title": title}

        return None
