"""Thin Claude cog wrapper around extracted helper modules."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from discord import ApplicationContext, Attachment
from discord.commands import SlashCommandGroup, option
from discord.ext import commands, tasks

from discord_claude.config.auth import GUILD_IDS, SHOW_COST_EMBEDS, validate_required_config
from discord_claude.logging_setup import bind_request_id
from discord_claude.util import ChatCompletionParameters, Conversation, ConversationKey, ToolChoice

from .attachments import (
    SUPPORTED_DOCUMENT_TYPES,
    SUPPORTED_IMAGE_TYPES,
    build_attachment_content_block,
    fetch_attachment_bytes,
)
from .chat import (
    build_api_params,
    build_thinking_config,
    call_api_with_tool_loop,
    compact_conversation,
    handle_check_permissions,
    handle_on_message,
    run_chat_command,
    run_effort_command,
    validate_request_configuration,
)
from .chat import (
    handle_new_message_in_conversation as run_followup_message,
)
from .chat import (
    keep_typing as keep_typing_loop,
)
from .client import (
    APIConnectionError,
    APIError,
    build_claude_client,
    close_http_session,
    get_http_session,
)
from .command_options import (
    CHAT_MODEL_CHOICES,
    RESPONSE_EFFORT_CHOICES,
    THINKING_DISPLAY_CHOICES,
    TOOL_CHOICE_CHOICES,
)
from .embeds import (
    append_citations_embed,
    append_compaction_embed,
    append_context_warning_embed,
    append_pricing_embed,
    append_response_embeds,
    append_stop_reason_embed,
    append_thinking_embeds,
)
from .models import ParsedResponse, ToolHandler, UsageTotals
from .responses import extract_response_content
from .state import (
    cleanup_conversation,
    prune_runtime_state,
    stop_conversation,
    strip_previous_view,
    track_daily_cost,
)
from .tool_handlers import MemoryToolHandler, default_tool_handlers

__all__ = [
    "SUPPORTED_DOCUMENT_TYPES",
    "SUPPORTED_IMAGE_TYPES",
    "APIConnectionError",
    "APIError",
    "ClaudeCog",
    "MemoryToolHandler",
    "ParsedResponse",
    "ToolChoice",
    "UsageTotals",
    "append_citations_embed",
    "append_compaction_embed",
    "append_context_warning_embed",
    "append_pricing_embed",
    "append_response_embeds",
    "append_stop_reason_embed",
    "append_thinking_embeds",
    "build_attachment_content_block",
    "extract_response_content",
]


class ClaudeCog(commands.Cog):
    claude = SlashCommandGroup("claude", "Claude AI commands", guild_ids=GUILD_IDS)

    def __init__(self, bot):
        validate_required_config()
        self.bot = bot
        self.client = build_claude_client()
        # Use a module logger only; host applications should configure logging
        # centrally at process startup (handlers/levels/formatters).
        self.logger = logging.getLogger(__name__)
        self.show_cost_embeds = SHOW_COST_EMBEDS

        self.conversations: dict[ConversationKey, Conversation] = {}
        self.views = {}
        self.last_view_messages = {}
        self.daily_costs: dict[tuple[int, str], tuple[float, datetime]] = {}
        self._http_session = None
        self._session_lock = asyncio.Lock()
        self._tool_handlers: dict[str, ToolHandler] = default_tool_handlers()

    async def _get_http_session(self):
        return await get_http_session(self)

    async def _compact_conversation(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> str:
        return await compact_conversation(self, messages, system=system)

    async def _strip_previous_view(self, user) -> None:
        await strip_previous_view(self, user)

    async def _cleanup_conversation(self, user) -> None:
        await cleanup_conversation(self, user)

    async def _stop_conversation(self, conversation_key: ConversationKey, user) -> None:
        await stop_conversation(self, conversation_key, user)

    def _track_daily_cost(
        self,
        user_id: int,
        model: str,
        parsed: ParsedResponse,
        advisor_model: str | None = None,
    ) -> tuple[float, float]:
        return track_daily_cost(self, user_id, model, parsed, advisor_model=advisor_model)

    async def _fetch_attachment_bytes(self, attachment: Attachment) -> bytes | None:
        return await fetch_attachment_bytes(self, attachment)

    async def _prune_runtime_state(self) -> None:
        await prune_runtime_state(self)

    @tasks.loop(minutes=15)
    async def _runtime_cleanup_task(self) -> None:
        await self._prune_runtime_state()

    @_runtime_cleanup_task.before_loop
    async def _before_runtime_cleanup_task(self) -> None:
        await self.bot.wait_until_ready()

    async def cog_before_invoke(self, ctx: ApplicationContext) -> None:
        """Bind a fresh request id on every slash-command entry into this cog."""
        bind_request_id()

    def cog_unload(self):
        if self._runtime_cleanup_task.is_running():
            self._runtime_cleanup_task.cancel()
        close_http_session(self)

    async def _call_api_with_tool_loop(
        self,
        api_params: dict[str, Any],
        messages: list[dict[str, Any]],
        user_id: int,
        max_iterations: int = 10,
        progress_callback=None,
    ):
        return await call_api_with_tool_loop(
            self,
            api_params=api_params,
            messages=messages,
            user_id=user_id,
            max_iterations=max_iterations,
            progress_callback=progress_callback,
        )

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any], user_id: int) -> str:
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"
        return await handler.execute(tool_input, user_id)

    def register_tool_handler(self, tool_name: str, handler: ToolHandler) -> None:
        self._tool_handlers[tool_name] = handler

    def unregister_tool_handler(self, tool_name: str) -> ToolHandler | None:
        return self._tool_handlers.pop(tool_name, None)

    @staticmethod
    def _build_thinking_config(params: ChatCompletionParameters) -> dict[str, Any] | None:
        return build_thinking_config(params)

    @classmethod
    def _validate_request_configuration(cls, params: ChatCompletionParameters) -> str | None:
        return validate_request_configuration(params)

    @classmethod
    def _build_api_params(
        cls,
        params: ChatCompletionParameters,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_api_params(params, messages)

    async def handle_new_message_in_conversation(self, message, conversation: Conversation):
        await run_followup_message(self, message, conversation)

    async def keep_typing(self, channel):
        await keep_typing_loop(self, channel)

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info("Logged in as %s (ID: %s)", self.bot.user, self.bot.owner_id)
        self.logger.info("Attempting to sync commands for guilds: %s", GUILD_IDS)
        if not self._runtime_cleanup_task.is_running():
            self._runtime_cleanup_task.start()
        try:
            await self.bot.sync_commands()
            self.logger.info("Commands synchronized successfully.")
        except Exception as error:
            self.logger.error("Error during command synchronization: %s", error, exc_info=True)

    @commands.Cog.listener()
    async def on_message(self, message):
        bind_request_id()
        await handle_on_message(self, message)

    @commands.Cog.listener()
    async def on_error(self, event, *args, **kwargs):
        self.logger.error("Error in event %s: %s %s", event, args, kwargs, exc_info=True)

    @claude.command(
        name="check_permissions",
        description="Check if bot has necessary permissions in this channel",
    )
    async def check_permissions(self, ctx: ApplicationContext):
        await handle_check_permissions(ctx)

    @claude.command(
        name="chat",
        description="Starts a conversation with Claude.",
    )
    @option("prompt", description="Prompt", required=True, type=str)
    @option(
        "system",
        description="System prompt to set Claude's behavior. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "model",
        description="Choose from the following Claude models. (default: Claude Opus 5.5. warning: Opus is expensive!)",
        required=False,
        choices=CHAT_MODEL_CHOICES,
        type=str,
    )
    @option(
        "attachment",
        description="Attach an image, PDF, text document, or code file. (default: not set)",
        required=False,
        type=Attachment,
    )
    @option(
        "max_tokens",
        description="Maximum tokens in the response. (default: 16384)",
        required=False,
        type=int,
    )
    @option(
        "temperature",
        description="(Advanced) Controls the randomness of the model. 0.0 to 1.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_p",
        description="(Advanced) Nucleus sampling. 0.0 to 1.0. (default: not set)",
        required=False,
        type=float,
    )
    @option(
        "top_k",
        description="(Advanced) Limits sampling to top K tokens. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "effort",
        description="Control response effort: low, medium, high, xhigh, max (higher = more thorough). (default: not set)",
        required=False,
        choices=RESPONSE_EFFORT_CHOICES,
        type=str,
    )
    @option(
        "thinking_budget",
        description="Token budget for legacy extended thinking models. (default: not set)",
        required=False,
        type=int,
    )
    @option(
        "thinking_display",
        description="Summarized reasoning, or progress lines between tool calls (Fable, Opus 5.5). (default: summarized)",
        required=False,
        choices=THINKING_DISPLAY_CHOICES,
        type=str,
    )
    @option(
        "web_search",
        description="Enable web search to find current information. (default: true)",
        required=False,
        type=bool,
    )
    @option(
        "web_fetch",
        description="Enable web fetch to retrieve full web page content. (default: true)",
        required=False,
        type=bool,
    )
    @option(
        "code_execution",
        description="Enable code execution to run code in a sandbox. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "memory",
        description="Enable memory to save and recall information across conversations. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "edit_reply",
        description="Let Claude edit its earlier replies in place when you ask for a fix. (default: true)",
        required=False,
        type=bool,
    )
    @option(
        "advisor",
        description="Enable the Anthropic advisor beta for strategic mid-generation guidance. (default: false)",
        required=False,
        type=bool,
    )
    @option(
        "mcp",
        description="Comma-separated MCP preset names to enable. (default: not set)",
        required=False,
        type=str,
    )
    @option(
        "tool_choice",
        description="Tool behavior when tools are enabled. (default: Anthropic default)",
        required=False,
        choices=TOOL_CHOICE_CHOICES,
        type=str,
    )
    async def chat(
        self,
        ctx: ApplicationContext,
        prompt: str,
        model: str = "claude-opus-5-5",
        system: str | None = None,
        attachment: Attachment | None = None,
        max_tokens: int = 16384,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        thinking_budget: int | None = None,
        thinking_display: str = "summarized",
        web_search: bool = True,
        web_fetch: bool = True,
        code_execution: bool = False,
        memory: bool = False,
        edit_reply: bool = True,
        advisor: bool = False,
        mcp: str | None = None,
        tool_choice: str | None = None,
    ):
        await run_chat_command(
            self,
            ctx=ctx,
            prompt=prompt,
            model=model,
            system=system,
            attachment=attachment,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            effort=effort,
            thinking_budget=thinking_budget,
            thinking_display=thinking_display,
            web_search=web_search,
            web_fetch=web_fetch,
            code_execution=code_execution,
            memory=memory,
            edit_reply=edit_reply,
            advisor=advisor,
            mcp=mcp,
            tool_choice=tool_choice,
        )

    @claude.command(
        name="effort",
        description="Change the effort level of your active conversation in this channel.",
    )
    @option(
        "effort",
        description="Effort for your following messages: low, medium, high, xhigh, or max.",
        required=True,
        choices=RESPONSE_EFFORT_CHOICES,
        type=str,
    )
    async def effort(self, ctx: ApplicationContext, effort: str):
        await run_effort_command(self, ctx=ctx, effort=effort)
