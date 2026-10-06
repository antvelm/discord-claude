import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from discord import ApplicationContext, Attachment, Colour, Embed

from discord_claude.config.auth import SHOW_COST_EMBEDS
from discord_claude.config.mcp import parse_mcp_preset_names, resolve_mcp_presets
from discord_claude.util import (
    ADAPTIVE_ONLY_THINKING_MODELS,
    ADAPTIVE_THINKING_MODELS,
    ADVISOR_BETA,
    ADVISOR_MAX_USES,
    ADVISOR_MODEL_COMPATIBILITY,
    ADVISOR_TOOL_NAME,
    ADVISOR_TOOL_TYPE,
    CACHE_TTL,
    COMPACTION_MODELS,
    EFFORT_LEVELS,
    EXTENDED_THINKING_MODELS,
    FORCED_TOOL_CHOICE_UNSUPPORTED_MODELS,
    MODEL_CONTEXT_WINDOWS,
    PER_MESSAGE_EFFORT_BETA,
    PER_MESSAGE_EFFORT_MODELS,
    PROGRAMMATIC_TOOL_CALLING_UNSUPPORTED_MODELS,
    REFUSAL_FALLBACK_BETA,
    REFUSAL_FALLBACK_MODEL,
    REFUSAL_FALLBACK_MODELS,
    SAMPLING_LOCKED_MODELS,
    THINKING_DISPLAY_SUMMARIZED,
    THINKING_DISPLAY_UPDATES,
    THINKING_DISPLAY_UPDATES_BETA,
    THINKING_DISPLAY_UPDATES_MODELS,
    ChatCompletionParameters,
    Conversation,
    ConversationKey,
    ToolChoice,
    UsageTotals,
    format_anthropic_error,
    get_default_advisor_model,
    manual_compaction_trigger,
    supported_effort_levels,
    truncate_text,
)

from .attachments import build_attachment_content_block, fetch_attachment_bytes
from .embed_delivery import send_embed_batches
from .embeds import (
    append_citations_embed,
    append_compaction_embed,
    append_context_warning_embed,
    append_fallback_embed,
    append_pricing_embed,
    append_response_embeds,
    append_stop_reason_embed,
    append_thinking_embeds,
)
from .responses import ParsedResponse, extract_response_content
from .state import compact_conversation, create_button_view
from .tool_registry import TOOL_REGISTRY, get_anthropic_tools


async def keep_typing(cog, channel) -> None:
    """Keep the Discord typing indicator alive while Claude is working."""
    try:
        cog.logger.debug("Starting typing indicator loop in channel %s", channel.id)
        while True:
            async with channel.typing():
                cog.logger.debug("Sent typing indicator to channel %s", channel.id)
                await asyncio.sleep(5)
    except asyncio.CancelledError:
        cog.logger.debug("Typing indicator cancelled for channel %s", channel.id)
        raise


def handle_check_permissions(ctx: ApplicationContext) -> Any:
    """Check whether the bot can read the current server channel."""
    if ctx.guild is None or not hasattr(ctx.channel, "permissions_for"):
        return ctx.respond("This command must be used in a server channel.")
    permissions = ctx.channel.permissions_for(ctx.guild.me)  # type: ignore[union-attr]
    if permissions.read_messages and permissions.read_message_history:
        return ctx.respond("Bot has permission to read messages and message history.")
    return ctx.respond("Bot is missing necessary permissions in this channel.")


ProgressCallback = Callable[[str], Awaitable[None]]


def build_thinking_config(params: ChatCompletionParameters) -> dict[str, Any] | None:
    """Return the Anthropic thinking configuration for the current model/settings."""
    if params.model in ADAPTIVE_THINKING_MODELS:
        return {"type": "adaptive", "display": params.thinking_display}
    if params.thinking_budget is not None and params.model in EXTENDED_THINKING_MODELS:
        return {
            "type": "enabled",
            "budget_tokens": params.thinking_budget,
        }
    return None


def validate_request_configuration(params: ChatCompletionParameters) -> str | None:
    """Validate request options before dispatching to Anthropic."""
    tool_choice = params.tool_choice
    has_thinking = build_thinking_config(params) is not None
    advisor_model = params.advisor_model
    sampling_parameters = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
    }

    if params.model in ADAPTIVE_ONLY_THINKING_MODELS and params.thinking_budget is not None:
        return (
            f"`{params.model}` only supports adaptive thinking. "
            "Leave `thinking_budget` unset for this model."
        )

    if (
        params.thinking_display == THINKING_DISPLAY_UPDATES
        and params.model not in THINKING_DISPLAY_UPDATES_MODELS
    ):
        supported_models = ", ".join(f"`{m}`" for m in sorted(THINKING_DISPLAY_UPDATES_MODELS))
        return (
            f"`{params.model}` does not write progress updates between tool calls; "
            f"thinking display `updates` is only available on {supported_models}. "
            "Use `summarized` for this model."
        )

    if params.model in SAMPLING_LOCKED_MODELS:
        overridden_parameters = [
            name for name, value in sampling_parameters.items() if value is not None
        ]
        if overridden_parameters:
            formatted_parameters = ", ".join(f"`{name}`" for name in overridden_parameters)
            return (
                f"`{params.model}` does not support custom sampling parameters. "
                f"Leave {formatted_parameters} unset for this model."
            )

    if params.effort is not None:
        supported_levels = supported_effort_levels(params.model)
        if not supported_levels:
            return (
                f"`{params.model}` does not support the `effort` parameter. "
                "Leave `effort` unset for this model."
            )
        if params.effort not in supported_levels:
            formatted_levels = ", ".join(
                f"`{level}`" for level in EFFORT_LEVELS if level in supported_levels
            )
            return (
                f"`{params.model}` does not support effort `{params.effort}`. "
                f"Supported effort levels: {formatted_levels}."
            )

    if advisor_model is not None:
        compatible_models = ADVISOR_MODEL_COMPATIBILITY.get(params.model)
        if not compatible_models:
            supported_models = ", ".join(sorted(ADVISOR_MODEL_COMPATIBILITY))
            return (
                f"Advisor is not supported for `{params.model}`. "
                f"Supported executor models: {supported_models}."
            )
        if advisor_model not in compatible_models:
            supported_advisors = ", ".join(compatible_models)
            return (
                f"Advisor model `{advisor_model}` is not supported for `{params.model}`. "
                f"Supported advisor models: {supported_advisors}."
            )

    if tool_choice is None:
        return None

    choice_type = tool_choice["type"]

    if advisor_model is not None and choice_type == "none":
        return (
            "Advisor requires tool behavior `auto` or Anthropic default. "
            "Tool behavior `none` disables advisor calls."
        )

    if choice_type in {"any", "tool"} and params.model in FORCED_TOOL_CHOICE_UNSUPPORTED_MODELS:
        return (
            f"`{params.model}` does not support forced tool use (`any` / `tool`). "
            "Use tool behavior `auto` or `none` for this model."
        )

    if has_thinking and choice_type in {"any", "tool"}:
        return (
            "Thinking mode only supports tool behavior `auto` or `none`. "
            "Forced tool modes are not compatible with thinking."
        )

    if choice_type in {"any", "tool"} and not params.tools:
        return "Forced tool behavior requires at least one enabled tool."

    if choice_type == "tool":
        tool_name = tool_choice.get("name")
        if not tool_name:
            return "Tool behavior `tool` requires a tool name."
        if tool_name not in TOOL_REGISTRY:
            return f"Unknown forced tool `{tool_name}`."
        if tool_name not in params.tools:
            return (
                f"Tool behavior `tool` requires `{tool_name}` to be enabled in the "
                "conversation tools."
            )

    return None


def build_api_params(
    params: ChatCompletionParameters,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the Anthropic API parameter dict from ChatCompletionParameters."""
    api_params: dict[str, Any] = {
        "model": params.model,
        "max_tokens": params.max_tokens,
        "messages": messages,
    }
    thinking_config = build_thinking_config(params)
    if thinking_config is not None:
        api_params["thinking"] = thinking_config
    if params.effort is not None:
        output_config = api_params.setdefault("output_config", {})
        output_config["effort"] = params.effort
    if params.system:
        api_params["system"] = params.system
    # anthropic 1.x dropped temperature/top_p/top_k from the typed request
    # signature; extra_body merges them flat into the JSON request body.
    sampling_parameters = {
        name: value
        for name, value in (
            ("temperature", params.temperature),
            ("top_p", params.top_p),
            ("top_k", params.top_k),
        )
        if value is not None
    }
    if sampling_parameters:
        api_params["extra_body"] = sampling_parameters
    mcp_presets, mcp_error = resolve_mcp_presets(params.mcp_preset_names)
    if mcp_error:
        raise ValueError(mcp_error)

    api_tools = get_anthropic_tools(params.tools)
    if params.model in PROGRAMMATIC_TOOL_CALLING_UNSUPPORTED_MODELS:
        # The web tools' default allowed_callers needs programmatic tool calling,
        # which this model lacks; "direct" keeps them callable by the model itself.
        api_tools = [
            {**tool, "allowed_callers": ["direct"]}
            if str(tool.get("type", "")).startswith(("web_search_", "web_fetch_"))
            else tool
            for tool in api_tools
        ]
    if params.advisor_model is not None:
        api_tools.append(
            {
                "type": ADVISOR_TOOL_TYPE,
                "name": ADVISOR_TOOL_NAME,
                "model": params.advisor_model,
                "max_uses": ADVISOR_MAX_USES,
            }
        )
    if mcp_presets:
        api_params["mcp_servers"] = []
        for preset in mcp_presets:
            server_config: dict[str, Any] = {
                "type": "url",
                "url": preset.server_url,
                "name": preset.name,
            }
            if preset.authorization_env_var:
                from os import getenv

                authorization_token = getenv(preset.authorization_env_var)
                if authorization_token:
                    server_config["authorization_token"] = authorization_token
            api_params["mcp_servers"].append(server_config)

            toolset: dict[str, Any] = {
                "type": "mcp_toolset",
                "mcp_server_name": preset.name,
            }
            default_config: dict[str, Any] = {}
            if preset.allowed_tools:
                default_config["enabled"] = False
                toolset["configs"] = {
                    tool_name: {"enabled": True} for tool_name in preset.allowed_tools
                }
            if preset.defer_loading:
                default_config["defer_loading"] = True
            if default_config:
                toolset["default_config"] = default_config
            api_tools.append(toolset)

    if api_tools:
        api_params["tools"] = api_tools
        if params.tool_choice is not None:
            api_params["tool_choice"] = params.tool_choice
    return api_params


def _is_advisor_tool(tool_definition: Any) -> bool:
    """Return True when the tool definition is the advisor beta tool."""
    if not isinstance(tool_definition, dict):
        return False
    return (
        tool_definition.get("type") == ADVISOR_TOOL_TYPE
        and tool_definition.get("name") == ADVISOR_TOOL_NAME
    )


def is_effort_override_message(message: Any) -> bool:
    """True for the effort-only `role: "system"` messages `apply_effort_change` appends."""
    return (
        isinstance(message, dict)
        and message.get("role") == "system"
        and isinstance(message.get("output_config"), dict)
    )


def effort_override_message(level: str) -> dict[str, Any]:
    """Per-message effort change: applies from the next user turn without invalidating the cache."""
    return {"role": "system", "content": [], "output_config": {"effort": level}}


def current_effort(conversation: Conversation) -> str | None:
    """The effort that applies to later turns: the latest override if any, else the top-level value."""
    for message in reversed(conversation.messages):
        if is_effort_override_message(message):
            return message["output_config"].get("effort")
    return conversation.params.effort


def apply_effort_change(conversation: Conversation, level: str) -> str | None:
    """Change the conversation's effort for its following turns.

    Models in PER_MESSAGE_EFFORT_MODELS get an effort-only system message appended to
    the history (a trailing one is replaced, so consecutive changes do not accumulate)
    and keep the top-level `output_config.effort` unchanged, so the cached prompt prefix
    stays valid. Every other model gets the top-level value changed instead, which
    changes the rendered prompt and invalidates the cache. Returns a user-facing
    error, or None on success.
    """
    params = conversation.params
    supported_levels = supported_effort_levels(params.model)
    if not supported_levels:
        return f"`{params.model}` does not support the `effort` parameter."
    if level not in supported_levels:
        formatted_levels = ", ".join(
            f"`{effort}`" for effort in EFFORT_LEVELS if effort in supported_levels
        )
        return (
            f"`{params.model}` does not support effort `{level}`. "
            f"Supported effort levels: {formatted_levels}."
        )
    if params.model in PER_MESSAGE_EFFORT_MODELS:
        override = effort_override_message(level)
        if conversation.messages and is_effort_override_message(conversation.messages[-1]):
            conversation.messages[-1] = override
        else:
            conversation.messages.append(override)
    else:
        params.effort = level
    conversation.touch()
    return None


def _progress_lines(response: Any) -> list[str]:
    """Readable progress text in a tool_use response: text blocks plus any thinking
    block that came back non-empty (under `display: "updates"` those are the
    between-tool-call status lines; reasoning itself is returned empty)."""
    lines: list[str] = []
    for block in response.content:
        if block.type == "text" and (block.text or "").strip():
            lines.append(block.text.strip())
        elif block.type == "thinking" and (getattr(block, "thinking", "") or "").strip():
            lines.append(block.thinking.strip())
    return lines


def make_progress_poster(send, logger) -> ProgressCallback:
    """Build the callback that posts a progress update to Discord as a small embed."""

    async def _post(text: str) -> None:
        await send_embed_batches(
            send,
            embed=Embed(description=f"⏳ {truncate_text(text, 3900)}", color=Colour.greyple()),
            logger=logger,
        )

    return _post


async def call_api_with_tool_loop(
    cog,
    api_params: dict[str, Any],
    messages: list[dict[str, Any]],
    user_id: int,
    max_iterations: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> ParsedResponse:
    """Call the Anthropic API, handling tool-use loops and context management.

    ``progress_callback`` receives the readable progress text of each tool_use
    iteration when the request runs with ``thinking.display == "updates"``.
    """
    model = api_params.get("model", "")
    use_compaction = model in COMPACTION_MODELS
    tools = api_params.get("tools") or []
    has_tools = bool(tools)
    has_advisor = any(_is_advisor_tool(tool) for tool in tools)
    has_thinking = bool(api_params.get("thinking"))
    has_mcp = bool(api_params.get("mcp_servers"))
    thinking_display = (api_params.get("thinking") or {}).get("display")
    show_progress = thinking_display == THINKING_DISPLAY_UPDATES
    has_effort_override = any(is_effort_override_message(message) for message in messages)
    # The per-message effort beta is sent from the FIRST request on the models that
    # support it, not only once an override exists: adding the header mid-conversation
    # changes how the prompt is rendered and writes the whole prefix to cache again
    # instead of reading it, which is exactly the cost the per-message change is
    # meant to avoid.
    per_message_effort = model in PER_MESSAGE_EFFORT_MODELS or has_effort_override

    betas: list[str] = []
    edits: list[dict[str, Any]] = []
    if show_progress:
        betas.append(THINKING_DISPLAY_UPDATES_BETA)
    if per_message_effort:
        betas.append(PER_MESSAGE_EFFORT_BETA)
    if has_thinking:
        edits.append(
            {
                "type": "clear_thinking_20251015",
                "keep": {"type": "thinking_turns", "value": 2},
            }
        )
    if has_tools and not has_advisor:
        edits.append(
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 50000},
                "keep": {"type": "tool_uses", "value": 5},
            }
        )
    if edits:
        betas.append("context-management-2025-06-27")
    if has_advisor:
        betas.append(ADVISOR_BETA)
    if has_mcp:
        betas.append("mcp-client-2025-11-20")
    if use_compaction:
        betas.append("compact-2026-01-12")
        edits.append({"type": "compact_20260112"})
    if model in REFUSAL_FALLBACK_MODELS:
        betas.append(REFUSAL_FALLBACK_BETA)
        api_params["fallbacks"] = [{"model": REFUSAL_FALLBACK_MODEL}]

    totals = UsageTotals()
    context_window = MODEL_CONTEXT_WINDOWS.get(model, 200_000)
    compaction_trigger = manual_compaction_trigger(context_window)
    parsed: ParsedResponse | None = None

    async def compact_if_over_trigger() -> None:
        """Manual compaction for models outside COMPACTION_MODELS.

        The trigger compares the full prompt of the most recent response, cached
        tokens included, because the top-level cache_control reports most of a long
        history as cache reads rather than input_tokens. It runs before each
        follow-up request of the tool loop and once more after the final response,
        so the next user turn starts from the summary.
        """
        if use_compaction or totals.prompt_tokens <= compaction_trigger or len(messages) <= 1:
            return
        cog.logger.info(
            "Prompt tokens (%d) exceeded compaction trigger (%.0f), compacting...",
            totals.prompt_tokens,
            compaction_trigger,
        )
        await compact_conversation(
            cog, messages, system=api_params.get("system"), usage_totals=totals
        )
        totals.context_compacted = True
        # The history is now the summary alone; the next response measures it again.
        totals.prompt_tokens = 0

    async def finish(result: ParsedResponse) -> ParsedResponse:
        """Compact if the final response passed the trigger, then stamp the totals."""
        await compact_if_over_trigger()
        totals.apply_to(result, context_window)
        return result

    for iteration in range(max_iterations):
        await compact_if_over_trigger()

        api_params["messages"] = messages
        if betas:
            response = await cog.client.beta.messages.create(  # type: ignore[call-overload]
                **api_params,
                betas=betas,
                context_management={"edits": edits} if edits else None,  # type: ignore[arg-type]
                cache_control={"type": "ephemeral", "ttl": CACHE_TTL},
            )
        else:
            response = await cog.client.messages.create(
                **api_params,
                cache_control={"type": "ephemeral", "ttl": CACHE_TTL},
            )

        parsed = extract_response_content(response)
        parsed.stop_reason = response.stop_reason
        # Served-by signal for refusal fallbacks: a "fallback_message" entry in
        # usage.iterations covers every fallback-served turn, including sticky
        # turns that carry no "fallback" content block.
        usage_iterations = getattr(getattr(response, "usage", None), "iterations", None) or []
        if any(getattr(entry, "type", None) == "fallback_message" for entry in usage_iterations):
            parsed.served_model = getattr(response, "model", None)
            cog.logger.info(
                "Refusal fallback served this response: requested=%s served=%s",
                model,
                parsed.served_model,
            )
        stop_details = getattr(response, "stop_details", None)
        if stop_details is not None:
            parsed.stop_details = {
                "type": getattr(stop_details, "type", None),
                "category": getattr(stop_details, "category", None),
                "explanation": getattr(stop_details, "explanation", None),
            }
        totals.accumulate(getattr(response, "usage", None))

        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            return await finish(parsed)
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            cog.logger.info("pause_turn received, continuing (iteration %d)", iteration + 1)
            continue
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            if show_progress and progress_callback is not None:
                progress_text = "\n".join(_progress_lines(response))
                if progress_text:
                    try:
                        await progress_callback(progress_text)
                    except Exception:
                        cog.logger.warning("Failed to post progress update", exc_info=True)
            tool_results = []
            for tool_block in parsed.tool_use_blocks:
                result_text = await cog._execute_tool(tool_block.name, tool_block.input, user_id)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result_text,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
            cog.logger.info("tool_use handled, re-sending (iteration %d)", iteration + 1)
            continue

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason not in (
            "max_tokens",
            "refusal",
            "model_context_window_exceeded",
        ):
            cog.logger.warning("Unknown stop_reason: %s", response.stop_reason)
        return await finish(parsed)

    cog.logger.warning("Tool loop hit max_iterations (%d)", max_iterations)
    if parsed is None:
        raise RuntimeError("Tool loop completed without any API response")
    return await finish(parsed)


async def handle_new_message_in_conversation(cog, message, conversation: Conversation) -> None:
    """Handle a new Discord message in an ongoing Claude conversation."""
    params = conversation.params
    messages = conversation.messages

    cog.logger.info("Handling new message in conversation %s.", params.conversation_id)
    typing_task = None
    embeds = []

    try:
        if message.author != params.conversation_starter or params.paused:
            return

        validation_error = validate_request_configuration(params)
        if validation_error:
            await send_embed_batches(
                message.reply,
                embed=Embed(
                    title="Unsupported Tool Configuration",
                    description=validation_error,
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return

        typing_task = asyncio.create_task(keep_typing(cog, message.channel))

        user_content: list[dict[str, Any]] = []
        if message.content:
            user_content.append({"type": "text", "text": message.content})
        if message.attachments:
            for attachment in message.attachments:
                attachment_data = await fetch_attachment_bytes(cog, attachment)
                if attachment_data is not None:
                    content_block = build_attachment_content_block(
                        attachment.content_type or "",
                        attachment_data,
                        attachment.filename,
                    )
                    if content_block:
                        user_content.append(content_block)

        messages.append({"role": "user", "content": user_content})
        api_params = build_api_params(params, messages)
        parsed = await call_api_with_tool_loop(
            cog,
            api_params=api_params,
            messages=messages,
            user_id=message.author.id,
            progress_callback=(
                make_progress_poster(message.channel.send, cog.logger)
                if params.thinking_display == THINKING_DISPLAY_UPDATES
                else None
            ),
        )
        conversation.touch()
        response_text = parsed.text

        if typing_task:
            typing_task.cancel()
            typing_task = None

        append_thinking_embeds(embeds, parsed.thinking)
        append_response_embeds(embeds, response_text)
        append_stop_reason_embed(embeds, parsed.stop_reason, parsed.stop_details)
        append_fallback_embed(embeds, params.model, parsed.served_model)
        if parsed.context_compacted:
            append_compaction_embed(embeds)
        if parsed.context_warning:
            append_context_warning_embed(embeds)
        append_citations_embed(embeds, parsed.citations)

        request_cost, daily_cost = cog._track_daily_cost(
            message.author.id,
            params.model,
            parsed,
            advisor_model=params.advisor_model,
        )
        if SHOW_COST_EMBEDS:
            append_pricing_embed(embeds, parsed, request_cost, daily_cost)

        if conversation.params.conversation_id is None:
            cog.logger.error("Conversation ID is None, cannot track message")
            return

        await cog._strip_previous_view(message.author)
        view = cog.views.get(message.author)

        if embeds:
            reply_message = await send_embed_batches(
                message.reply,
                embeds=embeds,
                view=view,
                logger=cog.logger,
            )
            cog.last_view_messages[message.author] = reply_message
        else:
            reply_message = await message.reply(
                content="An error occurred: No content to send.",
                view=view,
            )
            cog.last_view_messages[message.author] = reply_message

    except Exception as error:
        if isinstance(error, aiohttp.ClientError):
            description = format_anthropic_error(error)
            cog.logger.error(
                "API error in handle_new_message_in_conversation: %s",
                description,
                exc_info=True,
            )
            if len(description) > 4000:
                description = description[:4000] + "\n\n... (error message truncated)"
            await cog._cleanup_conversation(message.author)
            await send_embed_batches(
                message.reply,
                embed=Embed(title="Error", description=description, color=Colour.red()),
                logger=cog.logger,
            )
            conv_key = (message.author.id, message.channel.id)
            cog.conversations.pop(conv_key, None)
            return

        description = format_anthropic_error(error)
        cog.logger.error(
            "Unexpected error in handle_new_message_in_conversation: %s",
            error,
            exc_info=True,
        )
        await cog._cleanup_conversation(message.author)
        conv_key = (message.author.id, message.channel.id)
        cog.conversations.pop(conv_key, None)
        with contextlib.suppress(Exception):
            await send_embed_batches(
                message.reply,
                embed=Embed(
                    title="Error",
                    description=f"An unexpected error occurred: {type(error).__name__}",
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
    finally:
        if typing_task:
            typing_task.cancel()


async def handle_on_message(cog, message) -> None:
    """Process Discord messages that belong to active Claude conversations."""
    if message.author == cog.bot.user:
        return

    cog.logger.debug(
        "Received message from %s in channel %s: %r",
        message.author,
        message.channel.id,
        message.content,
    )
    conv_key: ConversationKey = (message.author.id, message.channel.id)
    conversation = cog.conversations.get(conv_key)
    if conversation is not None:
        cog.logger.info(
            "Processing followup message for conversation %s",
            conversation.params.conversation_id,
        )
        await handle_new_message_in_conversation(cog, message, conversation)


async def run_effort_command(cog, ctx: ApplicationContext, *, effort: str) -> None:
    """Run the /claude effort command: change effort for the caller's active conversation."""
    if ctx.channel is None:
        await ctx.respond(
            embed=Embed(
                title="Error",
                description="This command must be used in a channel.",
                color=Colour.red(),
            )
        )
        return

    conversation = cog.conversations.get((ctx.author.id, ctx.channel.id))
    if conversation is None:
        await ctx.respond(
            embed=Embed(
                title="Error",
                description=(
                    "You have no active conversation in this channel. "
                    "Start one with `/claude chat` first."
                ),
                color=Colour.red(),
            )
        )
        return

    error = apply_effort_change(conversation, effort)
    if error:
        await ctx.respond(
            embed=Embed(title="Unsupported Effort", description=error, color=Colour.red())
        )
        return

    model = conversation.params.model
    if model in PER_MESSAGE_EFFORT_MODELS:
        note = (
            f"Effort set to `{effort}` for your next messages. `{model}` applies it "
            "per message (beta), so the prompt cache is preserved."
        )
    else:
        note = (
            f"Effort set to `{effort}` for your next messages. `{model}` changes the rendered "
            "prompt for a new effort level, so its cached prefix is invalidated."
        )
    cog.logger.info(
        "Effort changed to %s for conversation %s (%s)",
        effort,
        conversation.params.conversation_id,
        model,
    )
    await ctx.respond(embed=Embed(title="Effort Updated", description=note, color=Colour.green()))


async def run_chat_command(
    cog,
    ctx: ApplicationContext,
    *,
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
    thinking_display: str = THINKING_DISPLAY_SUMMARIZED,
    web_search: bool = True,
    web_fetch: bool = True,
    code_execution: bool = False,
    memory: bool = False,
    advisor: bool = False,
    mcp: str | None = None,
    tool_choice: str | None = None,
) -> None:
    """Run the /claude chat command."""
    await ctx.defer()
    typing_task = None

    if ctx.channel is None:
        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(
                title="Error",
                description="Cannot start conversation: channel context is unavailable.",
                color=Colour.red(),
            ),
            logger=cog.logger,
        )
        return

    if (ctx.author.id, ctx.channel.id) in cog.conversations:
        await send_embed_batches(
            ctx.send_followup,
            embed=Embed(
                title="Error",
                description="You already have an active conversation in this channel. Please finish it before starting a new one.",
                color=Colour.red(),
            ),
            logger=cog.logger,
        )
        return

    try:
        typing_task = asyncio.create_task(keep_typing(cog, ctx.channel))

        enabled_tools: list[str] = []
        if web_search:
            enabled_tools.append("web_search")
        if web_fetch:
            enabled_tools.append("web_fetch")
        if code_execution:
            enabled_tools.append("code_execution")
        if memory:
            enabled_tools.append("memory")
        advisor_model = get_default_advisor_model(model) if advisor else None
        if advisor and advisor_model is None:
            supported_models = ", ".join(sorted(ADVISOR_MODEL_COMPATIBILITY))
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Unsupported Tool Configuration",
                    description=(
                        f"Advisor is not supported for `{model}`. "
                        f"Supported executor models: {supported_models}."
                    ),
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return
        mcp_preset_names = parse_mcp_preset_names(mcp)
        _, mcp_error = resolve_mcp_presets(mcp_preset_names)
        if mcp_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Error",
                    description=mcp_error,
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return

        resolved_tool_choice: ToolChoice | None = None
        if tool_choice == "auto":
            resolved_tool_choice = {"type": "auto"}
        elif tool_choice == "none":
            resolved_tool_choice = {"type": "none"}
        if mcp_preset_names and resolved_tool_choice == {"type": "none"}:
            resolved_tool_choice = {"type": "auto"}

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if attachment:
            attachment_data = await fetch_attachment_bytes(cog, attachment)
            if attachment_data is not None:
                content_block = build_attachment_content_block(
                    attachment.content_type or "",
                    attachment_data,
                    attachment.filename,
                )
                if content_block:
                    user_content.append(content_block)

        conversation_messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        params = ChatCompletionParameters(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            effort=effort,
            thinking_budget=thinking_budget,
            thinking_display=thinking_display,
            conversation_starter=ctx.author,
            channel_id=ctx.channel.id,
            conversation_id=ctx.interaction.id,
            tools=enabled_tools,
            mcp_preset_names=mcp_preset_names,
            advisor_model=advisor_model,
            tool_choice=resolved_tool_choice,
        )

        validation_error = validate_request_configuration(params)
        if validation_error:
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Unsupported Tool Configuration",
                    description=validation_error,
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
            return

        api_params = build_api_params(params, conversation_messages)
        parsed = await call_api_with_tool_loop(
            cog,
            api_params=api_params,
            messages=conversation_messages,
            user_id=ctx.author.id,
            progress_callback=(
                make_progress_poster(ctx.send_followup, cog.logger)
                if thinking_display == THINKING_DISPLAY_UPDATES
                else None
            ),
        )
        response_text = parsed.text

        description = f"**Prompt:** {truncate_text(prompt, 2000)}\n"
        description += f"**Model:** {model}\n"
        if system:
            description += f"**System:** {truncate_text(system, 500)}\n"
        description += f"**Max Tokens:** {max_tokens}\n"
        if temperature is not None:
            description += f"**Temperature:** {temperature}\n"
        if top_p is not None:
            description += f"**Top P:** {top_p}\n"
        if top_k is not None:
            description += f"**Top K:** {top_k}\n"
        if effort is not None:
            description += f"**Effort:** {effort}\n"
        if thinking_budget is not None:
            description += f"**Thinking Budget:** {thinking_budget} tokens\n"
        if thinking_display != THINKING_DISPLAY_SUMMARIZED:
            description += f"**Thinking Display:** {thinking_display}\n"
        if enabled_tools:
            description += f"**Tools:** {', '.join(enabled_tools)}\n"
        if mcp_preset_names:
            description += f"**MCP Presets:** {', '.join(mcp_preset_names)}\n"
        if advisor_model is not None:
            description += f"**Advisor:** {advisor_model} (beta)\n"
        if resolved_tool_choice is not None:
            description += f"**Tool Choice:** {resolved_tool_choice['type']}\n"

        embeds = [
            Embed(
                title="Conversation Started",
                description=description,
                color=Colour.green(),
            )
        ]
        if mcp_preset_names:
            embeds.append(
                Embed(
                    title="MCP Enabled",
                    description=(
                        "Trusted MCP presets are active for this conversation. "
                        "Only enable servers you trust, because tool calls can share conversation data with them."
                    ),
                    color=Colour.orange(),
                )
            )
        append_thinking_embeds(embeds, parsed.thinking)
        append_response_embeds(embeds, response_text)
        append_stop_reason_embed(embeds, parsed.stop_reason, parsed.stop_details)
        append_fallback_embed(embeds, model, parsed.served_model)
        if parsed.context_compacted:
            append_compaction_embed(embeds)
        if parsed.context_warning:
            append_context_warning_embed(embeds)
        append_citations_embed(embeds, parsed.citations)

        request_cost, daily_cost = cog._track_daily_cost(
            ctx.author.id,
            model,
            parsed,
            advisor_model=advisor_model,
        )
        if SHOW_COST_EMBEDS:
            append_pricing_embed(embeds, parsed, request_cost, daily_cost)

        if len(embeds) == 1:
            await ctx.send_followup("No response generated.")
            return

        await cog._strip_previous_view(ctx.author)

        conv_key: ConversationKey = (ctx.author.id, ctx.channel.id)
        view = create_button_view(
            cog,
            user=ctx.author,
            conversation_key=conv_key,
            initial_tools=enabled_tools,
            initial_tool_choice=resolved_tool_choice,
        )

        message = await send_embed_batches(
            ctx.send_followup,
            embeds=embeds,
            view=view,
            logger=cog.logger,
        )
        cog.last_view_messages[ctx.author] = message

        conversation = Conversation(params=params, messages=conversation_messages)
        cog.conversations[conv_key] = conversation

    except Exception as error:
        description = format_anthropic_error(error)
        cog.logger.error("Unexpected error in chat: %s", error, exc_info=True)
        await cog._cleanup_conversation(ctx.author)
        with contextlib.suppress(Exception):
            await send_embed_batches(
                ctx.send_followup,
                embed=Embed(
                    title="Error",
                    description=description
                    if description
                    else f"An unexpected error occurred: {type(error).__name__}",
                    color=Colour.red(),
                ),
                logger=cog.logger,
            )
    finally:
        if typing_task:
            typing_task.cancel()


__all__ = [
    "build_api_params",
    "build_thinking_config",
    "call_api_with_tool_loop",
    "handle_check_permissions",
    "handle_new_message_in_conversation",
    "handle_on_message",
    "keep_typing",
    "run_chat_command",
    "validate_request_configuration",
]
