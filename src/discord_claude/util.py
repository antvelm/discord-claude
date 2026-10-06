from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypedDict

from discord import Embed, Member, User

from discord_claude.config.pricing import (
    CACHE_READ_PRICING,
    MODEL_CONTEXT_WINDOWS,
    MODEL_PRICING,
    UNKNOWN_MODEL_PRICING,
    WEB_SEARCH_COST_PER_REQUEST,
)

CHUNK_TEXT_SIZE = 3500  # Maximum number of characters in each text chunk.

CACHE_TTL = "1h"  # 1-hour TTL for prompt caching (2x base input price for writes)

ADVISOR_BETA = "advisor-tool-2026-03-01"
ADVISOR_TOOL_TYPE = "advisor_20260301"
ADVISOR_TOOL_NAME = "advisor"
ADVISOR_MAX_USES = 3
# Executor model -> advisor models the API accepts for it: the advisor-tool
# compatibility table plus claude-opus-5-5, which the table does not list but the
# API accepts as an advisor for every executor here except Fable 5.1, and as an
# executor with an Opus 5.5 / Opus 5 / Fable 5 / Fable 5.1 advisor (it rejects
# claude-opus-4-8).
# get_default_advisor_model takes the FIRST entry and the advisor slash option is
# a bare toggle, so tuple order is the selection surface: claude-opus-4-8 leads
# every tuple that allows it because it returns plaintext advice, whereas
# claude-opus-5 / claude-fable-5 / claude-fable-5-1 advisors return an encrypted
# advisor_redacted_result (harmless here: the bot skips advisor_tool_result blocks
# by their outer type and replays them verbatim). An Opus 5.5 executor defaults to
# claude-opus-5-5, the cheapest advisor it accepts. claude-mythos-5 /
# claude-mythos-5-1 are in the table but not publicly callable, so they are
# omitted; claude-sonnet-4-5 and claude-opus-4-5 are not executors.
ADVISOR_MODEL_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "claude-haiku-4-5": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ),
    "claude-sonnet-4-6": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ),
    "claude-sonnet-5": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
    ),
    "claude-opus-4-6": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
    ),
    "claude-opus-4-7": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
    ),
    "claude-opus-4-8": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
    ),
    "claude-opus-5-5": ("claude-opus-5-5", "claude-opus-5", "claude-fable-5", "claude-fable-5-1"),
    "claude-opus-5": ("claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"),
    "claude-fable-5": ("claude-opus-5", "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1"),
    # Fable 5.1 executors pair only with Fable 5.1 (and the non-public Mythos 5.1);
    # the API rejects an Opus 5.5 advisor here.
    "claude-fable-5-1": ("claude-fable-5-1",),
}

# Models that support adaptive thinking
ADAPTIVE_THINKING_MODELS = {
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}

# Models that reject explicit sampling parameter overrides.
# claude-opus-5 and claude-sonnet-5 join the effort-parameter generation
# (Fable 5 / 5.1, Opus 4.8/4.7). The anthropic 1.x SDK no longer exposes
# temperature/top_p/top_k as typed request parameters, so build_api_params
# sends them through extra_body; this set gates that path so these models
# never receive the values and return a 400 from the API.
SAMPLING_LOCKED_MODELS = {
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
}

# Models that 400 on forced tool use — `tool_choice` type "any" or "tool"
# ("tool_choice: type \"tool\" and \"any\" are not supported for this model"):
# thinking is always on for them and a forced call
# would skip it. Only "auto" and "none" are accepted.
FORCED_TOOL_CHOICE_UNSUPPORTED_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5"})

# Models without programmatic tool calling. web_search_20260209 and
# web_fetch_20260309 default `allowed_callers` to the code-execution caller, and a
# model that lacks programmatic tool calling rejects that default with a 400
# ("does not support programmatic tool calling ... Explicitly set
# allowed_callers=["direct"]"). build_api_params sets allowed_callers=["direct"]
# on the web tools for these models only.
PROGRAMMATIC_TOOL_CALLING_UNSUPPORTED_MODELS = frozenset({"claude-haiku-4-5"})

# thinking.display values. "summarized" (the bot's default) returns summarized
# reasoning in thinking blocks. "updates" (beta header THINKING_DISPLAY_UPDATES_BETA)
# returns reasoning empty, as under "omitted", while the short progress updates
# Fable 5 / Fable 5.1 / Opus 5.5 write between tool calls come back readable — at
# most one thinking block before a tool call — so each can be shown as a status
# line while the tools run.
# Other adaptive models accept the value but write no updates (Opus 5 returns an
# empty thinking block without an error), so the option is gated to the models
# that produce them.
THINKING_DISPLAY_SUMMARIZED = "summarized"
THINKING_DISPLAY_UPDATES = "updates"
THINKING_DISPLAY_UPDATES_BETA = "thinking-display-updates-2026-08-18"
THINKING_DISPLAY_UPDATES_MODELS = frozenset(
    {"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5"}
)

# Per-message effort (beta header PER_MESSAGE_EFFORT_BETA): a `role: "system"`
# message carrying `output_config.effort` changes effort from the next user turn
# while the top-level value, and so the cached prompt prefix, is unchanged. The
# header is sent on EVERY request for these models (see call_api_with_tool_loop):
# adding it only when an override appears re-renders the prompt and rewrites the
# cache on that turn, which defeats the purpose.
# Opus 5.5, Opus 5 and Fable 5.1 accept it (two consecutive override messages
# too); Fable 5 400s ("output_config.effort requires a model
# that supports per-turn effort; this model does not"); without the header every
# model 400s ("messages.N.output_config: Extra inputs are not permitted").
PER_MESSAGE_EFFORT_BETA = "mid-conversation-output-config-2026-07-01"
PER_MESSAGE_EFFORT_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5"})

# output_config.effort ladder in ascending order. Each model accepts only a
# prefix of it (plus/minus "xhigh"), gated by supported_effort_levels below;
# sending anything else returns a 400.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Models that accept the effort parameter at all. claude-sonnet-4-5 and
# claude-haiku-4-5 reject it outright ("This model does not support the
# effort parameter"), so they stay out of this set.
EFFORT_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
    }
)

# Models that accept effort "xhigh" (introduced with Opus 4.7). Every one of
# these takes the full low..max ladder.
XHIGH_EFFORT_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
    }
)

# Models that accept effort "max". Opus 4.6 / Sonnet 4.6 predate "xhigh" but do
# take "max"; claude-opus-4-5 is capped at "high".
MAX_EFFORT_MODELS = XHIGH_EFFORT_MODELS | {"claude-opus-4-6", "claude-sonnet-4-6"}


def supported_effort_levels(model: str) -> frozenset[str]:
    """Return the output_config.effort values the API accepts for a model.

    Empty for models that reject the parameter entirely.
    """
    if model not in EFFORT_MODELS:
        return frozenset()
    levels = {"low", "medium", "high"}
    if model in XHIGH_EFFORT_MODELS:
        levels.add("xhigh")
    if model in MAX_EFFORT_MODELS:
        levels.add("max")
    return frozenset(levels)


# Models that only support adaptive thinking (no budget_tokens mode).
# claude-fable-5, claude-fable-5-1 and claude-opus-5-5 additionally reject an
# explicit {"type": "disabled"} config (thinking is always on for them);
# build_thinking_config never emits one (it omits the param instead), so the
# existing adaptive path is safe for them. claude-opus-5 and claude-sonnet-5 are
# adaptive-only too (manual extended thinking with budget_tokens returns a 400)
# but do accept {"type": "disabled"}.
ADAPTIVE_ONLY_THINKING_MODELS = {
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
}

# Models that support server-side compaction (beta compact-2026-01-12 with the
# compact_20260112 edit). Every other selectable model (Opus 4.5, Sonnet 4.5,
# Haiku 4.5) takes the manual path bounded by manual_compaction_trigger.
COMPACTION_MODELS = {
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
}

# Server-side refusal fallback (beta). These models' safety classifiers can
# decline a request (HTTP 200 with stop_reason "refusal") — Anthropic's
# refusals page names Claude Fable 5 and Claude Opus 5, and the Fable 5.1
# launch notes add Claude Fable 5.1 (its permitted fallback targets are Opus 4.8
# and Opus 5). Claude Opus 5.5 also has the
# classifiers and permits the same two targets. The target Opus 4.8 has no
# classifier, which is what makes it a fallback. With the beta active the API
# retries the same request on REFUSAL_FALLBACK_MODEL in one round trip. The
# explicit-list form used here accepts up to three named fallback models; a
# fallbacks="default" routing mode also exists under the
# server-side-fallback-2026-07-01 header but has not been adopted.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-06-01"
REFUSAL_FALLBACK_MODELS = {"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5"}
REFUSAL_FALLBACK_MODEL = "claude-opus-4-8"

# Context management thresholds
CONTEXT_WARNING_THRESHOLD = 0.85  # Show warning embed at 85% of context window
CONTEXT_COMPACTION_THRESHOLD = 0.75  # Trigger manual compaction at 75% (non-compaction models)
COMPACTION_SUMMARY_MODEL = "claude-haiku-4-5"  # Cheap model for generating summaries


def manual_compaction_trigger(context_window: int) -> float:
    """Token count at which manual compaction must fire for a given chat model.

    Manual compaction hands the *entire* message list to
    COMPACTION_SUMMARY_MODEL, so the trigger has to stay inside that
    summarizer's window rather than the chat model's. Today's manual-path
    models (Opus 4.5, Sonnet 4.5, Haiku 4.5) all share the summarizer's 200k
    window, so the bound is currently a no-op; it stays because a 1M-window
    model on this path (claude-opus-5 was, before it joined COMPACTION_MODELS)
    would otherwise fire at 750k and hand a 750k-token payload to a 200k-token
    summarizer and 400.
    """
    summary_window = MODEL_CONTEXT_WINDOWS.get(COMPACTION_SUMMARY_MODEL, 200_000)
    return min(context_window, summary_window) * CONTEXT_COMPACTION_THRESHOLD


# Models that support manual extended thinking (type: "enabled" with budget_tokens)
EXTENDED_THINKING_MODELS = {
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
}

# Discord embed limits
DISCORD_EMBED_TOTAL_LIMIT = 6000  # Max chars across all embeds in a single message
CITATION_EMBED_RESERVE = 500  # Chars reserved for a potential citations embed


def available_embed_space(embeds: list[Embed], reserve: int = 0) -> int:
    """Calculate remaining character budget across all embeds in a message."""
    used = sum(len(e.description or "") + len(e.title or "") for e in embeds)
    return DISCORD_EMBED_TOTAL_LIMIT - used - reserve


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    web_search_requests: int = 0,
) -> float:
    """Calculate the cost in dollars for a given model and token usage.

    Cache write tokens cost 2x base input price (1h TTL); cache read tokens cost 0.1x
    unless pricing.yaml declares a `cache_read_per_million` for the model (Fable 5.1
    reads at 0.025x, Opus 5.5 at 0.05x). Web search requests cost $0.01 each ($10 per
    1,000 searches).
    """
    input_price, output_price = MODEL_PRICING.get(model, UNKNOWN_MODEL_PRICING)
    cache_read_price = CACHE_READ_PRICING.get(model, input_price * 0.10)
    return (
        (input_tokens / 1_000_000) * input_price
        + (output_tokens / 1_000_000) * output_price
        + (cache_creation_tokens / 1_000_000) * input_price * 2.0
        + (cache_read_tokens / 1_000_000) * cache_read_price
        + web_search_requests * WEB_SEARCH_COST_PER_REQUEST
    )


def priced_model(model: str | None, request_model: str) -> str:
    """Return the pricing-row id for usage reported under `model`.

    None stands for the request model. An id without a pricing row (such as a dated
    snapshot id) also resolves to the request model rather than to
    UNKNOWN_MODEL_PRICING.
    """
    if model is not None and model in MODEL_PRICING:
        return model
    return request_model


def get_default_advisor_model(executor_model: str) -> str | None:
    """Return the default compatible advisor model for an executor model."""
    compatible_models = ADVISOR_MODEL_COMPATIBILITY.get(executor_model)
    if not compatible_models:
        return None
    return compatible_models[0]


@dataclass
class ModelTokenUsage:
    """Token counts billed at one model's rates."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


def _usage_model(usage: Any) -> str | None:
    """The model a usage.iterations entry names, or None when it names none."""
    model = getattr(usage, "model", None)
    return model if isinstance(model, str) else None


def _prompt_tokens(usage: Any) -> int:
    """Full prompt size of one response or iteration: uncached, cache-read and
    cache-write input tokens."""
    return sum(
        getattr(usage, name, 0) or 0
        for name in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )


@dataclass
class UsageTotals:
    """Accumulates token/tool usage across multiple API iterations."""

    input_tokens: int = 0
    output_tokens: int = 0
    # Subset of output_tokens spent on extended/adaptive thinking (billed at the
    # output rate). Anthropic 0.105+ reports this via usage.output_tokens_details.
    thinking_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0
    code_execution_requests: int = 0
    advisor_calls: int = 0
    advisor_input_tokens: int = 0
    advisor_output_tokens: int = 0
    advisor_cache_creation_tokens: int = 0
    advisor_cache_read_tokens: int = 0
    context_compacted: bool = False
    # The token counts above, grouped by the model whose rates apply. The None key
    # is the request model: top-level usage, compaction entries and entries that
    # name no model.
    tokens_by_model: dict[str | None, ModelTokenUsage] = field(default_factory=dict)
    # Full prompt size of the most recent response (uncached + cache-read +
    # cache-write input tokens): the measure for the context warning and the manual
    # compaction trigger.
    prompt_tokens: int = 0

    def _accumulate_executor_usage(self, usage: Any, model: str | None = None) -> None:
        """Add usage billed at `model`'s rates (None: the request model's)."""
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_creation_tokens += cache_creation_tokens
        self.cache_read_tokens += cache_read_tokens
        bucket = self.tokens_by_model.setdefault(model, ModelTokenUsage())
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.cache_creation_tokens += cache_creation_tokens
        bucket.cache_read_tokens += cache_read_tokens

    def _accumulate_thinking_tokens(self, usage: Any) -> None:
        """Add the thinking subset of a response's output tokens.

        Only the top-level usage object carries output_tokens_details; the
        usage.iterations entries do not, so this reads the top-level object whether
        or not iterations are present.
        """
        details = getattr(usage, "output_tokens_details", None)
        if details is not None:
            thinking = getattr(details, "thinking_tokens", 0) or 0
            if isinstance(thinking, int):
                self.thinking_tokens += thinking

    def _accumulate_advisor_usage(self, usage: Any) -> None:
        """Add usage billed at the advisor model's rates."""
        self.advisor_calls += 1
        self.advisor_input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.advisor_output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.advisor_cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.advisor_cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def accumulate(self, usage: Any) -> None:
        """Add a single API response's usage to running totals."""
        if usage is None:
            return

        iterations = getattr(usage, "iterations", None)
        if isinstance(iterations, (list, tuple)):
            for iteration in iterations:
                iteration_type = getattr(iteration, "type", None)
                if iteration_type == "advisor_message":
                    self._accumulate_advisor_usage(iteration)
                elif iteration_type in ("message", "fallback_message"):
                    # Each attempt bills at the rates of the model named on its
                    # entry: in a refusal-fallback turn the declined model's
                    # "message" entry and the fallback model's "fallback_message"
                    # entry name different models.
                    self._accumulate_executor_usage(iteration, _usage_model(iteration))
                    # The last sampling entry gives the context size; a compaction
                    # entry reports only the summarisation call.
                    self.prompt_tokens = _prompt_tokens(iteration)
                elif iteration_type == "compaction":
                    # Server-side compaction (compact_20260112): the summarisation
                    # call bills at the request model's rates and is excluded from
                    # the top-level input/output counts, so it is added here.
                    self._accumulate_executor_usage(iteration)
                    self.context_compacted = True
        else:
            self._accumulate_executor_usage(usage)
            self.prompt_tokens = _prompt_tokens(usage)
        self._accumulate_thinking_tokens(usage)

        server_tool_use = getattr(usage, "server_tool_use", None)
        if server_tool_use:
            self.web_search_requests += getattr(server_tool_use, "web_search_requests", 0) or 0
            self.web_fetch_requests += getattr(server_tool_use, "web_fetch_requests", 0) or 0
            self.code_execution_requests += (
                getattr(server_tool_use, "code_execution_requests", 0) or 0
            )

    def accumulate_compaction_summary(self, usage: Any) -> None:
        """Add a manual compaction's summarizer call, billed at COMPACTION_SUMMARY_MODEL's
        rates. The call does not continue the conversation, so prompt_tokens is kept."""
        if usage is not None:
            self._accumulate_executor_usage(usage, COMPACTION_SUMMARY_MODEL)
            self._accumulate_thinking_tokens(usage)

    def apply_to(self, parsed: Any, context_window: int) -> None:
        """Stamp all accumulated totals onto a ParsedResponse."""
        parsed.input_tokens = self.input_tokens
        parsed.output_tokens = self.output_tokens
        parsed.thinking_tokens = self.thinking_tokens
        parsed.cache_creation_tokens = self.cache_creation_tokens
        parsed.cache_read_tokens = self.cache_read_tokens
        parsed.web_search_requests = self.web_search_requests
        parsed.web_fetch_requests = self.web_fetch_requests
        parsed.code_execution_requests = self.code_execution_requests
        parsed.advisor_calls = self.advisor_calls
        parsed.advisor_input_tokens = self.advisor_input_tokens
        parsed.advisor_output_tokens = self.advisor_output_tokens
        parsed.advisor_cache_creation_tokens = self.advisor_cache_creation_tokens
        parsed.advisor_cache_read_tokens = self.advisor_cache_read_tokens
        parsed.tokens_by_model = {
            model: replace(tokens) for model, tokens in self.tokens_by_model.items()
        }
        parsed.context_compacted = self.context_compacted
        parsed.context_warning = self.prompt_tokens > context_window * CONTEXT_WARNING_THRESHOLD


class ToolChoiceAuto(TypedDict):
    """Allow Claude to decide whether to use available tools."""

    type: Literal["auto"]


class ToolChoiceAny(TypedDict):
    """Require Claude to use one of the available tools."""

    type: Literal["any"]


class ToolChoiceNone(TypedDict):
    """Disable tool invocation while still allowing tool definitions."""

    type: Literal["none"]


class ToolChoiceTool(TypedDict):
    """Force Claude to call a specific available tool."""

    type: Literal["tool"]
    name: str


ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceNone | ToolChoiceTool


@dataclass
class ChatCompletionParameters:
    """A dataclass to store the parameters for a chat completion."""

    model: str
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int = 16384
    conversation_starter: Member | User | None = None
    conversation_id: int | None = None
    channel_id: int | None = None
    effort: str | None = None
    thinking_budget: int | None = None
    thinking_display: str = THINKING_DISPLAY_SUMMARIZED
    paused: bool | None = False
    tools: list[str] = field(default_factory=list)
    mcp_preset_names: list[str] = field(default_factory=list)
    advisor_model: str | None = None
    tool_choice: ToolChoice | None = None


# Conversation key: (user_id, channel_id) for O(1) lookup
ConversationKey = tuple[int, int]


@dataclass
class Conversation:
    """A dataclass to store conversation state."""

    params: ChatCompletionParameters
    messages: list[dict[str, Any]]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Discord messages of the bot's recent replies (reply_edits.SentMessage), oldest first.
    replies: list[list[Any]] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


class ToolHandler(Protocol):
    """Protocol for client-side tool handlers."""

    async def execute(self, tool_input: dict[str, Any], user_id: int) -> str: ...


def chunk_text(text: str, chunk_size: int = CHUNK_TEXT_SIZE) -> list[str]:
    """
    Splits a string into chunks of a specified size.

    Args:
        text: The string to split.
        chunk_size: The maximum size of each chunk.

    Returns:
        A list of strings, where each string is a chunk of the original text.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def truncate_text(text: str | None, max_length: int, suffix: str = "...") -> str | None:
    """
    Truncate text to max_length, adding suffix if truncated.

    Args:
        text: The text to truncate
        max_length: Maximum length before truncation
        suffix: String to append when truncated (default "...")

    Returns:
        Original text if under max_length, otherwise truncated with suffix
    """
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def format_anthropic_error(error: Exception) -> str:
    """Return a readable description for exceptions raised by Anthropic operations."""
    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = str(error).strip()

    status = getattr(error, "status_code", None)
    api_error_type = getattr(error, "error_type", None)
    error_type = api_error_type or type(error).__name__

    details = []
    if status is not None:
        details.append(f"Status: {status}")
    if error_type and error_type != "Exception":
        details.append(f"Error: {error_type}")

    if details:
        return f"{message}\n\n" + "\n".join(details)
    return message
