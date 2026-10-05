"""Plain-text reply mode: send Claude's answer as a normal Discord message.

Enabled with PLAIN_TEXT_REPLIES=true. Response embeds become message content,
the "Conversation Started" and "Thinking" embeds are dropped (the prompt is
quoted instead), and the cost line becomes small grey subtext. Any other
embeds (sources, warnings) stay attached to the last message.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from discord import AllowedMentions, Embed

TEXT_CHUNK_SIZE = 1900
PROMPT_QUOTE_LIMIT = 300
DROPPED_TITLES = frozenset({"Thinking"})
_RESPONSE_TITLE = re.compile(r"^Response( \(Part \d+\))?$")
_PROMPT = re.compile(r"\*\*Prompt:\*\* (.*?)\n\*\*Model:\*\*", re.DOTALL)
_FENCE = re.compile(r"^\s*```")


def plain_text_replies_enabled() -> bool:
    return os.getenv("PLAIN_TEXT_REPLIES", "false").strip().lower() in {"true", "1", "yes"}


def _quote(prompt: str) -> str:
    prompt = prompt.strip()
    if len(prompt) > PROMPT_QUOTE_LIMIT:
        prompt = prompt[: PROMPT_QUOTE_LIMIT - 1].rstrip() + "…"
    return "\n".join(f"> {line}" if line.strip() else ">" for line in prompt.splitlines())


def split_embeds(embeds: list[Embed]) -> tuple[str, list[Embed]] | None:
    """Return (message text, embeds to keep), or None if there is no Response embed."""
    head: list[str] = []
    body: list[str] = []
    foot: list[str] = []
    keep: list[Embed] = []

    for embed in embeds:
        data = embed.to_dict()
        title = data.get("title") or ""
        description = data.get("description") or ""
        if title in DROPPED_TITLES:
            continue
        if title == "Conversation Started":
            match = _PROMPT.search(description)
            if match:
                head.append(_quote(match.group(1)))
            continue
        if _RESPONSE_TITLE.match(title):
            body.append(description)
            continue
        if not title and not data.get("fields") and description.startswith("$"):
            foot.append(f"-# {description}")
            continue
        keep.append(embed)

    if not body:
        return None
    text = "\n\n".join(
        part for part in ("\n".join(head), "".join(body).strip(), "\n".join(foot)) if part
    )
    return text, keep


def chunk_markdown(text: str, limit: int = TEXT_CHUNK_SIZE) -> list[str]:
    """Split on line boundaries, closing and reopening code fences across chunks."""
    chunks: list[str] = []
    current = ""
    fence: str | None = None

    def lines_of(raw: str) -> list[str]:
        out: list[str] = []
        for line in raw.splitlines(keepends=True):
            while len(line) > limit:
                out.append(line[:limit])
                line = line[limit:]
            out.append(line)
        return out

    for line in lines_of(text):
        if len(current) + len(line) + 4 > limit and current.strip():
            if fence is not None:
                current += "\n```" if not current.endswith("\n") else "```"
            chunks.append(current.rstrip())
            current = f"{fence}\n" if fence is not None else ""
        if _FENCE.match(line):
            fence = None if fence is not None else line.strip()
        current += line
    if current.strip():
        chunks.append(current.rstrip())
    return chunks or ["(empty response)"]


async def send_plain(
    send: Callable[..., Awaitable[Any]],
    text: str,
    keep: list[Embed],
    *,
    view: Any = None,
    **kwargs: Any,
) -> Any:
    chunks = chunk_markdown(text)
    final_message = None
    for index, chunk in enumerate(chunks):
        send_kwargs = dict(kwargs)
        send_kwargs["content"] = chunk
        send_kwargs["allowed_mentions"] = AllowedMentions.none()
        if index == len(chunks) - 1:
            if keep:
                send_kwargs["embeds"] = keep[:10]
            if view is not None:
                send_kwargs["view"] = view
        final_message = await send(**send_kwargs)
    return final_message
