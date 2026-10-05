from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from discord import Embed, HTTPException

from .plain_delivery import plain_text_replies_enabled, send_plain, split_embeds

DISCORD_EMBED_TOTAL_LIMIT = 6000
DISCORD_EMBEDS_PER_MESSAGE_LIMIT = 10
DISCORD_MESSAGE_CONTENT_LIMIT = 2000
PLAIN_TEXT_CHUNK_SIZE = 1900
ATTACHMENT_URL_PREFIX = "attachment://"


def count_embed_chars(embed: Embed) -> int:
    """Count Discord's visible text-bearing embed fields."""

    data = embed.to_dict()
    total = _text_len(data.get("title")) + _text_len(data.get("description"))
    for field in data.get("fields") or []:
        total += _text_len(field.get("name")) + _text_len(field.get("value"))
    footer = data.get("footer") or {}
    total += _text_len(footer.get("text"))
    author = data.get("author") or {}
    total += _text_len(author.get("name"))
    return total


def pack_embeds(embeds: Iterable[Embed]) -> list[list[Embed]]:
    """Pack embeds into Discord-safe message batches while preserving order."""

    batches: list[list[Embed]] = []
    current: list[Embed] = []
    current_chars = 0

    for embed in embeds:
        embed_chars = count_embed_chars(embed)
        if current and (
            len(current) >= DISCORD_EMBEDS_PER_MESSAGE_LIMIT
            or current_chars + embed_chars > DISCORD_EMBED_TOTAL_LIMIT
        ):
            batches.append(current)
            current = []
            current_chars = 0

        current.append(embed)
        current_chars += embed_chars

    if current:
        batches.append(current)
    return batches


async def send_embed_batches(
    send: Callable[..., Awaitable[Any]],
    *,
    embed: Embed | None = None,
    embeds: Iterable[Embed] | None = None,
    view: Any = None,
    file: Any = None,
    files: Iterable[Any] | None = None,
    logger: Any = None,
    **kwargs: Any,
) -> Any:
    """Send embeds in Discord-safe batches and return the final sent message."""

    normalized_embeds = _normalize_embeds(embed=embed, embeds=embeds)
    normalized_files = _normalize_files(file=file, files=files)
    single_file_input = file is not None and files is None

    if normalized_embeds and not normalized_files and plain_text_replies_enabled():
        plain = split_embeds(normalized_embeds)
        if plain is not None:
            text, keep = plain
            return await send_plain(send, text, keep, view=view, **kwargs)

    if not normalized_embeds:
        send_kwargs = dict(kwargs)
        if view is not None:
            send_kwargs["view"] = view
        _add_files_to_kwargs(send_kwargs, normalized_files, single_file_input=single_file_input)
        return await send(**send_kwargs)

    batches = pack_embeds(normalized_embeds)
    batch_refs = [_attachment_filenames_for_embeds(batch) for batch in batches]
    all_refs = set().union(*batch_refs) if batch_refs else set()
    unreferenced_files = [
        item
        for item in normalized_files
        if _file_name(item) is None or _file_name(item) not in all_refs
    ]
    sent_file_ids: set[int] = set()
    final_message = None

    for index, batch in enumerate(batches):
        is_first = index == 0
        is_last = index == len(batches) - 1
        refs = batch_refs[index]
        batch_files = [
            item
            for item in normalized_files
            if id(item) not in sent_file_ids
            and (_file_name(item) in refs or (is_first and item in unreferenced_files))
        ]
        sent_file_ids.update(id(item) for item in batch_files)

        send_kwargs = dict(kwargs)
        if len(batch) == 1 and embed is not None and embeds is None:
            send_kwargs["embed"] = batch[0]
        else:
            send_kwargs["embeds"] = batch
        if is_last and view is not None:
            send_kwargs["view"] = view
        _add_files_to_kwargs(send_kwargs, batch_files, single_file_input=single_file_input)

        try:
            final_message = await send(**send_kwargs)
        except HTTPException as error:
            _log_embed_send_failure(logger, error, batch, batches)
            final_message = await _send_plain_text_fallback(
                send,
                batch,
                batch_files=batch_files,
                view=view if is_last else None,
                single_file_input=single_file_input,
                **kwargs,
            )

    return final_message


def _normalize_embeds(*, embed: Embed | None, embeds: Iterable[Embed] | None) -> list[Embed]:
    normalized: list[Embed] = []
    if embed is not None:
        normalized.append(embed)
    if embeds is not None:
        normalized.extend(list(embeds))
    return normalized


def _normalize_files(*, file: Any, files: Iterable[Any] | None) -> list[Any]:
    normalized: list[Any] = []
    if file is not None:
        normalized.append(file)
    if files is not None:
        normalized.extend(list(files))
    return normalized


def _add_files_to_kwargs(
    send_kwargs: dict[str, Any],
    files: list[Any],
    *,
    single_file_input: bool,
) -> None:
    if not files:
        return
    if single_file_input and len(files) == 1:
        send_kwargs["file"] = files[0]
    else:
        send_kwargs["files"] = files


def _attachment_filenames_for_embeds(embeds: Iterable[Embed]) -> set[str]:
    names: set[str] = set()
    for embed in embeds:
        _collect_attachment_filenames(embed.to_dict(), names)
    return names


def _collect_attachment_filenames(value: Any, names: set[str]) -> None:
    if isinstance(value, str):
        if value.startswith(ATTACHMENT_URL_PREFIX):
            names.add(value.removeprefix(ATTACHMENT_URL_PREFIX))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_attachment_filenames(item, names)
        return
    if isinstance(value, list):
        for item in value:
            _collect_attachment_filenames(item, names)


def _send_text_for_embed(embed: Embed) -> str:
    data = embed.to_dict()
    parts: list[str] = []
    title = data.get("title")
    if title:
        parts.append(f"**{title}**")
    description = data.get("description")
    if description:
        parts.append(str(description))
    for field in data.get("fields") or []:
        name = field.get("name")
        value = field.get("value")
        if name and value:
            parts.append(f"**{name}**\n{value}")
        elif name:
            parts.append(str(name))
        elif value:
            parts.append(str(value))
    footer = data.get("footer") or {}
    if footer.get("text"):
        parts.append(str(footer["text"]))
    author = data.get("author") or {}
    author_name = author.get("name")
    if author_name:
        parts.append(str(author_name))
    return "\n\n".join(parts)


async def _send_plain_text_fallback(
    send: Callable[..., Awaitable[Any]],
    batch: list[Embed],
    *,
    batch_files: list[Any],
    view: Any,
    single_file_input: bool,
    **kwargs: Any,
) -> Any:
    fallback_text = "\n\n".join(_send_text_for_embed(embed) for embed in batch).strip()
    if not fallback_text:
        fallback_text = "No embed text content available."
    chunks = _chunk_plain_text(fallback_text)
    final_message = None
    for index, chunk in enumerate(chunks):
        send_kwargs = dict(kwargs)
        send_kwargs["content"] = chunk
        if index == 0:
            _add_files_to_kwargs(
                send_kwargs,
                batch_files,
                single_file_input=single_file_input,
            )
        if index == len(chunks) - 1 and view is not None:
            send_kwargs["view"] = view
        final_message = await send(**send_kwargs)
    return final_message


def _chunk_plain_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + PLAIN_TEXT_CHUNK_SIZE, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk[:DISCORD_MESSAGE_CONTENT_LIMIT])
        start = end
        while start < len(text) and text[start] == "\n":
            start += 1
    return chunks or ["No embed text content available."]


def _log_embed_send_failure(
    logger: Any,
    error: HTTPException,
    batch: list[Embed],
    batches: list[list[Embed]],
) -> None:
    if logger is None:
        return
    batch_sizes = [sum(count_embed_chars(embed) for embed in item) for item in batches]
    logger.warning(
        "Discord rejected embed batch; falling back to plain text: %s "
        "(failed_batch_embeds=%s failed_batch_chars=%s all_batch_chars=%s)",
        error,
        len(batch),
        sum(count_embed_chars(embed) for embed in batch),
        batch_sizes,
    )


def _file_name(file: Any) -> str | None:
    filename = getattr(file, "filename", None)
    if filename is None:
        filename = getattr(file, "name", None)
    return str(filename) if filename else None


def _text_len(value: Any) -> int:
    return len(str(value)) if value is not None else 0


__all__ = [
    "DISCORD_EMBEDS_PER_MESSAGE_LIMIT",
    "DISCORD_EMBED_TOTAL_LIMIT",
    "count_embed_chars",
    "pack_embeds",
    "send_embed_batches",
]
