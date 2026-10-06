"""Let Claude edit its own earlier replies in place (the `edit_reply` client tool).

Every Discord message a reply is sent as is recorded with its text, so an edit is
matched against that local copy and costs a single Discord API call. Edits go
through ``channel.get_partial_message(id).edit``, which uses the bot token: the
interaction token behind a ``/claude chat`` followup expires after 15 minutes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from discord import AllowedMentions, Embed, HTTPException, NotFound

# Discord's limits for message content and an embed description.
CONTENT_LIMIT = 2000
EMBED_DESCRIPTION_LIMIT = 4096
# Earlier replies kept per conversation; older ones can no longer be edited.
MAX_TRACKED_REPLIES = 10

EDIT_REPLY_TOOL: dict[str, Any] = {
    "name": "edit_reply",
    "description": (
        "Edit one of your own earlier replies in this Discord conversation in place, by "
        "replacing an exact snippet of its text. Use it when the user asks you to fix, "
        "correct, shorten or update something you already said, instead of posting a "
        "corrected copy. Your replies are searched newest first, and old_text must appear "
        "exactly once in the newest reply that contains it. A long reply is split across "
        "several Discord messages of at most 2000 characters, and one edit changes one "
        "message, so keep old_text short and inside one paragraph; call the tool again "
        "for further changes. After editing, answer the user with a short confirmation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "old_text": {
                "type": "string",
                "description": "Exact text to replace, copied from your earlier reply.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text; an empty string deletes old_text.",
            },
        },
        "required": ["old_text", "new_text"],
    },
}


@dataclass
class SentMessage:
    """Local copy of one Discord message a reply was sent as."""

    id: int
    content: str
    embeds: list[dict[str, Any]] = field(default_factory=list)


def recording_send(
    send: Callable[..., Awaitable[Any]], sink: list[SentMessage]
) -> Callable[..., Awaitable[Any]]:
    """Wrap a Discord send callable so every message it creates is recorded in ``sink``."""

    async def _send(*args: Any, **kwargs: Any) -> Any:
        message = await send(*args, **kwargs)
        message_id = getattr(message, "id", None)
        if isinstance(message_id, int):
            content = getattr(message, "content", None)
            sink.append(
                SentMessage(
                    id=message_id,
                    content=content if isinstance(content, str) else "",
                    embeds=[embed.to_dict() for embed in getattr(message, "embeds", None) or []],
                )
            )
        return message

    return _send


def remember_reply(replies: list[list[SentMessage]], sent: list[SentMessage]) -> None:
    """Add a sent reply to a conversation's history, keeping the newest MAX_TRACKED_REPLIES."""
    if not sent:
        return
    replies.append(sent)
    del replies[:-MAX_TRACKED_REPLIES]


class EditReplyHandler:
    """Executes `edit_reply` against the replies of one conversation."""

    def __init__(self, channel: Any, replies: list[list[SentMessage]]) -> None:
        self.channel = channel
        self.replies = replies

    async def execute(self, tool_input: dict[str, Any], user_id: int) -> str:
        old_text = tool_input.get("old_text")
        new_text = tool_input.get("new_text")
        if not isinstance(old_text, str) or not old_text:
            return "Error: old_text must be a non-empty string."
        if not isinstance(new_text, str):
            return "Error: new_text must be a string."
        if not self.replies:
            return "Error: there is no earlier reply in this conversation to edit."

        for reply in reversed(self.replies):
            matches = [
                (message, location)
                for message in reply
                for location, text in _texts(message)
                if old_text in text
            ]
            count = sum(text.count(old_text) for message in reply for _, text in _texts(message))
            if count == 0:
                continue
            if count > 1:
                return (
                    f"Error: old_text appears {count} times in that reply. "
                    "Include more surrounding text so it matches exactly once."
                )
            message, location = matches[0]
            return await self._apply(message, location, old_text, new_text)

        return (
            "Error: old_text was not found in your earlier replies. It must match exactly, "
            "and cannot span two Discord messages; try a shorter snippet."
        )

    async def _apply(
        self, message: SentMessage, location: int | None, old_text: str, new_text: str
    ) -> str:
        partial = self.channel.get_partial_message(message.id)
        try:
            if location is None:
                content = message.content.replace(old_text, new_text, 1)
                if len(content) > CONTENT_LIMIT:
                    return _too_long(len(content), CONTENT_LIMIT)
                if not content.strip() and not message.embeds:
                    return "Error: the edit would leave the message empty."
                await partial.edit(content=content, allowed_mentions=AllowedMentions.none())
                message.content = content
            else:
                embeds = [dict(embed) for embed in message.embeds]
                description = embeds[location]["description"].replace(old_text, new_text, 1)
                if len(description) > EMBED_DESCRIPTION_LIMIT:
                    return _too_long(len(description), EMBED_DESCRIPTION_LIMIT)
                embeds[location]["description"] = description
                await partial.edit(embeds=[Embed.from_dict(embed) for embed in embeds])
                message.embeds = embeds
        except NotFound:
            return "Error: that message no longer exists on Discord (it was deleted)."
        except HTTPException as error:
            return f"Error: Discord rejected the edit ({error.status}): {error.text}"
        return "Edited your earlier reply in place."


def _texts(message: SentMessage) -> list[tuple[int | None, str]]:
    """Editable texts of a message: content (location None), then embed descriptions."""
    texts: list[tuple[int | None, str]] = [(None, message.content)]
    texts.extend(
        (index, embed["description"])
        for index, embed in enumerate(message.embeds)
        if isinstance(embed.get("description"), str)
    )
    return texts


def _too_long(length: int, limit: int) -> str:
    return (
        f"Error: the edited message would be {length} characters; Discord allows {limit}. "
        "Shorten new_text, or post the longer version as a normal reply instead."
    )


__all__ = [
    "EDIT_REPLY_TOOL",
    "MAX_TRACKED_REPLIES",
    "EditReplyHandler",
    "SentMessage",
    "recording_send",
    "remember_reply",
]
