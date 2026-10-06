from unittest.mock import AsyncMock, MagicMock

from discord import Embed, NotFound

from discord_claude.cogs.claude.reply_edits import (
    MAX_TRACKED_REPLIES,
    EditReplyHandler,
    SentMessage,
    recording_send,
    remember_reply,
)


def _channel():
    channel = MagicMock()
    partial = MagicMock()
    partial.edit = AsyncMock()
    channel.get_partial_message.return_value = partial
    return channel, partial


class TestRecordingSend:
    async def test_records_each_sent_message(self):
        first = MagicMock(id=1, content="part one", embeds=[])
        second = MagicMock(id=2, content="part two", embeds=[Embed(description="sources")])
        send = AsyncMock(side_effect=[first, second])
        sink: list[SentMessage] = []

        wrapped = recording_send(send, sink)
        assert await wrapped(content="a") is first
        assert await wrapped(content="b") is second

        assert [m.id for m in sink] == [1, 2]
        assert sink[0].content == "part one"
        assert sink[1].embeds[0]["description"] == "sources"

    async def test_skips_results_without_an_int_id(self):
        sink: list[SentMessage] = []
        await recording_send(AsyncMock(return_value=MagicMock()), sink)()
        await recording_send(AsyncMock(return_value=None), sink)()
        assert sink == []


class TestRememberReply:
    def test_keeps_only_the_newest_replies(self):
        replies: list[list[SentMessage]] = []
        for index in range(MAX_TRACKED_REPLIES + 3):
            remember_reply(replies, [SentMessage(id=index, content=str(index))])
        assert len(replies) == MAX_TRACKED_REPLIES
        assert replies[-1][0].id == MAX_TRACKED_REPLIES + 2

    def test_ignores_empty_reply(self):
        replies: list[list[SentMessage]] = []
        remember_reply(replies, [])
        assert replies == []


class TestEditReplyHandler:
    async def test_edits_message_content_in_place(self):
        channel, partial = _channel()
        message = SentMessage(id=42, content="Top game: Hollow Knight\n-# $0.01")
        handler = EditReplyHandler(channel, [[message]])

        result = await handler.execute({"old_text": "Hollow Knight", "new_text": "Silksong"}, 1)

        assert result.startswith("Edited")
        channel.get_partial_message.assert_called_once_with(42)
        assert partial.edit.await_args.kwargs["content"] == "Top game: Silksong\n-# $0.01"
        assert message.content == "Top game: Silksong\n-# $0.01"

    async def test_prefers_newest_reply(self):
        channel, _ = _channel()
        older = SentMessage(id=1, content="link: example.com")
        newer = SentMessage(id=2, content="link: example.com")
        handler = EditReplyHandler(channel, [[older], [newer]])

        await handler.execute({"old_text": "example.com", "new_text": "example.org"}, 1)

        channel.get_partial_message.assert_called_once_with(2)
        assert older.content == "link: example.com"

    async def test_searches_every_message_of_a_split_reply(self):
        channel, _ = _channel()
        reply = [SentMessage(id=1, content="first half"), SentMessage(id=2, content="second half")]
        handler = EditReplyHandler(channel, [reply])

        await handler.execute({"old_text": "second", "new_text": "2nd"}, 1)

        channel.get_partial_message.assert_called_once_with(2)

    async def test_edits_embed_description(self):
        channel, partial = _channel()
        message = SentMessage(
            id=7,
            content="",
            embeds=[{"title": "Response", "description": "Price: $20", "type": "rich"}],
        )
        handler = EditReplyHandler(channel, [[message]])

        result = await handler.execute({"old_text": "$20", "new_text": "$25"}, 1)

        assert result.startswith("Edited")
        [embed] = partial.edit.await_args.kwargs["embeds"]
        assert embed.description == "Price: $25"
        assert message.embeds[0]["description"] == "Price: $25"

    async def test_ambiguous_match_is_rejected(self):
        channel, partial = _channel()
        handler = EditReplyHandler(channel, [[SentMessage(id=1, content="a b a")]])

        result = await handler.execute({"old_text": "a", "new_text": "c"}, 1)

        assert "2 times" in result
        partial.edit.assert_not_awaited()

    async def test_missing_text_is_reported(self):
        channel, partial = _channel()
        handler = EditReplyHandler(channel, [[SentMessage(id=1, content="hello")]])

        result = await handler.execute({"old_text": "bye", "new_text": "x"}, 1)

        assert "not found" in result
        partial.edit.assert_not_awaited()

    async def test_no_earlier_reply(self):
        channel, _ = _channel()
        result = await EditReplyHandler(channel, []).execute({"old_text": "a", "new_text": "b"}, 1)
        assert "no earlier reply" in result

    async def test_rejects_content_over_discord_limit(self):
        channel, partial = _channel()
        handler = EditReplyHandler(channel, [[SentMessage(id=1, content="x")]])

        result = await handler.execute({"old_text": "x", "new_text": "y" * 2001}, 1)

        assert "2001 characters" in result
        partial.edit.assert_not_awaited()

    async def test_invalid_input(self):
        channel, _ = _channel()
        handler = EditReplyHandler(channel, [[SentMessage(id=1, content="x")]])
        assert "old_text" in await handler.execute({"old_text": "", "new_text": "y"}, 1)
        assert "new_text" in await handler.execute({"old_text": "x"}, 1)

    async def test_deleted_message(self):
        channel, partial = _channel()
        partial.edit.side_effect = NotFound(MagicMock(status=404, reason="Not Found"), "gone")
        message = SentMessage(id=1, content="hello")
        handler = EditReplyHandler(channel, [[message]])

        result = await handler.execute({"old_text": "hello", "new_text": "hi"}, 1)

        assert "deleted" in result
        assert message.content == "hello"
