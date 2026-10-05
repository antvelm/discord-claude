import pytest
from discord import AllowedMentions, Colour, Embed

from discord_claude.cogs.claude.embed_delivery import send_embed_batches
from discord_claude.cogs.claude.plain_delivery import chunk_markdown, split_embeds


def _embeds(response: str) -> list[Embed]:
    return [
        Embed(
            title="Conversation Started",
            description="**Prompt:** hello there\n**Model:** claude-opus-5-5\n",
        ),
        Embed(title="Thinking", description="||secret||", color=Colour.light_grey()),
        Embed(title="Response", description=response, color=Colour.orange()),
        Embed(title="Sources", description="[a](https://example.com)"),
        Embed(description="$0.0020 · 64 in / 86 out · <$0.01 today"),
    ]


def test_split_embeds_builds_plain_message():
    text, keep = split_embeds(_embeds("Yes, go ahead."))
    assert text == "> hello there\n\nYes, go ahead.\n\n-# $0.0020 · 64 in / 86 out · <$0.01 today"
    assert [e.title for e in keep] == ["Sources"]


def test_split_embeds_ignores_messages_without_response():
    assert split_embeds([Embed(title="Error", description="boom")]) is None


def test_chunk_markdown_keeps_code_fences_balanced():
    text = "intro\n```python\n" + "print(1)\n" * 600 + "```\nend"
    chunks = chunk_markdown(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    assert all(c.count("```") % 2 == 0 for c in chunks)
    assert chunks[1].startswith("```python")


@pytest.mark.asyncio
async def test_send_embed_batches_plain_mode(monkeypatch):
    monkeypatch.setenv("PLAIN_TEXT_REPLIES", "true")
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)
        return kwargs

    view = object()
    await send_embed_batches(send, embeds=_embeds("Hi @everyone"), view=view)
    assert len(sent) == 1
    assert sent[0]["content"].startswith("> hello there")
    assert sent[0]["view"] is view
    assert [e.title for e in sent[0]["embeds"]] == ["Sources"]
    assert sent[0]["allowed_mentions"].to_dict() == AllowedMentions.none().to_dict()


@pytest.mark.asyncio
async def test_send_embed_batches_default_unchanged(monkeypatch):
    monkeypatch.delenv("PLAIN_TEXT_REPLIES", raising=False)
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)

    await send_embed_batches(send, embeds=_embeds("Hi"))
    assert "content" not in sent[0]
    assert len(sent[0]["embeds"]) == 5
