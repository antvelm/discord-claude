<h1 align="center">Discord Claude Bot</h1>

<div align="center">

![Hits](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fjdmsharpe%2Fdiscord-claude%2F&label=discord-claude&icon=github&color=%23198754&message=&style=flat&tz=UTC)
[![Version](https://img.shields.io/github/v/tag/jdmsharpe/discord-claude?sort=semver&label=version)](https://github.com/jdmsharpe/discord-claude/tags)
[![License](https://img.shields.io/github/license/jdmsharpe/discord-claude?label=license)](./LICENSE)
[![CI](https://github.com/jdmsharpe/discord-claude/actions/workflows/main.yml/badge.svg)](https://github.com/jdmsharpe/discord-claude/actions/workflows/main.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/jsgreen152/discord-claude?logo=docker&logoColor=white)](https://hub.docker.com/r/jsgreen152/discord-claude)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Codecov](https://codecov.io/gh/jdmsharpe/discord-claude/branch/main/graph/badge.svg)](https://app.codecov.io/gh/jdmsharpe/discord-claude)

</div>

## Overview

A Discord bot built on Pycord 2.0 that wraps Anthropic's Claude API, providing an easy-to-use interface for conversing with Claude models directly in Discord.

## Features

- **Multi-turn Conversations:** Start conversations with Claude that maintain context across multiple messages.
- **Multiple Claude Models:** Choose from Claude Fable (5.1, 5), Opus (5.5, 5, 4.8, 4.7, 4.6, 4.5), Sonnet (5, 4.6, 4.5), and Haiku (4.5).
- **Refusal Fallback (Beta):** If the safety classifiers of Claude Fable 5.1, Claude Fable 5, Claude Opus 5.5, or Claude Opus 5 decline a request, the API retries it on Claude Opus 4.8 in the same round trip. The response notes the fallback, and cost tracking bills each attempt at the rates of the model that ran it.
- **Multimodal Input:** Attach images (JPEG, PNG, GIF, WEBP), PDFs, or text files (TXT, MD, CSV).
- **Built-In Tools:** Enable web search, web fetch, code execution, and memory with `tool_choice` control (`auto` / `none`) and mid-conversation toggles. On Haiku 4.5, which lacks programmatic tool calling, the web tools are sent with `allowed_callers: ["direct"]` so they work instead of failing. Code execution uses `code_execution_20260521`, which can be enabled together with the web tools.
- **Advisor Mode (Beta):** Enable Anthropic's advisor tool so supported executor models can consult a stronger Claude model (Opus 4.8 by default; Opus 5.5 for Opus 5.5 executors; Opus 5 for Opus 5 / Fable 5 executors; Fable 5.1 for Fable 5.1 executors) for higher-quality planning during complex tasks.
- **Remote MCP Support:** Enable trusted remote MCP servers per conversation through named presets, featuring optional authorization, allow-lists, and deferred tool loading.
- **Citations:** Web search and document citations are displayed as a separate Sources embed.
- **Prompt Caching:** Automatic prompt caching reduces costs (cache reads at 10% of input price; 5% on Opus 5.5 and 2.5% on Fable 5.1) and latency on multi-turn conversations.
- **Context Management:** Uses server-side compaction on Fable 5.1, Fable 5, Opus 5.5, Opus 5, Sonnet 5, Opus 4.8 / 4.7 / 4.6, and Sonnet 4.6; Opus 4.5, Sonnet 4.5, and Haiku 4.5 fall back to local summarization once a request's full prompt (cached tokens included) passes 75% of the smaller of the model window and the 200K summarizer window, with warnings as conversations approach the context limit. The tokens either kind of compaction consumes are included in the reply's cost embed; the local summarizer's tokens are billed at Haiku 4.5's rates.
- **Pricing Display:** A one-line embed after each response shows the request cost, input tokens (cache reads and writes included; the cache-read part in parentheses), output tokens (thinking included; the thinking part in parentheses), tool and advisor call counts, and your spend so far today, for example `$0.0871 · 12.5k in (12.5k cached) / 405 out (80 thinking) · 2 searches · $0.09 today` (configurable).
- **Conversation Controls:** Pause, resume, regenerate responses, and end conversations with interactive buttons.
- **Customization:** Fine-tune supported models with system prompts, sampling controls, effort, thinking budgets, and output limits.

## Commands

### `/claude chat`

Start a conversation with Claude.

- **`prompt`** *(required)*: Your initial message to Claude.
- **`model`**: Choose the Claude model (default: Claude Opus 5.5).
- **`system`**: System prompt to set Claude's behavior.
- **`attachment`**: Attach an image, PDF, or text file.
- **`max_tokens`**: Maximum tokens in the response (default: 16384).
- **`web_search` / `web_fetch` / `code_execution` / `memory`**: Toggle individual tools (default: `web_search` and `web_fetch` on, the others off).
- **`advisor`**: Enable Anthropic's advisor beta on supported executor models. The advisor model is picked per executor: Claude Opus 4.8 for Haiku 4.5, Sonnet 4.6, Sonnet 5, Opus 4.6, Opus 4.7, and Opus 4.8 executors; Claude Opus 5.5 for Opus 5.5 executors and Claude Opus 5 for Opus 5 and Fable 5 executors, which only accept Opus 5.5 / Opus 5 / Fable 5 / Fable 5.1 advisors; and Claude Fable 5.1 for Fable 5.1 executors, which accept only Fable 5.1 advisors. Those Opus 5 / Fable advisors return encrypted `advisor_redacted_result` blocks instead of plaintext advice. Opus 4.5 and Sonnet 4.5 cannot use the advisor.
- **`effort`**: Control response effort — low (fast), medium (balanced), high (thorough), xhigh, or max. Per-model limits are enforced before the request is sent: Fable 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, and Sonnet 5 accept all five; Opus 4.6 and Sonnet 4.6 accept everything except `xhigh`; Opus 4.5 stops at `high`; Sonnet 4.5 and Haiku 4.5 do not accept `effort` at all. When `effort` is not set, the model's own default applies (`medium` on Opus 5.5).
- **`thinking_budget`**: Token budget for legacy models that still support extended thinking budgets.
- **`thinking_display`**: `summarized` (default) shows Claude's summarized reasoning in a spoiler after each reply. `updates` (Fable 5.1, Fable 5, and Opus 5.5 only) hides the reasoning and instead posts the short progress lines the model writes between tool calls while the tools run, instead of showing only the typing indicator until the reply is ready.
- **`tool_choice`**: Tool behavior for enabled tools (`auto` or `none`; Fable 5.1 and Opus 5.5 reject forced tool use entirely).
- **Advanced Tuning**: `temperature`, `top_p`, `top_k` (supported models only).
- **`mcp`**: Optional comma-separated MCP preset names (persists for the life of the conversation).

### `/claude effort`

Change the effort level of your active conversation in this channel for the messages that follow (`low`, `medium`, `high`, `xhigh`, or `max`; the same per-model limits as `/claude chat` apply). On Claude Fable 5.1, Claude Opus 5.5, and Claude Opus 5 the change is applied per message (Anthropic's per-message effort beta), so the cached prompt prefix stays valid; on every other model the top-level effort is changed, which changes the rendered prompt and invalidates the cache.

### `/claude check_permissions`

Check if the bot has the necessary permissions in the current channel.

## Setup & Installation

### Prerequisites

- Python 3.11+
- Discord Bot Token
- Anthropic API Key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jdmsharpe/discord-claude.git
   cd discord-claude
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and its runtime dependencies:

   ```bash
   python -m pip install .
   ```

4. Configure your environment variables:

   ```bash
   cp .env.example .env
   ```

### Contributor Setup

Install development tooling for tests, linting, and type checking:

```bash
python -m pip install -e ".[dev]"
```

### Configuration (`.env`)

| Variable | Required | Description |
| --- | --- | --- |
| `BOT_TOKEN` | **Yes** | Your Discord bot token |
| `GUILD_IDS` | **Yes** | Comma-separated Discord server IDs |
| `ANTHROPIC_API_KEY` | **Yes** | Your Anthropic API key |
| `SHOW_COST_EMBEDS` | No | Show cost/spend embeds (`true`, `1`, or `yes` to enable) (Default: `true`) |
| `MEMORIES_DIR` | No | Directory for per-user memory files (Default: `./memories`) |
| `ANTHROPIC_MCP_PRESETS_JSON` | No | Inline JSON object of named MCP presets |
| `ANTHROPIC_MCP_PRESETS_PATH` | No | Path to a JSON file of named MCP presets |
| `CLAUDE_PRICING_PATH` | No | Path to a pricing YAML that overrides the bundled `src/discord_claude/config/pricing.yaml` |
| `LOG_FORMAT` | No | `text` (default) for human-readable logs, or `json` for structured JSON-lines output with per-request IDs |

#### MCP Setup

Configure named presets in `ANTHROPIC_MCP_PRESETS_JSON` or `ANTHROPIC_MCP_PRESETS_PATH`. Duplicate names across both sources are rejected. Example schema:

```json
{
  "github": {
    "url": "https://api.githubcopilot.com/mcp/",
    "authorization_env_var": "GITHUB_MCP_TOKEN",
    "allowed_tools": ["search_issues"],
    "defer_loading": true
  }
}
```

### Running the Bot

**Locally:**

```bash
python src/bot.py
```

*(Note: `src/bot.py` is a thin launcher that delegates to `discord_claude.bot.main`)*

**With Docker:**

```bash
docker compose up -d --build
```

### Using as a Cog

To compose this repo into a larger bot, import the namespaced package:

```python
from discord_claude import ClaudeCog

bot.add_cog(ClaudeCog(bot=bot))
```

## Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create a new application and add a bot in the "Bot" section.
3. Enable **Server Members Intent** and **Message Content Intent** under Privileged Gateway Intents.
4. Copy the bot token and add it to your `.env` file.
5. Go to OAuth2 > URL Generator.
6. Select scopes: `bot`, `applications.commands`.
7. Select permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`, `Embed Links`, `Attach Files`.
8. Use the generated URL to invite the bot to your server.

## Usage

1. Use `/claude chat` to start a conversation.
2. Type messages in the same channel to continue the conversation seamlessly.
3. Use the interactive controls below the message to:
   - 🔄 Regenerate the last response
   - ⏯️ Pause/resume the conversation
   - ⏹️ End the conversation
   - 🔧 Toggle built-in tools mid-conversation via the select menu.
4. **Note on MCP:** If MCP presets are enabled, the bot adds an explicit MCP safety note to the opening embeds and keeps those presets active until the conversation ends.
5. **Note on Advisor:** Advisor guidance is billed separately at the advisor model's rates, so request cost can rise even when the executor model stays the same. The cost line counts advisor calls, but its token counts cover only the executor model.

## Development

### Testing

Tests use `pytest` with `pytest-asyncio` (`asyncio_mode = "auto"`). All tests are mocked (no real API calls).

```bash
# Install developer tooling if you have not already
python -m pip install -e ".[dev]"

# Run tests locally
python -m pytest -q

# Run tests in Docker
docker build --build-arg PYTHON_VERSION=3.11 -f Dockerfile.test -t discord-claude-test:3.11 .
docker run --rm discord-claude-test:3.11 python -m pytest -q

# Run linting and type checks in Docker
docker run --rm discord-claude-test:3.11 sh -lc 'ruff check src tests && ruff format --check src tests && pyright'
```

### Linting & Type Checking

```bash
ruff check src tests
ruff format --check src tests
pyright
```

*Run `git config core.hooksPath .githooks` after cloning to enable the pre-commit hook.*

## License

MIT License - see [LICENSE](LICENSE) for details.
