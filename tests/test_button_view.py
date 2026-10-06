from unittest.mock import AsyncMock, MagicMock

from discord.ui import Select

from discord_claude.cogs.claude.tool_registry import TOOL_REGISTRY
from discord_claude.cogs.claude.views import ButtonView


def _make_view(
    conversation_starter=None,
    conversation_key=None,
    initial_tools=None,
    initial_tool_choice=None,
    get_conversation=None,
    on_stop=None,
):
    return ButtonView(
        conversation_starter=conversation_starter or MagicMock(),
        conversation_key=conversation_key or (123, 456),
        initial_tools=initial_tools,
        initial_tool_choice=initial_tool_choice,
        get_conversation=get_conversation or MagicMock(return_value=None),
        on_regenerate=AsyncMock(),
        on_stop=on_stop or AsyncMock(),
    )


def test_init_without_running_event_loop():
    view = _make_view()
    selects = [c for c in view.children if isinstance(c, Select)]
    assert len(selects) == 1
    assert selects[0].min_values == 0
    assert selects[0].max_values == len(TOOL_REGISTRY)


class TestButtonView:
    async def test_init_creates_select(self):
        view = _make_view()
        selects = [c for c in view.children if isinstance(c, Select)]
        assert len(selects) == 1
        assert selects[0].min_values == 0
        assert selects[0].max_values == len(TOOL_REGISTRY)

    async def test_initial_tools_set_defaults(self):
        view = _make_view(initial_tools=["web_search", "memory"])
        select = next(c for c in view.children if isinstance(c, Select))
        defaults = {o.value: o.default for o in select.options}
        assert defaults["web_search"] is True
        assert defaults["memory"] is True
        assert defaults["code_execution"] is False
        assert defaults["web_fetch"] is False

    async def test_initial_tool_choice_none_clears_defaults(self):
        view = _make_view(
            initial_tools=["web_search", "memory"],
            initial_tool_choice={"type": "none"},
        )
        select = next(c for c in view.children if isinstance(c, Select))
        defaults = {o.value: o.default for o in select.options}
        assert defaults["web_search"] is False
        assert defaults["memory"] is False

    async def test_tool_select_updates_tools_and_defaults(self):
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.mcp_preset_names = []
        conversation.params.tool_choice = None
        view = _make_view(
            conversation_starter=starter,
            get_conversation=MagicMock(return_value=conversation),
        )
        # Get real select for verifying defaults afterward
        real_select = next(c for c in view.children if isinstance(c, Select))
        # Use a mock select for .values (read-only property in py-cord)
        mock_select = MagicMock()
        mock_select.values = ["web_search", "memory"]

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        assert conversation.params.tools == ["web_search", "memory"]
        assert conversation.params.tool_choice == {"type": "auto"}
        # Check select defaults updated on the real Select widget
        defaults = {o.value: o.default for o in real_select.options}
        assert defaults["web_search"] is True
        assert defaults["memory"] is True
        assert defaults["code_execution"] is False
        call_args = interaction.response.send_message.call_args
        assert "Tools updated" in call_args.args[0]
        assert "Tool behavior: auto" in call_args.args[0]

    async def test_tool_select_empty_sets_none_and_preserves_existing_tools(self):
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = ["web_search", "memory"]
        conversation.params.mcp_preset_names = []
        conversation.params.tool_choice = {"type": "auto"}
        view = _make_view(
            conversation_starter=starter,
            initial_tools=["web_search", "memory"],
            get_conversation=MagicMock(return_value=conversation),
        )
        real_select = next(c for c in view.children if isinstance(c, Select))
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        assert conversation.params.tools == []
        assert conversation.params.tool_choice == {"type": "none"}
        defaults = {o.value: o.default for o in real_select.options}
        assert defaults["web_search"] is False
        assert defaults["memory"] is False
        call_args = interaction.response.send_message.call_args
        assert "Tool behavior: none" in call_args.args[0]

    async def test_tool_select_none_values_treated_as_empty(self):
        # py-cord 2.8 types Select.values as Optional; the callback guards with
        # `tool_select.values or []`, so a None payload must behave like empty.
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = ["web_search", "memory"]
        conversation.params.mcp_preset_names = []
        conversation.params.tool_choice = {"type": "auto"}
        view = _make_view(
            conversation_starter=starter,
            initial_tools=["web_search", "memory"],
            get_conversation=MagicMock(return_value=conversation),
        )
        mock_select = MagicMock()
        mock_select.values = None

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        assert conversation.params.tools == []
        assert conversation.params.tool_choice == {"type": "none"}
        assert "Tool behavior: none" in interaction.response.send_message.call_args.args[0]

    async def test_tool_select_empty_keeps_auto_when_mcp_active(self):
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = ["web_search"]
        conversation.params.mcp_preset_names = ["github"]
        conversation.params.tool_choice = {"type": "auto"}
        view = _make_view(
            conversation_starter=starter,
            initial_tools=["web_search"],
            get_conversation=MagicMock(return_value=conversation),
        )
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        assert conversation.params.tools == []
        assert conversation.params.tool_choice == {"type": "auto"}
        assert "Tool behavior: auto" in interaction.response.send_message.call_args.args[0]

    async def test_tool_select_empty_keeps_auto_when_advisor_active(self):
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.tools = []
        conversation.params.mcp_preset_names = []
        conversation.params.advisor_model = "claude-opus-4-6"
        conversation.params.tool_choice = None
        view = _make_view(
            conversation_starter=starter,
            get_conversation=MagicMock(return_value=conversation),
        )
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.tool_select_callback(interaction, mock_select)

        assert conversation.params.tools == []
        assert conversation.params.tool_choice == {"type": "auto"}
        assert "Tool behavior: auto" in interaction.response.send_message.call_args.args[0]

    async def test_tool_select_rejects_non_owner(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction, mock_select)
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    async def test_tool_select_no_conversation(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)
        mock_select = MagicMock()
        mock_select.values = []

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.tool_select_callback(interaction, mock_select)
        assert "No active conversation" in interaction.response.send_message.call_args.args[0]

    async def test_stop_button_calls_on_stop(self):
        starter = MagicMock()
        conversation = MagicMock()
        on_stop = AsyncMock()
        view = ButtonView(
            conversation_starter=starter,
            conversation_key=(123, 456),
            get_conversation=MagicMock(return_value=conversation),
            on_regenerate=AsyncMock(),
            on_stop=on_stop,
        )

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.stop_button.callback(interaction)

        on_stop.assert_awaited_once_with((123, 456), starter)
        assert "ended" in interaction.response.send_message.call_args.args[0]

    async def test_stop_button_rejects_non_owner(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.stop_button.callback(interaction)
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    async def test_stop_button_no_conversation(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.stop_button.callback(interaction)
        assert "No active conversation" in interaction.response.send_message.call_args.args[0]

    async def test_play_pause_toggles(self):
        starter = MagicMock()
        conversation = MagicMock()
        conversation.params = MagicMock()
        conversation.params.paused = False
        view = _make_view(
            conversation_starter=starter,
            get_conversation=MagicMock(return_value=conversation),
        )

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()
        interaction.response.is_done = MagicMock(return_value=False)

        await view.play_pause_button.callback(interaction)
        assert conversation.params.paused is True
        assert "paused" in interaction.response.send_message.call_args.args[0]

        interaction.response.send_message.reset_mock()
        await view.play_pause_button.callback(interaction)
        assert conversation.params.paused is False
        assert "resumed" in interaction.response.send_message.call_args.args[0]

    async def test_play_pause_rejects_non_owner(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.play_pause_button.callback(interaction)
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    async def test_regenerate_rejects_non_owner(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)

        interaction = MagicMock()
        interaction.user = MagicMock()
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.regenerate_button.callback(interaction)
        assert "not allowed" in interaction.response.send_message.call_args.args[0]

    async def test_regenerate_no_conversation(self):
        starter = MagicMock()
        view = _make_view(conversation_starter=starter)

        interaction = MagicMock()
        interaction.user = starter
        interaction.response = MagicMock()
        interaction.response.send_message = AsyncMock()

        await view.regenerate_button.callback(interaction)
        assert "No active conversation" in interaction.response.send_message.call_args.args[0]
