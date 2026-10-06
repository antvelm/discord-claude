from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .reply_edits import EDIT_REPLY_TOOL

ExecutionMode = Literal["server", "client", "mcp"]


@dataclass(frozen=True)
class ToolRegistryEntry:
    """Canonical metadata describing an available tool."""

    id: str
    anthropic_tool: dict[str, Any]
    ui_label: str
    ui_description: str
    execution_mode: ExecutionMode
    handler_key: str | None = None


TOOL_REGISTRY: dict[str, ToolRegistryEntry] = {
    "memory": ToolRegistryEntry(
        id="memory",
        anthropic_tool={
            "type": "memory_20250818",
            "name": "memory",
        },
        ui_label="Memory",
        ui_description="Save and recall memories across conversations.",
        execution_mode="client",
        handler_key="memory",
    ),
    "web_search": ToolRegistryEntry(
        id="web_search",
        anthropic_tool={
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 5,
        },
        ui_label="Web Search",
        ui_description="Search the web for current information.",
        execution_mode="server",
    ),
    "web_fetch": ToolRegistryEntry(
        id="web_fetch",
        anthropic_tool={
            "type": "web_fetch_20260309",
            "name": "web_fetch",
            "max_uses": 5,
            "use_cache": False,
        },
        ui_label="Web Fetch",
        ui_description="Fetch full content from web pages.",
        execution_mode="server",
    ),
    # With their default allowed_callers, the web tool versions above make the API
    # add its own `code_execution` tool; a request that also carries an explicit
    # code_execution tool older than code_execution_20260120 fails with a 400
    # tool-name conflict.
    "code_execution": ToolRegistryEntry(
        id="code_execution",
        anthropic_tool={
            "type": "code_execution_20260521",
            "name": "code_execution",
        },
        ui_label="Code Execution",
        ui_description="Run code in a sandbox.",
        execution_mode="server",
    ),
    "edit_reply": ToolRegistryEntry(
        id="edit_reply",
        anthropic_tool=EDIT_REPLY_TOOL,
        ui_label="Edit Replies",
        ui_description="Let Claude fix its earlier replies in place.",
        execution_mode="client",
        handler_key="edit_reply",
    ),
}


def get_anthropic_tools(tool_ids: list[str]) -> list[dict[str, Any]]:
    """Build Anthropic tool payloads for selected registry ids."""
    return [
        entry.anthropic_tool
        for tool_id in tool_ids
        if (entry := TOOL_REGISTRY.get(tool_id)) is not None
    ]


def get_tool_select_options(selected_tools: set[str]) -> list[dict[str, Any]]:
    """Build UI select option values from the registry."""
    return [
        {
            "label": entry.ui_label,
            "value": entry.id,
            "description": entry.ui_description,
            "default": entry.id in selected_tools,
        }
        for entry in TOOL_REGISTRY.values()
    ]


__all__ = [
    "TOOL_REGISTRY",
    "ExecutionMode",
    "ToolRegistryEntry",
    "get_anthropic_tools",
    "get_tool_select_options",
]
