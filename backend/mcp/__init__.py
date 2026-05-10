"""MCP package - in-process server with role-based tool governance."""
from backend.mcp.server import mcp_server, MCPServer
from backend.mcp.tool_registry import ToolRegistry, ToolSpec
from backend.mcp.role_manager import RoleManager
from backend.mcp.permissions import ROLE_PERMISSIONS

__all__ = [
    "mcp_server",
    "MCPServer",
    "ToolRegistry",
    "ToolSpec",
    "RoleManager",
    "ROLE_PERMISSIONS",
]
