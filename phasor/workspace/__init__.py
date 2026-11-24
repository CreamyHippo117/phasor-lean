"""The shared editing surface of the clients.

``Workspace`` (``tools.py``) is every tool as plain data in and out; the MCP server names its
methods, the GUI calls them, and ``remote.py`` carries them over a localhost socket so a
running GUI can host the scene and ``phasor mcp --attach`` can edit the same live engine.
Clients are siblings and never import each other (contract 6); they meet here.
"""

from phasor.workspace.tools import ToolError, Workspace

__all__ = ["ToolError", "Workspace"]
