"""内置 MCP server（W8 D2）：搜索与抓取。

两个都同时具备"桩上游"与"真 HTTP 上游"两种接法，见 `upstream.py`。
server 本身只做三件事：声明工具、校验参数、把上游结果整理成 MCP 形状。
"""

from .fetch_server import FETCH_TOOL, FetchServer, fetch_client
from .search_server import SEARCH_TOOL, SearchServer, search_client
from .upstream import (
    FetchedPage,
    FetchUpstream,
    HttpFetchUpstream,
    HttpSearchUpstream,
    SearchHit,
    SearchUpstream,
    StubFetchUpstream,
    StubSearchUpstream,
    html_to_text,
)
from .workspace_server import WORKSPACE_TOOLS, WorkspaceServer, workspace_client

__all__ = [
    "FETCH_TOOL",
    "SEARCH_TOOL",
    "WORKSPACE_TOOLS",
    "FetchServer",
    "FetchUpstream",
    "FetchedPage",
    "HttpFetchUpstream",
    "HttpSearchUpstream",
    "SearchHit",
    "SearchServer",
    "SearchUpstream",
    "StubFetchUpstream",
    "StubSearchUpstream",
    "WorkspaceServer",
    "fetch_client",
    "html_to_text",
    "search_client",
    "workspace_client",
]
