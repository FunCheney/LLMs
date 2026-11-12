"""搜索/抓取的**上游**（W8 D2）：真 API 与离线桩只差在这一个接口。

分成两档而不是写死一档，是因为"离线可复现"和"真实外部检索"都要：

- `StubSearchUpstream` / `StubFetchUpstream`：确定性桩，**不联网**。
  CI、`make eval`、录制生成前的冒烟都走它。结果里带 `synthetic: true`，
  报告能一眼看出"这批数字不是真网络"。
- `HttpSearchUpstream` / `HttpFetchUpstream`：真 HTTP，只在显式配置
  `SEARCH_API_URL` / `FETCH_*` 时装配。它们的测试用 `httpx.MockTransport`
  回放固定字节——能力被测到了，CI 不需要联网。

一个刻意的克制：这里**只**负责"发请求 + 抽字段"，不做重试、不做缓存、
不做限流。重试在 `LLMGateway`（ADR-003），缓存与限流在 W11 的中间件；
工具层再写一套，三处策略打架时没人知道该看哪个。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

__all__ = [
    "FetchedPage",
    "FetchUpstream",
    "HttpFetchUpstream",
    "HttpSearchUpstream",
    "SearchHit",
    "SearchUpstream",
    "StubFetchUpstream",
    "StubSearchUpstream",
    "html_to_text",
]

# 抓回来的正文上限（字符）。10 万字符 ≈ 5 万 token，远超任何窗口；
# 真正该做的是"先截断再进 context"，而不是让上游决定我们烧多少 token。
DEFAULT_MAX_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class SearchHit:
    """一条搜索结果。字段刻意少：title/url/snippet 已经够模型判断要不要点进去。"""

    title: str
    url: str
    snippet: str = ""
    # 上游给的名次（1 开始）。回放时要能验证"名次没变"，所以存下来。
    rank: int = 1


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """一次抓取。`text` 是**已抽成纯文本**的正文，不是 HTML。"""

    url: str
    status: int
    text: str = ""
    title: str = ""
    # 是否截断过（正文超过 max_chars）。报告要能回答"答案漏了是因为没抓到还是抓到了没用上"。
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class SearchUpstream(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class FetchUpstream(Protocol):
    async def fetch(self, url: str, max_chars: int) -> FetchedPage: ...


# --------------------------------------------------------------------------- #
# 离线桩：确定性、不联网
# --------------------------------------------------------------------------- #

# 一个极小的"语料"：只为了让离线链路有东西可返回。**不要**拿它当评测语料
# （那是 evals/golden 的职责），它只是让 `demo` 与单测不需要网络。
_STUB_INDEX: tuple[tuple[str, tuple[SearchHit, ...]], ...] = (
    (
        "增长",
        (
            SearchHit("2026 年 B2B 内容营销基准报告", "https://example.com/b2b-content-bench-2026",
                      "样本 480 家企业：内容营销预算占市场预算 28%，线索转化周期中位数 94 天。", 1),
            SearchHit("增长实验的样本量陷阱", "https://example.com/growth-sample-size",
                      "A/B 实验在转化率低于 3% 时需要每组至少 1.2 万样本才有 80% 功效。", 2),
        ),
    ),
    (
        "AI",
        (
            SearchHit("生成式 AI 在营销中的落地现状", "https://example.com/genai-marketing-2026",
                      "调研 312 家团队：67% 已用于文案初稿，仅 19% 用于投放决策。", 1),
        ),
    ),
)

_STUB_PAGES: dict[str, FetchedPage] = {
    "https://example.com/b2b-content-bench-2026": FetchedPage(
        url="https://example.com/b2b-content-bench-2026",
        status=200,
        title="2026 年 B2B 内容营销基准报告",
        text=(
            "样本量 480 家企业，覆盖 SaaS / 制造 / 专业服务三个行业。\n"
            "内容营销预算占市场预算 28%（2025 年为 24%）。\n"
            "线索转化周期中位数 94 天，同比延长 11 天。\n"
            "68% 的受访团队表示「内容产量」是最大瓶颈，而不是「内容质量」。"
        ),
    ),
    "https://example.com/growth-sample-size": FetchedPage(
        url="https://example.com/growth-sample-size",
        status=200,
        title="增长实验的样本量陷阱",
        text=(
            "基线转化率 3%、最小可检测提升 10%（相对）时，\n"
            "每组需要约 1.2 万样本才能达到 80% 功效。\n"
            "样本不足时看到的「显著提升」多数是回归到均值。"
        ),
    ),
}


class StubSearchUpstream:
    """按关键词命中的离线搜索。命中不到返回空列表（**空结果是合法结果**）。"""

    def __init__(self, index: tuple[tuple[str, tuple[SearchHit, ...]], ...] | None = None) -> None:
        self._index = index or _STUB_INDEX

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for keyword, entries in self._index:
            if keyword.lower() in (query or "").lower():
                hits.extend(entries)
        return hits[:max_results]


class StubFetchUpstream:
    """按 URL 精确命中的离线抓取。未命中返回 404 页（不是抛异常）。"""

    def __init__(self, pages: dict[str, FetchedPage] | None = None) -> None:
        self._pages = pages or _STUB_PAGES

    async def fetch(self, url: str, max_chars: int) -> FetchedPage:
        page = self._pages.get(url)
        if page is None:
            return FetchedPage(url=url, status=404, text="", title="", meta={"synthetic": True})
        text = page.text
        truncated = len(text) > max_chars
        return FetchedPage(
            url=page.url,
            status=page.status,
            text=text[:max_chars],
            title=page.title,
            truncated=truncated,
            meta={"synthetic": True},
        )


# --------------------------------------------------------------------------- #
# 真 HTTP 上游
# --------------------------------------------------------------------------- #


class HttpSearchUpstream:
    """真搜索 API。只认一种响应形态：`{"results": [{"title","url","snippet"}]}`。

    收窄成一种是对的：适配多家搜索 API 的差异属于**这个类**的职责，
    扩散到 server 层就意味着改一家要动三处。要接第二家，写第二个实现，
    保持 `SearchUpstream` 协议不变。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._client = client
        self._timeout_s = timeout_s

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            # 密钥只进 header，不进 URL：URL 会进日志、进 trace、进错误信息。
            headers["Authorization"] = f"Bearer {self._api_key}"

        params = {"q": query, "count": str(max_results)}
        response = await self._get(self._endpoint, params=params, headers=headers)
        response.raise_for_status()
        body = response.json()

        raw_results = body.get("results") if isinstance(body, dict) else None
        hits: list[SearchHit] = []
        for index, raw in enumerate(raw_results or [], start=1):
            if not isinstance(raw, dict) or not raw.get("url"):
                continue
            hits.append(
                SearchHit(
                    title=str(raw.get("title") or ""),
                    url=str(raw["url"]),
                    snippet=str(raw.get("snippet") or raw.get("description") or ""),
                    rank=index,
                )
            )
        return hits[:max_results]

    async def _get(
        self, url: str, *, params: dict[str, str], headers: dict[str, str]
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(url, params=params, headers=headers)
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            return await client.get(url, params=params, headers=headers)


class HttpFetchUpstream:
    """真抓取。HTML → 纯文本的抽取规则见 `html_to_text`（刻意粗糙）。"""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 10.0,
        user_agent: str = "agent-framework/0.1 (+w8-fetch)",
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s
        self._user_agent = user_agent

    async def fetch(self, url: str, max_chars: int) -> FetchedPage:
        headers = {"User-Agent": self._user_agent, "Accept": "text/html,*/*"}
        if self._client is not None:
            response = await self._client.get(url, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=self._timeout_s, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)

        content_type = response.headers.get("content-type", "")
        raw = response.text
        text = html_to_text(raw) if "html" in content_type else raw
        truncated = len(text) > max_chars

        return FetchedPage(
            url=str(response.url),
            status=response.status_code,
            text=text[:max_chars],
            title=_extract_title(raw) if "html" in content_type else "",
            truncated=truncated,
            meta={"content_type": content_type, "bytes": len(raw)},
        )


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BLANK_RE = re.compile(r"\n{3,}")


def _extract_title(html: str) -> str:
    match = _TITLE_RE.search(html or "")
    return _collapse(match.group(1)) if match else ""


def _collapse(text: str) -> str:
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", text or "")
    return _BLANK_RE.sub("\n\n", cleaned).strip()


def html_to_text(html: str) -> str:
    """HTML → 纯文本。**刻意粗糙**，不引 bs4/lxml。

    理由：这里的目标是"别把标签和脚本塞进 prompt"，不是"还原排版"。
    引一个解析器会带来三个成本（依赖、解析耗时、解析失败模式），
    换来的是更漂亮的空白——而模型对空白不敏感，对 token 敏感。
    真正需要结构化抽取的站点，应该在 W9 用专门的 Skill 处理。
    """
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html or "")
    # 块级标签换成换行，行内标签直接去掉：否则整篇会挤成一行，
    # 而模型对"段落边界"是有依赖的（列表项会连成一句）。
    spaced = re.sub(
        r"</?(p|div|br|li|tr|h[1-6]|section|article)[^>]*>", "\n", without_scripts, flags=re.I
    )
    return _collapse(_TAG_RE.sub("", spaced))
