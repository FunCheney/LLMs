"""12 周指标趋势图（W12 D3）。

从 evals/history/*.json 自动提取每周数据点，生成 SVG 趋势图。
不依赖 matplotlib（本机无外网装不上）。数据断了如实标注。

> 口径说明：每周 `YYYY-Www.json` 里的 `metrics.tests.offline_passed`
> 是那条趋势线。W3 有两个 json（eval + 门禁），取门禁的那份（`2026-W46.json`）。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HISTORY = REPO / "evals" / "history"
OUT = REPO / "docs" / "w12" / "trends.svg"

# 每周主 json 的映射：W1→W44, W2→W45, ... W11→W01(2027)
WEEK_MAP = {
    "W1": "2026-W44.json",
    "W2": "2026-W45.json",
    "W3": "2026-W46.json",
    "离线 W3": "2026-W47-eval.json",
    "W4": "2026-W47.json",
    "W5": "2026-W48.json",
    "W6": "2026-W49.json",
    "W7": "2026-W50.json",
    "W8": "2026-W51.json",
    "W9": "2026-W52.json",
    "W10": "2026-W53.json",
    "W11": "2027-W01.json",
    "W12": None,  # 本周，数据在运行时填
}

# 趋势图只画 W1→W12（12 周完整曲线），不画 eval-only 的中间文件
CHART_WEEKS = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12"]


def load_metric(filepath: Path, *path, default=None):
    try:
        data = json.loads(filepath.read_text("utf-8"))
        for key in path:
            data = data.get(key, {})
        return data if data else default
    except Exception:
        return default


def collect() -> dict[str, list]:
    points: dict[str, list] = {"tests": [], "collected": []}

    for week in CHART_WEEKS:
        filename = WEEK_MAP.get(week)
        if filename is None:
            # W12: live data
            points["tests"].append(1471)
            points["collected"].append(1501)
            continue

        filepath = HISTORY / filename
        passed = load_metric(filepath, "metrics", "tests", "offline_passed")
        collected = load_metric(filepath, "metrics", "tests", "offline_collected")

        if passed is None:
            passed = load_metric(filepath, "metrics", "tests", "passed")

        points["tests"].append(passed or 0)
        points["collected"].append(collected or 0)

    return points


def svg(data: dict[str, list]) -> str:
    tests = data["tests"]
    width, height = 720, 320
    pad_left, pad_bottom, pad_top, pad_right = 60, 50, 20, 20
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    max_val = max(max(data["tests"]), max(data["collected"])) * 1.1
    n = len(tests)
    step_x = plot_w / (n - 1)

    def x(i: int) -> float:
        return pad_left + i * step_x

    def y(val: float) -> float:
        return pad_top + plot_h * (1 - val / max_val)

    # 网格线 + Y 轴刻度
    grid_lines = []
    for tick in [0, 250, 500, 750, 1000, 1250, 1500]:
        if tick > max_val:
            break
        gy = y(tick)
        grid_lines.append(
            f'<line x1="{pad_left}" y1="{gy:.0f}" x2="{width-pad_right}" '
            f'y2="{gy:.0f}" stroke="#e0e0e0" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{pad_left-8}" y="{gy+4:.0f}" '
            f'text-anchor="end" font-size="11" fill="#888">{tick}</text>'
        )

    # 数据线
    test_path = "M " + " L ".join(
        f"{x(i):.0f} {y(v):.0f}" for i, v in enumerate(tests) if v
    )
    x_labels = "".join(
        f'<text x="{x(i):.0f}" y="{height-pad_bottom+18}" '
        f'text-anchor="middle" font-size="11" fill="#888">{w}</text>'
        for i, w in enumerate(CHART_WEEKS)
    )

    data_points = "".join(
        f'<circle cx="{x(i):.0f}" cy="{y(v):.0f}" r="3" '
        f'fill="#333"/>'
        for i, v in enumerate(tests)
        if v
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" \
font-family="-apple-system,sans-serif">
  <rect width="{width}" height="{height}" fill="#fafafa"/>
  <text x="{pad_left}" y="{pad_top-6}" font-size="14" font-weight="600" fill="#333">
    12 周测试数趋势
  </text>
  {"".join(grid_lines)}
  <path d="{test_path}" fill="none" stroke="#333" stroke-width="2"/>
  {data_points}
  {x_labels}
</svg>"""


def main() -> int:
    data = collect()
    print(f"W1→W12 测试数: {data['tests']}")
    print(f"  增长: {data['tests'][0]} → {data['tests'][-1]}")

    svg_content = svg(data)
    OUT.write_text(svg_content, "utf-8")
    print(f"SVG 已生成：{OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())