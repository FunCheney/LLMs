"""chunks.jsonl 脏数据清洗脚本。

针对人工抽检 + 冒烟评测暴露的三类数据质量问题做保守清洗（不改内容本身）：

1. 重复印刷：PDF 中第十五章被排印了两次（p117-124「Chapter 15」与 p126-133「第十五章」，
   17 个 chunk 内容完全一致，15.13 小结为近重复）。保留后一版（与第十六章衔接、排版为最终版），
   删除 p117-124 共 19 个 chunk。——解决评测中 15.x section 重复命中浪费 Top-K 名额的问题。
2. 脏 section_title：标题检测器把正文片段误判为标题，且污染后续 chunk 的 section_path。
   按 chunk_id 显式修复 4 个标题，并对全量 section_path 做「脏元素 → 正确父级」替换。
3. 截断标题：'第十五章 上下文压缩（Auto-' 补全为 '第十五章 上下文压缩（Auto-Compact）'。

用法：
    uv run python scripts/clean_chunks.py            # 清洗并原地写回（先生成 .bak 备份）
    uv run python scripts/clean_chunks.py --dry-run  # 只打印清洗报告，不写文件

清洗后必须重建索引（build_vector_index.py / build_bm25_index.py），并重映射 golden 数据集中
被删除 chunk 的 gold_chunk_ids。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
BACKUP_PATH = Path("data/processed/chunks.jsonl.bak")

# ---- 规则 1：删除重复印刷的第十五章（p117-124，旧版「Chapter 15」排版） ----
DROP_PAGE_RANGE = (117, 124)

# ---- 规则 2：按 chunk_id 修复脏 section_title（同时重建 section_path） ----
# 值: (新 section_title, 新 section_path)
TITLE_FIX_BY_ID: dict[str, tuple[str, list[str]]] = {
    # 封面页标题缺失
    "claude-code-0001-p001-p003-c000": ("封面与版权", ["封面与版权"]),
    # 正文片段「Jupyter 的创新：…立刻显示图表」被误判为标题，实际属于 4.5 节
    "claude-code-0001-p032-p033-c064": (
        "4.5 IPython/Jupyter：科学计算的 REPL",
        ["Part II", "Chapter 4", "4.5 IPython/Jupyter：科学计算的 REPL"],
    ),
    # 「没有上下文 / 有上下文」是 12.3 节中对比示例的小标题，被误判为独立标题
    "claude-code-0001-p097-p097-c202": (
        "12.3 为什么 Context Engineering 如此重要",
        ["Part V", "Chapter 12", "12.3 为什么 Context Engineering 如此重要"],
    ),
    "claude-code-0001-p097-p097-c203": (
        "12.3 为什么 Context Engineering 如此重要",
        ["Part V", "Chapter 12", "12.3 为什么 Context Engineering 如此重要"],
    ),
    # 正文片段被误判为标题，实际属于 20.5 Skills 的参数传递
    "claude-code-0001-p163-p163-c358": (
        "20.5 Skills 的参数传递",
        ["第七部分 扩展系统", "第二十章 Skills 系统", "20.5 Skills 的参数传递"],
    ),
}

# ---- 规则 2b：section_path 中脏元素 → 正确父级标题（全量替换） ----
PATH_ELEMENT_FIX: dict[str, str] = {
    "否则，分析 git diff 生成 message": "第二十章 Skills 系统",
    "立刻显示图表": "Chapter 4",
    "没有上下文": "12.3 为什么 Context Engineering 如此重要",
    "有上下文": "12.3 为什么 Context Engineering 如此重要",
    "目标：依赖": "Part II",  # 3.6 节 Makefile 示例片段泄漏为 Chapter 4 的路径根
    "第十五章 上下文压缩（Auto-": "第十五章 上下文压缩（Auto-Compact）",
}

# ---- 规则 3：截断标题补全（作用于所有 chunk 的 section_title） ----
TITLE_RENAME: dict[str, str] = {
    "第十五章 上下文压缩（Auto-": "第十五章 上下文压缩（Auto-Compact）",
}


def _dedupe_path(path: list[str]) -> list[str]:
    """去掉 section_path 中连续重复的元素（替换后可能产生父子同名）。"""
    out: list[str] = []
    for elem in path:
        if not out or out[-1] != elem:
            out.append(elem)
    return out


def clean_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """对 chunk 列表应用全部清洗规则，返回 (清洗后的列表, 统计报告)。"""
    report = {"dropped_dup_print": 0, "title_fixed": 0, "path_fixed": 0, "title_renamed": 0}
    cleaned: list[dict] = []

    for row in rows:
        # 规则 1：删除重复印刷页
        if DROP_PAGE_RANGE[0] <= row["page_start"] <= DROP_PAGE_RANGE[1]:
            report["dropped_dup_print"] += 1
            continue

        row = dict(row)  # 不改原对象

        # 规则 2：按 id 修复标题与路径
        if row["chunk_id"] in TITLE_FIX_BY_ID:
            row["section_title"], row["section_path"] = TITLE_FIX_BY_ID[row["chunk_id"]]
            report["title_fixed"] += 1
        else:
            # 规则 3：截断标题补全
            if row["section_title"] in TITLE_RENAME:
                row["section_title"] = TITLE_RENAME[row["section_title"]]
                report["title_renamed"] += 1
            # 规则 2b：修复 section_path 中的脏元素
            new_path = [PATH_ELEMENT_FIX.get(e, e) for e in row["section_path"]]
            new_path = _dedupe_path(new_path)
            if new_path != row["section_path"]:
                row["section_path"] = new_path
                report["path_fixed"] += 1

        cleaned.append(row)

    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 chunks.jsonl 脏数据")
    parser.add_argument("--dry-run", action="store_true", help="只打印报告，不写文件")
    args = parser.parse_args()

    rows = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    cleaned, report = clean_rows(rows)

    print(f"原始 chunk 数: {len(rows)}")
    print(f"删除重复印刷（p{DROP_PAGE_RANGE[0]}-{DROP_PAGE_RANGE[1]}）: {report['dropped_dup_print']}")
    print(f"修复脏 section_title: {report['title_fixed']}")
    print(f"补全截断标题: {report['title_renamed']}")
    print(f"修复 section_path: {report['path_fixed']}")
    print(f"清洗后 chunk 数: {len(cleaned)}")

    if args.dry_run:
        print("\n[dry-run] 未写文件。")
        return

    if not BACKUP_PATH.exists():
        shutil.copy2(CHUNKS_PATH, BACKUP_PATH)
        print(f"\n已备份原始文件到 {BACKUP_PATH}")
    else:
        print(f"\n备份已存在（{BACKUP_PATH}），跳过备份。")

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"清洗结果已写回 {CHUNKS_PATH}。请重建索引：rag-mvp vector-index && rag-mvp bm25-index")


if __name__ == "__main__":
    main()
