"""
rag_mvp：RAG 最小可用系统（MVP）的项目包。

pyproject.toml 声明了命令行入口：

    [project.scripts]
    rag-mvp = "rag_mvp:main"

因此本包必须暴露 main()。实现放在 rag_mvp/cli.py，这里再导出一次，
安装好之后（`uv sync`，editable 模式下改源码即时生效）即可使用：

    rag-mvp --help
    rag-mvp search "MCP 是什么" -k 3
"""

from rag_mvp.cli import main

__all__ = ["main"]
