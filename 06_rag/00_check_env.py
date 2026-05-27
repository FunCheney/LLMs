import sys
print("Python路径:", sys.executable)
try:
    import langchain
    print("langchain 版本:", langchain.__version__)
    import chromadb
    print("chromadb 版本:", chromadb.__version__)
except ImportError:
    print("langchain 未找到！")
    # 列出已安装的 langchain 相关包
    # pip list | grep langchain