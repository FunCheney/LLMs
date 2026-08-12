import langchain
import sys

# LangChain 核心包版本号
print("langchainVersion:  " + langchain.__version__)
# LangChain 实际安装路径（可确认是否来自当前虚拟环境）
print("langchainfile:" + langchain.__file__)

# 当前 Python 解释器版本（如 3.10.x），用于确认运行环境
print(sys.version)
# 当前 Python 可执行文件路径；当你怀疑“包装到了 A 环境，但运行却走了 B 环境”时尤其有用

print("pythonExecutable:"+ sys.executable)