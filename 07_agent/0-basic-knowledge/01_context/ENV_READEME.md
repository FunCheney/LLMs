### 本地 LLM 调用实例

#### 模型准备
```text
brew install ollama
ollama serve
# 安装千问 0.6b 模型
ollama pull qwen3:0.6b

# 验证下载的模型
 ollama list
```

#### 注意力机制

使用 snapshot_download 完整下载

方式一：

```python
from huggingface_hub import snapshot_download 
model_path = snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir="./qwen3-0.6B")

```

方式二： 使用 modelscope

```python
from modelscope import snapshot_download
#  下载模型到指定目录
model_dir = snapshot_download('Qwen/Qwen3-0.6B', local_dir='./qwen3-0.6b')
```