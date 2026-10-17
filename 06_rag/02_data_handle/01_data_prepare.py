# pip install unstructured[<extra>]
# pip install "unstructured[pdf]"

from unstructured.partition.auto import partition

# pdf 路径
path = "data/《证券分析》本杰明•格雷厄姆.pdf"
# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=path,
    content_type="application/pdf",
)

# 答应解析结果
print(f'解析完成{len(elements)} 个元素，{sum(len(str(e)) for e in elements)} 字符')

from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

for i, e in enumerate(elements[0:10]):
    print(f"{i}: element {e}")
    print(f"category: {types[e.category]}")

