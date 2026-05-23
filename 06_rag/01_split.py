import re

text = f"黄昏时分，天空像是被打翻的调色盘，橘红色与绛紫色在天边交融、渗透，温柔地笼罩着整个小镇。远处的山峦渐渐模糊了轮廓，化为一抹深沉的剪影。街道两旁的梧桐树在微风中轻轻摇曳，叶子发出沙沙的声响， 偶尔有几片枯叶旋转着飘落。行人的影子被拉得很长很长，最终融入了暮色之中。当最后一缕余晖消失在地平线下，路灯次第亮起，在深蓝色的夜幕中洒下温暖的光晕，为归家的人指引着方向。"


sent = re.split(r'(。|？|！|\...\...)', text)
print(sent)

# 按照句子来分割
chunks = []
for sen, pun in zip(sent[::2], sent[1::2]):
    chunks.append(sen + pun)

print(chunks)

# 按照字符数
chunks2 = []
for i in range(0, len(text), 28):
    chunks2.append(text[i:i+28])

print(chunks2)

# 按照重复数
def overlap_spit(text, n, stride):
    for i in range(0, len(text), stride):
        prefix = text[i:i+n]
        print(prefix)

overlap_spit(text, 100, 50)


# 递归
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter



