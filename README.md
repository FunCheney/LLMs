# 手搓大模型

### 1.环境准备

1. 00_setup 安装构建大模型的环境
2. [大模型基础](https://mp.weixin.qq.com/s/N1PsqYRtFJOO40ZKhbB-YA)

### 2.理解分词过程与位置嵌入

1. [手搓大模型--理解词嵌入与分词器](https://mp.weixin.qq.com/s/hoyNXu5LPXeO8LvQtsrsow)
2. [手搓大模型--为大模型准备输入数据](https://mp.weixin.qq.com/s/YFG67hFVLUBrF4LAi_x72Q)
3. 代码实现：02_ch2


### 3.深入理解注意力机制
1. [手搓大模型--实现简单的注意力机制](https://mp.weixin.qq.com/s/h8iyFWyvAzMzmaJkWYt2xQ)
2. [手搓大模型--实现带有可训练权重的自注意力机制](https://mp.weixin.qq.com/s/FStj2Lro-snSJdJrh2yQsQ)
3. [手搓大模型--实现因果自注意力机制](https://mp.weixin.qq.com/s/ME7LHvyJip-sQGgNjmD9Nw)
4. [手搓大模型--实现多头自注意力机制](https://mp.weixin.qq.com/s/WHKxrdzoiyVnhK_XSvd07Q)
5. [手搓大模型--实现一个 ChatGPT 框架](https://mp.weixin.qq.com/s/y9mPs_WuLMEN4KyVukntyw)
6. [手搓大模型--实现 transformer](https://mp.weixin.qq.com/s/zhsxpXvx7nVWtCCP_4tqbQ)
7. 代码实现：03_ch3

### 4.实现 GPT 模型并生成文本
1. [手搓大模型--用你手搓的大模型生成文本](https://mp.weixin.qq.com/s/qExnsj6qm_1864hW0929eA)
2. [手搓大模型--如何评估模型生成文本质量](https://mp.weixin.qq.com/s/aiIcp49tow26H0uQRzXihQ)
3. 代码实现：04_ch4

### 5.预训练
1. [手搓到模型--如何训练模型](https://mp.weixin.qq.com/s/ToqMTjSUsX0Rawki_h3Wcg)
2. [手搓大模型--控制模型的生成结果](https://mp.weixin.qq.com/s/tue307MaG4h5BvTStvGJFA)
3. [手搓大模型--加载开源模型权重，生成预测文本](https://mp.weixin.qq.com/s/nV8lbZsLZGpB6xUzPx0z5Q)
4. 代码实现：05_ch5


### 6.分类微调
06_ch6

### 7.指令微调
07_ch7

#### 什么是指令微调

论文：https://arxiv.org/pdf/2109.01652 

指令微调是一种简单的方法，它结合了预训练 - 微调（pretrain–finetune）和提示学习（prompting）两种范式的优势 —— 通过微调（finetuning）提供
的监督信号， 改善语言模型在推理阶段对文本交互的响应效果。我们的实证结果表明，语言模型具备良好的能力，能够完成完全通过指令描述的任务。


#### 什么是 FLAN 

FLAN 代表的是 "Fine-tuned Language Net"，这是一种通过特定任务微调（fine-tuning）来增强大语言模型的能力的技术。
FLAN 模型是由 Google 提出的，通过对一个基础的大语言模型（如 T5 或者 PaLM）进行微调，使其能够更好地执行指令或完成特定任务。

FLAN 的核心思想是通过多任务学习和针对性微调，使得大模型能够更好地理解并执行由自然语言描述的任务，而不仅仅是训练时所见过的任务。这种微调使得模型
在面对更广泛的任务时能够产生更高质量的结果，并且能够适应不同的输入类型。

简而言之，FLAN 是一种任务指令优化的方法，旨在提升大模型在各种任务下的表现，尤其是增强模型对自然语言指令的理解和响应能力。


# 构建智能体

### 1.环境准备

### 2.agent 初识

#### 2.1 什么是 agent

论文: https://arxiv.org/pdf/2309.07864

agent: 智能体 或者 智能代理.在人工智能领域，智能体被定义为任何能够通过传感器（Sensors）感知其所处环境（Environment），并自主地通过
执行器（Actuators）采取行动（Action）以达成特定目标的实体。

1. 感知环境
2. 做出决策
3. 采取行动



#### 2.2 智能体的构成与运行原理

#### 2.3 如何构建 agent

1. RecAt
2. PlanSolve
3. Reflection


#### 2.4 构建第一个智能体

### 3. 通过 google adk 理解 basic-agent
