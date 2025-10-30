
 **📖 从理论，设计，实践 三个方面研究 AI Agent
 🧑‍💻 学习使用成熟的人工智能智能体库，如smolagents、LlamaIndex和LangGraph。
 🤗 Keep Learning, stay awesome
 
### 学习路线 & 与计划安排

#### 2025.10.17 - 2025.10.24

1. 介绍 AI Agent
   * 什么是 Agent， 他是如何工作的
   * Angent 是如何使用决策和推理的
2. LLMs（大语言模型）在 Agent 中的作用
3. 工具与行动
4. Agent 的工作流程
   

#### 2025.10.25 - 2025.11.02

1. 提示词

    当我们在和大模型交互的时候，我们输入的是一段文本。当我们点击发送按钮之后，这一段我们输入的文本， 被
    结构化成大模型能够识别（解读的）文本信息。这里我们称之为--提示词。

    在提示词中会包含我们之前了解的到特殊的标识，这就像当与我们与大模型的约定，这写约定标记文本的开始，结束。大模型在输入的提示词中读取这些特殊的标记
    来采取对应的策略。

    提示词，在用户和大模型之间起到一个桥梁的作用。

2. 特殊 token

    模型正是利用它们来判断用户和大模型对话轮次的起止位置。每个大语言模型（LLM）都有自己的序列结束（EOS，End Of Sequence）标记，
它们在对话中对于消息也采用不同的格式规则和分隔符。

3. 系统提示词

   在本次会话中第一模型的行为，在多轮次的对话中都起知道作用。比如：
    
   ```
     system_message = {
       "role": "system",
       "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
     }
   ```
    
4. 会话消息
   
    多轮上下文消息的维护。
5. 聊天模板

   1. 基础模型
   
   2. 指令模型
   
   基础模型可以被不同的聊天模板微调。当我们在使用不同的指令模型时，我们要确保我们使用的聊天模板是正确的。
   
   3. chatML
   ChatML（Chat Markup Language，对话标记语言）是由 OpenAI 提出的一种结构化对话格式规范，核心作用是定义人类用户与 AI 模型之间的对话交互结构，
   让模型能清晰区分不同角色（如用户、AI 助手）的输入内容，从而更准确理解对话上下文、生成符合逻辑的回应。 其核心设计思路是通过 “角色 - 内容” 的键值对组合，
   明确对话中每一条消息的归属，避免模型混淆对话主体。常见的基础结构包含 3 类核心角色： 
      * system（系统角色）：用于向 AI 模型注入 “指令设定”，比如定义 AI 的身份（如 “专业翻译助手”）、回应规则（如 “用简洁口语化表达”），这类信息通常仅模型可见，不直接展示给用户； 
      * user（用户角色）：承载人类用户的提问、需求或输入内容，是 AI 需要响应的核心对象； 
      * assistant（助手角色）：对应 AI 模型生成的回复内容，也可用于提供 “示例回复”（即 “few-shot 提示”），帮助模型学习特定风格或逻辑。
      ```
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is calculus?"},
            {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
            {"role": "user", "content": "Can you give me an example?"},
         ]
      ```
6. 

8. 

#### 2025.11.03 - 2025.11.09**

