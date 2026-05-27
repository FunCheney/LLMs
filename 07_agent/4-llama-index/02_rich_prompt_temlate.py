from llama_index.core.prompts import RichPromptTemplate

context_str = """
 哈哈哈哈，你好
"""


question = '你是谁'

template = RichPromptTemplate(
    """ 提供了上下文信息
    {{ context_str }}
    有了这些信息，请回答问题：{{question}}
    """
)

prompt_str = template.format(context_str=context_str, question=question)
print(prompt_str)

message = template.format_messages(context_str=context_str, prompt_str=prompt_str)
print(message)

