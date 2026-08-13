import os
import dotenv
from langchain.chat_models import init_chat_model


dotenv.load_dotenv(encoding='utf-8')
chat_model = init_chat_model(
    model=os.getenv("DEEPSEEK_MODEL"),
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

print(chat_model.invoke("你是谁").content)

print("*" * 50)

