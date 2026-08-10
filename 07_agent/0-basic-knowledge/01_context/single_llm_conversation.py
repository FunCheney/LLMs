from openai import OpenAI

import os
import dotenv

dotenv.load_dotenv()
base_url = os.getenv('LOCAL_BASE_URL')
local_mode = os.getenv('LOCAL_MODE')
time_out = os.getenv('TIMEOUT')

client = OpenAI(api_key= "EMPTY", base_url=base_url)


response = client.chat.completions.create(
    model=local_mode,
    messages=[
        {"role":"system", "content":"You are a helpful coding assistant. Follow user instructions."},
        {"role": "user", "content": "hello"}
    ]
)
print(response.choices[0].message.role)
print(response.choices[0].message.content)