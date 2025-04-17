import os
import openai
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.base_config import API_SETTINGS

openai.api_key = API_SETTINGS['openai']['api_key']
openai.base_url = API_SETTINGS['openai']['base_url']
openai.default_headers = {"x-foo": "true"}

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Hello world!",
        },
    ],
)
print(response.choices[0].message.content)

# 正常会输出结果：Hello there! How can I assist you today ?