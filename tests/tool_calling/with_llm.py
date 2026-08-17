import os
import sys

from dotenv import load_dotenv

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from client import DeepseekClient
from client.base import ConnectionConfig

load_dotenv()

client = DeepseekClient(ConnectionConfig(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
))


def get_weather(location: str, unit: str = "celsius") -> str:
    """查询某个城市的当前天气。unit 可选 celsius / fahrenheit。"""
    return f"{location} 的天气：晴朗，23 度（{unit}）"


def main():
    prompt = "北京今天天气怎么样？"

    msg = client.chat_with_tools(prompt, [get_weather])

    if not msg.tool_calls:
        print("[no tool call] content =", msg.content)
        return

    for tc in msg.tool_calls:
        print("tool :", tc.function.name)
        print("args :", tc.function.arguments)


if __name__ == "__main__":
    main()
