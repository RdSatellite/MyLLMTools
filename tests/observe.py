"""观察 DeepseekClient 当前三种能力的实际返回，用于决定下一步设计。

运行：python tests/observe.py
每个 case 会打印返回值的 type 和 repr，暴露真实的数据结构。
"""
import json
import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from llm import DeepseekLLM
from llm.base import ConnectionConfig

load_dotenv()

client = DeepseekLLM(ConnectionConfig(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
))


# --- tools ---

def get_weather(location: str, unit: str = "celsius") -> str:
    """查询某个城市的当前天气。unit 可选 celsius / fahrenheit。"""
    return f"{location}：晴朗 23 度（{unit}）"


def get_time(city: str) -> str:
    """查询某个城市的当前当地时间。"""
    return f"{city} 现在是 14:30"


def show(label: str, result):
    print(f"\n===== {label} =====")
    print(f"type : {type(result).__name__}")
    print(f"value: {result!r}")


# --- Case 1: 基本 chat ---
def case_chat():
    r = client.chat([{"role": "user", "content": "用一句话介绍你自己"}])
    show("chat", r)


# --- Case 2: 结构化输出 ---
class Person(BaseModel):
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年龄")


def case_structured():
    r = client.chat_with_structured(
        [{"role": "user", "content": "小明今年 12 岁，请抽取人物信息"}],
        Person,
    )
    show("chat_with_structured", r)


# --- Case 3: 单工具调用 ---
def case_single_tool():
    r = client.chat_with_tools(
        [{"role": "user", "content": "北京今天天气怎么样？"}],
        [get_weather],
    )
    show("chat_with_tools(单工具)", r)
    if r.tool_calls:
        tc = r.tool_calls[0]
        print(f"tool_calls[0].function.name      : {tc.function.name!r}")
        print(f"tool_calls[0].function.arguments : {tc.function.arguments!r}")
        print(f"arguments 类型                   : {type(tc.function.arguments).__name__}")
        try:
            print(f"json.loads(arguments)            : {json.loads(tc.function.arguments)!r}")
        except Exception as e:
            print(f"json.loads 失败                  : {e}")


# --- Case 4: 多工具调用 ---
def case_multi_tool():
    r = client.chat_with_tools(
        [{"role": "user", "content": "北京现在的天气和时间分别是？"}],
        [get_weather, get_time],
    )
    show("chat_with_tools(多工具)", r)
    for tc in (r.tool_calls or []):
        print(f"  - {tc.function.name}({tc.function.arguments})")


# --- Case 5: 完整 tool 循环的缺口（观察当前是否支持多轮） ---
def case_tool_loop_gap():
    r = client.chat_with_tools(
        [{"role": "user", "content": "北京天气怎么样？"}],
        [get_weather],
    )
    if not r.tool_calls:
        print("\n[no tool call]")
        return
    tc = r.tool_calls[0]
    result = get_weather(**json.loads(tc.function.arguments))
    print(f"\n===== 完整 tool 循环（缺口观察） =====")
    print(f"工具执行结果: {result!r}")
    print("→ 观察：当前没有方法能把工具结果以 tool 角色回填给模型（多轮）")


if __name__ == "__main__":
    case_chat()
    case_structured()
    case_single_tool()
    case_multi_tool()
    case_tool_loop_gap()
