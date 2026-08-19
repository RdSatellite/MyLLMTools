import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 让脚本从仓库根目录直接运行也能 import 到 structured 包
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from structured import StructuredHelper
from llm import DeepseekLLM
from llm.base import ConnectionConfig


load_dotenv()

llm = DeepseekLLM(ConnectionConfig(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
))


def ask(prompt: str) -> str:
    """把 prompt 丢给 LLM，拿回文本。"""
    return llm.chat([{"role": "user", "content": prompt}])


# --------------------------------------------------------------------------- #
# 示例 1：最简单的单对象抽取
# --------------------------------------------------------------------------- #
class Person(BaseModel):
    name: str = Field(..., description="姓名")
    age: int = Field(..., description="年龄，单位：岁")


def example_1_basic():
    print("\n===== 示例 1：单对象抽取 =====")

    schema_hint = StructuredHelper.get_json_schema(Person)
    prompt = (
        "请从下面这句话中抽取人物信息：\n"
        "「小明今年 12 岁，喜欢打篮球。」\n\n"
        f"{schema_hint}"
    )

    raw = ask(prompt)
    print("[LLM 原始输出]\n", raw)

    person = StructuredHelper.parse(raw, Person)
    print("[解析结果]", person)
    assert person.name and person.age > 0


# --------------------------------------------------------------------------- #
# 示例 2：情感分析（枚举 + 置信度）
# --------------------------------------------------------------------------- #
class Sentiment(BaseModel):
    label: str = Field(..., description="情感倾向，取值：positive / neutral / negative")
    confidence: float = Field(..., description="置信度，0~1 之间的浮点数")
    reason: str = Field(..., description="给出该判断的简短理由")


def example_2_sentiment():
    print("\n===== 示例 2：情感分析 =====")

    text = "这家餐厅上菜很快，但味道一般，性价比不高。"
    prompt = (
        f"请对下面这句话做情感分析：\n「{text}」\n\n"
        f"{StructuredHelper.get_json_schema(Sentiment)}"
    )

    raw = ask(prompt)
    print("[LLM 原始输出]\n", raw)

    result = StructuredHelper.parse(raw, Sentiment)
    print(
        f"[解析结果] label={result.label}, "
        f"confidence={result.confidence}, reason={result.reason}"
    )


# --------------------------------------------------------------------------- #
# 示例 3：嵌套结构 + 列表
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    title: str = Field(..., description="任务标题")
    priority: int = Field(..., description="优先级 1~5，数字越大越紧急")


class TodoList(BaseModel):
    owner: str = Field(..., description="负责人")
    tasks: list[Task] = Field(..., description="任务列表")


def example_3_nested():
    print("\n===== 示例 3：嵌套结构 =====")

    prompt = (
        "帮我把下面这段口述整理成 TODO 列表：\n"
        "「我是 Sega，明天要先交周报（很急），下午跟设计师对齐 UI（一般），"
        "有空的话看下新框架的文档（不急）。」\n\n"
        f"{StructuredHelper.get_json_schema(TodoList)}"
    )

    raw = ask(prompt)
    print("[LLM 原始输出]\n", raw)

    todo = StructuredHelper.parse(raw, TodoList)
    print(f"[解析结果] owner={todo.owner}")
    for i, t in enumerate(todo.tasks, 1):
        print(f"  {i}. [P{t.priority}] {t.title}")


# --------------------------------------------------------------------------- #
# 示例 4：即使 LLM 用 ```json ``` 包裹，也能正确解析
# --------------------------------------------------------------------------- #
def example_4_markdown_wrapped():
    print("\n===== 示例 4：markdown 代码块输出 =====")

    prompt = (
        "请从这句话抽取信息：「Alice, 29 岁，来自上海」。\n"
        "请务必用 ```json ... ``` 代码块包裹你的 JSON 输出。\n\n"
        f"{StructuredHelper.get_json_schema(Person)}"
    )

    raw = ask(prompt)
    print("[LLM 原始输出]\n", raw)

    person = StructuredHelper.parse(raw, Person)
    print("[解析结果]", person)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    example_1_basic()
    example_2_sentiment()
    example_3_nested()
    example_4_markdown_wrapped()
