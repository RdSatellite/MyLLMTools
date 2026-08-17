from dataclasses import asdict
from typing import Callable, TypeVar

from .base import Chat, ToolCall, Structured, ConnectionConfig
from tool_calling.tool import Tool
from session.message import Message, AssistantMessage
from session.block import TextBlock
from structured import StructuredHelper

from openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class DeepseekClient(Chat, ToolCall, Structured):
    def __init__(self, cfg: ConnectionConfig):
        self._client = self._build_client(cfg)
        self._model = "deepseek-v4-flash"

    def _build_client(self, cfg) -> OpenAI:
        return OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url
        )

    @property
    def model(self):
        return self._model

    def chat(self, prompt: str) -> str:
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return rsp.choices[0].message.content

    def chat_with_tools(self, prompt: str, tools: list[Callable]):
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=[asdict(Tool.from_callable(t)) for t in tools],
        )
        return rsp.choices[0].message

    def chat_with_structured(self, prompt: str, schema: type[T]) -> T:
        formatted_prompt = (
            f"{prompt}\n\n"
            f"The answer MUST strictly follow this JSON schema:\n"
            f"{StructuredHelper.get_json_schema(schema)}"
            f"Return ONLY the valid JSON object with no extra explanation or markdown formatting."
        )
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_format={"type": "json_object"},
        )

        raw_content = rsp.choices[0].message.content
        return StructuredHelper.parse(raw_content, schema)


# --- helper --- #

def serialize_message(msg: Message) -> dict:
    text = "".join(
        block.text for block in msg.content if isinstance(block, TextBlock)
    )
    return {"role": msg.role, "content": text}
