from dataclasses import asdict
from typing import Callable, TypeVar

from .base import Chat, ToolCall, Structured, ConnectionConfig
from tool_calling.tool import Tool
from structured import StructuredHelper

from openai import OpenAI
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class DeepseekLLM(Chat, ToolCall, Structured):
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

    def chat(self, messages: list[dict]) -> str:
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return rsp.choices[0].message.content

    def chat_with_tools(self, messages: list[dict], tools: list[Callable]):
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[asdict(Tool.from_callable(t)) for t in tools],
        )
        return rsp.choices[0].message

    def chat_with_structured(self, messages: list[dict], schema: type[T]) -> T:
        schema_hint = {
            "role": "system",
            "content": (
                "The answer MUST strictly follow this JSON schema:\n"
                f"{StructuredHelper.get_json_schema(schema)}\n"
                "Return ONLY the valid JSON object with no extra explanation or markdown formatting."
            ),
        }
        rsp = self._client.chat.completions.create(
            model=self.model,
            messages=[*messages, schema_hint],
            response_format={"type": "json_object"},
        )

        raw_content = rsp.choices[0].message.content
        return StructuredHelper.parse(raw_content, schema)
