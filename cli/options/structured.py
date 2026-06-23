from typing import TypeVar
from pydantic import BaseModel

from langchain_core.output_parsers import PydanticOutputParser

from .base import InvokeOption
from .json import extract_json

T = TypeVar("T", bound=BaseModel)


class StructuredHelper:
    @staticmethod
    def build_prompt(prompt: str, schema: type[T]) -> str:
        parser = PydanticOutputParser(pydantic_object=schema)
        return (
            f"{prompt}\n\n"
            f"Output MUST follow the format below:\n"
            f"{parser.get_format_instructions()}"
        )

    @staticmethod
    def parse(raw: str, schema: type[T]) -> T:
        # 1. Assume pure json
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass
        
        # 2. Fallback to RE
        json_text = extract_json(raw)
        try:
            return schema.model_validate_json(json_text)
        except Exception:
            pass

        raise ValueError(
            f"Failed to parse structured output:\n{raw}"
        )


class StructuredOption(InvokeOption):
    def __init__(self, schema: type[T]):
        self.schema = schema

    def pre_handle(self, ctx):
        ctx.prompt = StructuredHelper.build_prompt(
            ctx.prompt,
            self.schema
        )
    
    def post_handle(self, ctx):
        ctx.response = StructuredHelper.parse(
            ctx.raw_response,
            self.schema
        )
