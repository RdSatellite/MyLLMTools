import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from structured import StructuredHelper


load_dotenv()

MODEL = "deepseek-v4-flash"
llm = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def invoke(prompt: str) -> str:
    rsp = llm.chat.completions.create(
        model=MODEL,
        messages = [{"role": "user", "content": prompt}],
        temperature=0,
    )
    return rsp.choices[0].message.content

class Person(BaseModel):
    name: str = Field(..., description="Name")
    age: int = Field(..., description="Age (years old)")

print("\n--- Example 1: Extract a person ---\n")

schema_hint = StructuredHelper.get_json_schema(Person)
prompt = (
    "Extract person info from this sentence: \n"
    "Sega Lee is 26 years old, he likes programming. \n\n"
    f"{schema_hint}"
)

raw = invoke(prompt)
print("[Original output]\n", raw)

person = StructuredHelper.parse(raw, Person)
print("[Result]\n", person)
