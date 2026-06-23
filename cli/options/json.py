"""
json.py

Util for json strings
"""
import re


def extract_json(raw: str) -> str:
    json_str: str = None
    
    # 1. First we try to get the json as markdown pattern
    json_str = _extract_markdown_json()
    if json_str:
        return json_str
    
    # 2. Fallback to json_object pattern
    json_str = _extract_json_object()
    if json_str:
        return json_str
    
    return None


JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL
)

def _extract_markdown_json(raw: str) -> str | None:
    """
    This function will try to extract markdown blocked json from raw,

    patterns are like:

    ```json
    ...
    ```
    """
    
    match = JSON_BLOCK_PATTERN.search(raw)
    if match:
        return match.group(1)
    
    return None


def _extract_json_object(raw: str) -> str | None:
    """
    Instead of extracting from .md block (which is prefered by LLMs),
    this function try to match {} quoted string, and assume it can be json strings.

    Validate will be finished by pydantic later.
    """
    start = raw.find("{")
    if start == -1:
        return None
    
    depth = 0

    for i in range(start, len(raw)):
        ch = raw[i]

        if ch == "{":
            depth += 1
        
        elif ch == "}":
            depth -= 1

            if depth == 0:
                return raw[start:i+1]
        
    return None
