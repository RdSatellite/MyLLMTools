# json.py 提供 json 解析工具

import re


def extract_json(raw: str) -> str:
    json_str: str = None
    
    # 1. 优先尝试从 markdown 代码块中提取
    json_str = _extract_markdown_json(raw)
    if json_str:
        return json_str

    # 2. 回退到裸 json 对象
    json_str = _extract_json_object(raw)
    if json_str:
        return json_str

    return None


JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.DOTALL
)

def _extract_markdown_json(raw: str) -> str | None:
    """
    尝试从 raw 中提取 markdown 代码块包裹的 json，形如：

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
    不使用 .md 代码块提取（LLM 更偏好这种形式），而是匹配成对的 {} 字符串，
    并假定其内容就是 json。

    合法性交由后续的 pydantic 校验负责。
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
