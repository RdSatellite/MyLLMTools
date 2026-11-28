# util.py (tool.util) 提供若干工具函数

from typing import Any, Callable, get_origin


# --- Type Translate --- #


_py2json = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_json2py = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def py_type_to_json_type(t: Any) -> str:
    """将 Python 类型转换为 JSON Schema 所需的 JSON 类型"""
    if t in _py2json:
        return _py2json[t]
    elif get_origin(t) in _py2json:
        return _py2json[get_origin(t)]
    else:
        raise ValueError(f"Unsupported type {t!r} when translating python type to json type.")


def json_type_matches(json_type: str, value: Any) -> bool:
    """判断运行时值是否符合给定的 JSON 类型"""
    if not isinstance(value, _json2py[json_type]):
        return False
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False
    return True


# --- Name Convert --- #


def tool_name_of(fn: Callable) -> str:
    """
    限定名，将`.`替换为`_`（部分 provider 会拒绝 `.`）。
    """

    # TODO: 该名称可能会很长；届时可考虑改为签名算法（举例来说，MD5）
    return f"{fn.__module__}.{fn.__qualname__}".replace(".", "_")
