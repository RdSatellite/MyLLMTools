import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, get_type_hints

from .function import Function


@dataclass
class Tool:
    type: str
    function: Function

    @classmethod
    def from_callable(cls, func: Callable[..., Any]) -> "Tool":
        type_map: Dict[Type, str] = {
            str: "string",
            int: "integer",
            float: "float",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties: Dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            # Skip *args or **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Resovle parameter JSON schema type
            param_type = type_hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")

            properties[param_name] = {"type": json_type}

            # If parameter has no default value, it is required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build JSON Schema object
        parameters_schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters_schema["required"] = required

        # Extract function description from docstring
        description = inspect.getdoc(func) or ""

        return cls(
            type="function",
            function=Function(
                name=func.__name__,
                description=description.strip(),
                parameters=parameters_schema,
            ),
        )


if __name__ == "__main__":
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Fetch the current weather for a given location"""
        return f"Weather for {location} in {unit}"

    tool = Tool.from_callable(get_weather)
    print(tool)
