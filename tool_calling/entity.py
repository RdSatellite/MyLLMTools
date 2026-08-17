from dataclasses import dataclass


@dataclass
class Function:
    description: str
    name: str


@dataclass
class Tool:
    type: str
    function: Function
    parameters: str     # JSON Schema
    