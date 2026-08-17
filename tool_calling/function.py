from dataclasses import dataclass


@dataclass
class Function:
    description: str
    name: str
    parameters: dict    # JSON Schema
