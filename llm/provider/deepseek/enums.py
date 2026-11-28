# enums.py 定义 DeepSeek 相关的字符串枚举

from enum import StrEnum


class Model(StrEnum):
    DEEPSEEK_FLASH = "deepseek-flash"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"


class ReasoningEffort(StrEnum):
    LOW = "low"
    HIGH = "high"
    MAX = "max"


class ThinkingType(StrEnum):
    DISABLED = "disabled"
    ENABLED = "enabled"
