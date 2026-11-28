# LLM Providers Usages

此文件记录 `llm.provider` 中各实现的用法

## deepseek (`llm.provider.deepseek`)

导入 package：

```python
from llm.provider.deepseek import (
    DeepSeek,
    Model,
    ReasoningEffort,
    ThinkingType,
)
```

### 1. 创建并发起调用

```python
llm = DeepSeek(api_key="sk-...")
reply = llm.chat("Hello, deepseek!")
```

### 2. 设置项

创建后，DeepSeek client 以默认设置项设置会话参数（详见 `llm.provider.deepseek` 中的 `ChatSettings`）。将会作为调用推理时的设置项。

修改参数：

```python
llm.settings.model = Model.DEEPSEEK_V4_PRO
llm.settings.reasoning_effort = ReasoningEffort.HIGH
```

若要提供一次性参数而不希望改变设置，可以通过 `kwargs` 设定，此参数将会优先于 `ChatSettings`，临时覆盖同名字段。

```python
reply = llm.chat(
    "Hello, deepseek!",
    model=Model.DEEPSEEK_V4_PRO,
    reasoning_effort=ReasoningEffort.HIGH,
    thinking_type=ThinkingType.ENABLED,
)
```

支持的参数如下表所示：

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model` | `Model` | `Model.DEEPSEEK_FLASH` | `DEEPSEEK_FLASH` / `DEEPSEEK_V4_PRO` |
| `stream` | `bool` | `False` | 仅支持 `False`；`True` 会抛 `NotImplementedError` |
| `reasoning_effort` | `ReasoningEffort` | `ReasoningEffort.LOW` | `LOW` / `HIGH` / `MAX` |
| `thinking_type` | `ThinkingType` | `ThinkingType.DISABLED` | `DISABLED` / `ENABLED`，自动映射为 `extra_body["thinking"]["type"]` |

其余 `kwargs`（如 `temperature`）会透传给 OpenAI 客户端；`messages` 由 `prompt` 自动生成，不接受覆盖。传入 `extra_body` 时按字典合并，其中 `thinking.type` 始终以 `thinking_type` 为准。
