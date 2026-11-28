

@runtime_checkable
class HistoryReq(Protocol):
    """HistoryReq 表示此请求携带历史消息"""

    def set_history(self):
        ...

    def get_history(self):
        ...


@runtime_checkable
class HistoryLLM(Protocol):
    """HistoryLLM 表示此 LLM Client 能够处理 HistoryReq"""

    def attach_history(self):
        ...