# session/history.py
# chat history definitions


from typing import List
from .message import Message


class History:
    def __init__(self):
        self._data: List[Message] = []

    def append(self, msg: Message):
        self._data.append(msg)

    @property
    def data(self) -> List[Message]:
        return self._data
