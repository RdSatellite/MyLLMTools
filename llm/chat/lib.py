# lib.py 定义对外开放的集成 api 和相关工具

from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .settings import ChatSettings
from message import MessagesBuilder

