from abc import ABC, abstractmethod
from .options import InvokeContext, InvokeOption

class BaseCLI(ABC):
    
    @abstractmethod
    def _invoke(self, msg: str) -> str:
        """
        令Agent/LLM CLI处理输入msg
        """
        ...

    def invoke(
        self,
        msg: str,
        *options: InvokeOption,
    ) -> str:
        """
        唤醒入口，触发CLI处理输入请求

        :param msg: Prompt
        :param *options: 输入选项(AOP)
        
        :returns: 处理完成的信息（不采用流式输出）
        """
        ctx = InvokeContext(prompt=msg)

        # 按照option的传入顺序逐个执行前处理
        for option in options:
            option.pre_handle(ctx)
        
        ctx.raw_response = self._invoke(ctx.prompt)
        ctx.response = ctx.raw_response

        # 以相反顺序执行后处理
        for option in reversed(options):
            option.post_handle(ctx)
        
        return ctx.response
