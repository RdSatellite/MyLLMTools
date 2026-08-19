# agent/react.py
from typing import List, Optional

from llm import AgentLLM
from ..session import History, Message, AssistantMessage, ToolMessage, UserMessage

from ..base import Agent


class ReactAgent(Agent):
    def __init__(
        self, 
        llm: AgentLLM,
        max_steps: int = 10
    ):
        super().__init__(llm)

        # Agent keeps history and know his last handled message
        self.history: History = History()
        self.handled: int = -1
        
        # Prevent infinite loop in ReAct execution
        self.max_steps: int = max_steps

    def _invoke(self) -> Optional[str]:
        if self.handled >= len(self.history) - 1:
            return None     # Nothing to handle, sleep

        step_count = 0
        while step_count < self.max_steps:
            step_count += 1

            # TODO 1: Send current self.history to LLM (with tools enabled)
            rsp = self._llm.chat()

            self.history.append(rsp)

            # TODO 2: Check if the LLM requested any tool executions
            if self._has_tool_calls(rsp):
                # TODO 3: Execute tool calls and append results to history
                tool_results = self._execute_tools(rsp)
                for rsp_msg in tool_results:
                    self.history.append(rsp)
                
                # Continue loop to let LLM observe tool output and decide next stop
                continue
            
            # TODO 4: If no tool calls, LLM completed its reasoning / answered user
            final_text = self._extract_text(rsp)

            # Update handled index to mark all messages up to current history length as processed
            self.handled = len(self.history) - 1
            return final_text
        
        # Safety fallback if max_steps excedded
        self.handled = len(self.history) - 1
        return "Reached maximum execution steps without concluding."

    # --- Helper Stubs to Implement ---

    def _call_llm(self) -> AssistantMessage:
        # TODO 1: Call self.llm with current self.history and self.tools
        pass

    def _has_tool_calls(self, message: AssistantMessage) -> bool:
        # TODO 2: Return True if message contains tool_calls
        pass

    def _execute_tools(self, message: AssistantMessage) -> List[ToolResultMessage]:
        # TODO 3: Iterate tool_calls in message, invoke target tool, wrap in ToolResultMessage
        pass

    def _extract_text(self, message: AssistantMessage) -> str:
        # TODO 4: Extract and join TextBlock content from assistant message
        pass
