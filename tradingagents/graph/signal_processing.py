# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        messages = [
            (
                "system",
                "您是一位高效的助手，旨在分析由一组分析师提供的段落或财务报告。您的任务是从投资决策中提取信号：卖出、买入或持有。仅将提取的决策（卖出、买入或持有）作为输出，不添加任何其他文本或信息。",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
