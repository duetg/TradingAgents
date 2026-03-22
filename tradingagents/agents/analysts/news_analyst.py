from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "您是一位新闻研究员，负责分析过去一周的最新新闻和趋势。请撰写一份关于当前与交易和宏观经济相关的世界状况的综合报告。使用可用工具：get_news(query, start_date, end_date)用于公司特定或定向新闻搜索，get_global_news(curr_date, look_back_days, limit)用于更广泛的宏观经济新闻。"
            + """ 请务必在报告末尾附加一个Markdown表格，以组织和呈现报告中的关键点，使其易于阅读。
Always preserve the exact ticker symbol provided by the user, including any exchange suffix, and never merge news for similarly named companies from other exchanges. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位有用的AI助手，与其他助手协作。"
                    " 使用提供的工具来推进回答问题。"
                    " 如果您无法完全回答，没关系；另一位拥有不同工具的助手将在您停止的地方继续提供帮助。执行您能做的以取得进展。"
                    " 如果您或任何其他助手有最终交易建议：**买入/持有/卖出**或交付物，"
                    " 请在您的回复前加上最终交易建议：**买入/持有/卖出**，以便团队知道停止。"
                    " 您可以使用以下工具：{tool_names}。\n{system_message}"
                    "供您参考，当前日期是 {current_date}。{instrument_context}"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
