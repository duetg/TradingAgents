from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_news,
        ]

        system_message = (
            "您是一位社交媒体和公司特定新闻研究员/分析师，负责分析过去一周特定公司的社交媒体帖子、最新公司新闻和公众情绪。您将获得一个公司的名称，您的目标是撰写一份综合长报告，详细说明您在查看社交媒体和人们对该公司评价后的分析、见解以及对交易者和投资者关于该公司当前状况的启示，分析人们对公司每天情绪的情绪数据，并查看最新公司新闻。使用get_news(query, start_date, end_date)工具搜索公司特定新闻和社交媒体讨论。尝试从社交媒体到情绪再到新闻，查看所有可能的来源。不要简单地陈述趋势是混合的，而是提供可帮助交易者做出决策的详细和精细的分析与见解。"
            + """ 请务必在报告末尾附加一个Markdown表格，以组织和呈现报告中的关键点，使其易于阅读。
Always preserve the exact ticker symbol provided by the user, including any exchange suffix, and never merge commentary for similarly named companies from other exchanges. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."""
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
