from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "您是一位研究员，负责分析过去一周关于公司的基本面信息。请撰写一份关于公司基本面信息的综合报告，包括财务报表、公司概况、公司基本财务状况和公司财务历史，以全面了解公司的基本面信息，为交易者提供参考。确保包含尽可能多的细节。不要简单地陈述趋势是混合的，而是提供可帮助交易者做出决策的详细和精细的分析与见解。"
            + " 请务必在报告末尾附加一个Markdown表格，以组织和呈现报告中的关键点，使其易于阅读。"
            + " 使用可用工具：`get_fundamentals`用于全面公司分析，`get_balance_sheet`、`get_cashflow`和`get_income_statement`用于特定财务报表。",
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
                    "供您参考，当前日期是 {current_date}。我们要关注的公司是 {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
