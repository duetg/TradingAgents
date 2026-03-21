import functools
import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "未找到过去的记忆。"

        context = {
            "role": "user",
            "content": f"基于分析师团队的综合分析，这是为 {company_name} 量身定制的投资计划。{instrument_context} 该计划融合了当前技术市场趋势、宏观经济指标和社交媒体情绪的见解。以此计划为基础来评估您的下一个交易决策。\n\n建议投资计划：{investment_plan}\n\n利用这些见解做出明智和战略性的决策。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""您是一位分析市场数据以做出投资决策的交易代理。根据您的分析，提供具体的买入、卖出或持有建议。以坚定的决策结束，并始终在回复结尾加上'最终交易建议：**买入/持有/卖出**'以确认您的建议。不要忘记利用过去决策的经验教训来从错误中学习。以下是您在类似情况下进行交易的一些反思和经验教训：{past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
