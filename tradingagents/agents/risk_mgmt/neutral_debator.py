import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_conservative_response = risk_debate_state.get("current_conservative_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为中立型风险分析师，您的角色是提供平衡的视角，权衡交易者决策或计划的潜在收益和风险。您优先考虑全面性方法，在评估利弊的同时考虑更广泛的市场趋势、潜在经济转变和多元化策略。以下是交易者的决策：

{trader_decision}

您的任务是挑战激进型和保守型分析师，指出每个观点可能在哪些方面过于乐观或过于谨慎。使用以下数据源的见解来支持温和、可持续的策略以调整交易者的决策：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界大事报告：{news_report}
公司基本面报告：{fundamentals_report}
以下是当前对话历史：{history} 以下是激进派分析师的最新回应：{current_aggressive_response} 以下是保守派分析师的最新回应：{current_conservative_response}。如果没有其他观点的回应，不要虚构，只需陈述您的观点。

通过批判性地分析双方，积极参与解决激进型和保守型论点中的弱点，倡导更平衡的方法。挑战他们的每个观点，以说明为什么中等风险策略可能提供两全其美的优势，在提供增长潜力的同时防止极端波动。专注于辩论而不是简单地呈现数据，旨在表明平衡的观点可以带来最可靠的结果。以对话方式输出，就像您在说话一样，没有任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"中立型分析师：{response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
