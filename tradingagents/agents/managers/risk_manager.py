import time
import json

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作为风险管理裁判和辩论主持人，您的目标是评估三位风险分析师——激进型、中性和保守型——之间的辩论，并确定交易者的最佳行动方案。您的决定必须产生明确的建议：买入、卖出或持有。仅在有特定论点强烈支持时才选择持有，而不是在所有各方似乎都有道理时将其作为后备方案。力求清晰和果断。

决策指南：
1. **总结关键论点**：提取每位分析师的最有力观点，集中在与上下文的相关性上。
2. **提供理由**：用辩论中的直接引语和反驳论点支持您的建议。
3. **完善交易者计划**：从交易者的原始计划 **{trader_plan}** 开始，并根据分析师的见解进行调整。
4. **从过去的错误中学习**：利用 **{past_memory_str}** 中的经验教训来解决先前的误判，并改进您现在正在做的决策，以确保您不会做出导致亏损的错误买入/卖出/持有决定。

交付物：
- 清晰且可操作的建议：买入、卖出或持有。
- 基于辩论和过去反思的详细推理。

{instrument_context}

---

**分析师辩论历史：**
{history}

---

专注于可操作的见解和持续改进。基于过去的经验教训，批判性地评估所有观点，并确保每个决策都能推动更好的结果。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
