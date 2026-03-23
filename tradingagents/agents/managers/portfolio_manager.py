from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

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

        prompt = f"""作为组合经理，整合风险分析师的辩论并给出最终交易决策。

{instrument_context}

---

**评级标准**（请使用其中之一）：
- **Buy（买入）**：强烈看多，开仓或加仓
- **Overweight（增持）**：看好，逐步增加仓位
- **Hold（持有）**：维持当前仓位，不操作
- **Underweight（减持）**：减仓，获利了结
- **Sell（卖出）**：清仓或回避

**背景信息：**
- 交易员提案计划：**{trader_plan}**
- 过往决策的经验教训：**{past_memory_str}**

**输出结构要求：**
1. **评级**：请明确输出 Buy / Overweight / Hold / Underweight / Sell 之一。
2. **执行摘要**：简洁的行动计划，包括入场策略、仓位配置、关键风险等级和时间框架。
3. **投资论点**：基于分析师辩论和过往反思的详细推理。

---

**风险分析师辩论历史：**
{history}

---

要果断，每一个结论都要有分析师的具体证据支撑。"""

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

    return portfolio_manager_node
