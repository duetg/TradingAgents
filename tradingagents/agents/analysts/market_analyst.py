from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """您是一位负责分析金融市场的交易助手。您的角色是从以下列表中为给定的市场状况或交易策略选择**最相关指标**。目标是选择最多**8个指标**，以提供互补且不冗余的见解。类别及各类的指标如下：

移动平均线：
- close_50_sma: 50日SMA：中期趋势指标。用途：识别趋势方向并作为动态支撑/阻力位。提示：它滞后于价格；结合更快指标以获得及时信号。
- close_200_sma: 200日SMA：长期趋势基准。用途：确认整体市场趋势并识别黄金交叉/死亡交叉形态。提示：反应较慢；最适合战略趋势确认而非频繁交易入场。
- close_10_ema: 10日EMA：响应灵敏的短期平均线。用途：捕捉动量的快速变化和潜在入场点。提示：在震荡市场中容易产生噪音；与更长平均线配合使用以过滤虚假信号。

MACD相关：
- macd: MACD：通过EMA差值计算动量。用途：寻找交叉和背离作为趋势变化信号。提示：在低波动或横盘市场中用其他指标确认。
- macds: MACD Signal：MACD线的EMA平滑。用途：使用与MACD线的交叉来触发交易。提示：应作为更广泛策略的一部分以避免假信号。
- macdh: MACD Histogram：显示MACD线与其信号线之间的差距。用途：可视化动量强度并及早发现背离。提示：可能波动较大；在快速移动市场中配合额外过滤器使用。

动量指标：
- rsi: RSI：测量动量以标识超买/超卖状况。用途：应用70/30阈值并观察背离以发出反转信号。提示：在强势趋势中，RSI可能保持极端值；始终与趋势分析交叉验证。

波动性指标：
- boll: Bollinger Middle：作为布林带基础的20日SMA。用途：作为价格运动的动态基准。提示：与上下布林带结合使用以有效识别突破或反转。
- boll_ub: Bollinger Upper Band：通常为中间线以上2个标准差。用途：发出潜在超买状况和突破区域信号。提示：用其他工具确认信号；在强势趋势中价格可能沿着布林带运行。
- boll_lb: Bollinger Lower Band：通常为中间线以下2个标准差。用途：表明潜在超卖状况。提示：使用额外分析以避免虚假反转信号。
- atr: ATR：平均真实范围以测量波动性。用途：根据当前市场波动性设置止损位和调整仓位大小。提示：它是反应性指标，因此将其作为更广泛风险管理策略的一部分使用。

成交量指标：
- vwma: VWMA：成交量加权移动平均线。用途：通过整合价格走势与成交量数据来确认趋势。提示：注意成交量突增导致的偏差；与其他成交量分析结合使用。

- 选择提供多样化且互补信息的指标。避免冗余（例如，不要同时选择rsi和stochrsi）。同时简要解释它们为何适合给定的市场背景。当您调用工具时，请使用上述指标的确切名称，因为它们是已定义的参数，否则您的调用将失败。请确保首先调用get_stock_data获取生成指标所需的CSV数据。然后使用get_indicators并指定具体指标名称。请务必在报告末尾附加一个Markdown表格，以组织和呈现报告中的关键点，使其易于阅读。
Always preserve the exact ticker symbol provided by the user, including any exchange suffix, and never mix in similarly named companies from other exchanges. Write a very detailed and nuanced report of the trends you observe. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."""
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
            "market_report": report,
        }

    return market_analyst_node
