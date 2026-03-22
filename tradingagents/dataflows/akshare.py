"""akshare-based data fetching for Chinese A-shares market.

This module provides data interfaces for:
- Real-time/historical stock data (沪深京A股)
- Financial statements (financial reports)
- Technical indicators
- News data
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated
import os


# A-share stock symbol normalization
def _normalize_symbol(symbol: str) -> str:
    """Normalize A-share stock symbol to standard format.

    A-shares use 6-digit codes:
    - 6xxxxxx: Shanghai (上交所)
    - 0xxxxxx: Shenzhen (深交所)
    - 3xxxxxx: ChiNext (创业板)
    - 8xxxxxx: STAR (科创板)
    """
    # Remove spaces and convert to string
    symbol = str(symbol).strip()

    # If already 6 digits, return as is
    if len(symbol) == 6 and symbol.isdigit():
        return symbol

    # Try to extract 6-digit code
    import re
    match = re.search(r'(\d{6})', symbol)
    if match:
        return match.group(1)

    return symbol


def get_stock_data(
    symbol: Annotated[str, "ticker symbol (6-digit A-share code)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    adjust: Annotated[str, "adjustment type: '' for none, 'qfq' for forward, 'hfq' for backward"] = "qfq",
) -> str:
    """Get historical OHLCV data for A-share stocks using akshare.

    Args:
        symbol: 6-digit A-share stock code (e.g., "600519" for 贵州茅台)
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
        adjust: Price adjustment type - "qfq" (forward), "hfq" (backward), or "" (none)
    """
    try:
        symbol = _normalize_symbol(symbol)

        # Fetch historical daily data from East Money
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust=adjust
        )

        if df is None or df.empty:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

        # Rename columns to standard names if needed
        column_mapping = {
            "日期": "Date",
            "股票代码": "Code",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
            "成交额": "Turnover",
            "振幅": "Amplitude",
            "涨跌幅": "Change_Pct",
            "涨跌额": "Change",
            "换手率": "Turnover_Rate",
        }

        # Rename columns if they exist
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df = df.rename(columns={cn: en})

        # Convert to CSV string
        csv_string = df.to_csv(index=False)

        # Add header information
        header = f"# A-share stock data for {symbol} from {start_date} to {end_date} (adjust={adjust})\n"
        header += f"# Total records: {len(df)}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving stock data for {symbol}: {str(e)}"


def get_indicators(
    symbol: Annotated[str, "ticker symbol (6-digit A-share code)"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "number of days to look back"] = 60,
) -> str:
    """Get technical indicators for A-share stocks.

    Supported indicators:
    - RSI: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    - MACD_S: MACD Signal line
    - MACD_H: MACD Histogram
    - BOLL: Bollinger Bands (middle)
    - BOLL_U: Bollinger Bands (upper)
    - BOLL_L: Bollinger Bands (lower)
    - MA5, MA10, MA20, MA60: Moving averages
    """
    try:
        symbol = _normalize_symbol(symbol)

        # Calculate date range
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

        # Fetch historical data
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df is None or df.empty:
            return f"No data found for symbol '{symbol}'"

        # Standard column names from akshare
        df = df.rename(columns={
            "日期": "Date",
            "开盘": "Open",
            "收盘": "Close",
            "最高": "High",
            "最低": "Low",
            "成交量": "Volume",
        })

        # Calculate indicators
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        results = []

        indicator = indicator.lower()

        if indicator in ["rsi"]:
            # RSI: Relative Strength Index
            period = 14
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = pd.Series(gain).rolling(window=period).mean()
            avg_loss = pd.Series(loss).rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            for i in range(len(df)):
                date_str = str(df.iloc[i]["Date"])
                if not pd.isna(rsi.iloc[i]):
                    results.append(f"{date_str}: RSI={rsi.iloc[i]:.2f}")

        elif indicator in ["macd", "macds", "macdh"]:
            # MACD: 12, 26, 9
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            for i in range(len(df)):
                date_str = str(df.iloc[i]["Date"])
                if indicator == "macd":
                    val = macd_line.iloc[i]
                    results.append(f"{date_str}: MACD={val:.4f}" if not pd.isna(val) else f"{date_str}: MACD=N/A")
                elif indicator == "macds":
                    val = signal_line.iloc[i]
                    results.append(f"{date_str}: MACD_Signal={val:.4f}" if not pd.isna(val) else f"{date_str}: MACD_Signal=N/A")
                else:
                    val = histogram.iloc[i]
                    results.append(f"{date_str}: MACD_Hist={val:.4f}" if not pd.isna(val) else f"{date_str}: MACD_Hist=N/A")

        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            # Bollinger Bands: 20-period, 2 std
            period = 20
            sma = pd.Series(close).rolling(window=period).mean()
            std = pd.Series(close).rolling(window=period).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)

            for i in range(len(df)):
                date_str = str(df.iloc[i]["Date"])
                if indicator == "boll":
                    val = sma.iloc[i]
                    results.append(f"{date_str}: BOLL={val:.2f}" if not pd.isna(val) else f"{date_str}: BOLL=N/A")
                elif indicator == "boll_ub":
                    val = upper_band.iloc[i]
                    results.append(f"{date_str}: BOLL_Upper={val:.2f}" if not pd.isna(val) else f"{date_str}: BOLL_Upper=N/A")
                else:
                    val = lower_band.iloc[i]
                    results.append(f"{date_str}: BOLL_Lower={val:.2f}" if not pd.isna(val) else f"{date_str}: BOLL_Lower=N/A")

        elif indicator.startswith("close_"):
            # Moving averages like close_50_sma, close_200_sma
            try:
                ma_period = int(indicator.replace("close_", "").replace("_sma", "").replace("_ema", ""))
            except ValueError:
                ma_period = 20

            if "_sma" in indicator:
                ma = pd.Series(close).rolling(window=ma_period).mean()
            else:
                ma = pd.Series(close).ewm(span=ma_period, adjust=False).mean()

            for i in range(len(df)):
                date_str = str(df.iloc[i]["Date"])
                val = ma.iloc[i]
                ma_name = "SMA" if "_sma" in indicator else "EMA"
                results.append(f"{date_str}: MA{ma_period}({ma_name})={val:.2f}" if not pd.isna(val) else f"{date_str}: MA{ma_period}=N/A")

        elif indicator in ["atr"]:
            # ATR: Average True Range
            period = 14
            tr = pd.concat([
                pd.Series(high) - pd.Series(low),
                pd.Series(abs(pd.Series(high) - pd.Series(close).shift(1))),
                pd.Series(abs(pd.Series(low) - pd.Series(close).shift(1)))
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            for i in range(len(df)):
                date_str = str(df.iloc[i]["Date"])
                val = atr.iloc[i]
                results.append(f"{date_str}: ATR={val:.4f}" if not pd.isna(val) else f"{date_str}: ATR=N/A")

        else:
            return f"Indicator '{indicator}' is not supported. Supported: RSI, MACD, MACD_S, MACD_H, BOLL, BOLL_U, BOLL_L, ATR, MA (e.g., close_200_sma)"

        if not results:
            return f"No indicator data available for {indicator}"

        # Build result string
        indicator_descriptions = {
            "rsi": "RSI: Measures momentum to flag overbought(>70)/oversold(<30) conditions.",
            "macd": "MACD: Momentum indicator showing relationship between two EMAs.",
            "macds": "MACD Signal: 9-day EMA of MACD line.",
            "macdh": "MACD Histogram: Gap between MACD and signal line.",
            "boll": "Bollinger Middle: 20-period SMA, basis for Bollinger Bands.",
            "boll_ub": "Bollinger Upper: 2 standard deviations above middle band.",
            "boll_lb": "Bollinger Lower: 2 standard deviations below middle band.",
            "atr": "ATR: Average True Range measures market volatility.",
        }

        desc = indicator_descriptions.get(indicator.lower(), f"Indicator: {indicator}")

        result_str = f"## {indicator.upper()} values from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        result_str += "\n".join(results[-30:])  # Last 30 records
        result_str += f"\n\n{desc}"

        return result_str

    except Exception as e:
        return f"Error retrieving indicator {indicator} for {symbol}: {str(e)}"


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol (6-digit A-share code)"],
    curr_date: Annotated[str, "current date (not used)"] = None,
) -> str:
    """Get company fundamentals overview for A-share stocks.

    Uses East Money individual stock info API.
    """
    try:
        ticker = _normalize_symbol(ticker)

        # Get individual stock info from East Money
        df = ak.stock_individual_info_em(symbol=ticker)

        if df is None or df.empty:
            return f"No fundamentals data found for symbol '{ticker}'"

        # Convert to key-value format
        lines = []
        for _, row in df.iterrows():
            indicator = row.get("指标", "")
            value = row.get("Value", row.get("值", ""))
            if indicator and value:
                lines.append(f"{indicator}: {value}")

        header = f"# Company Fundamentals for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + "\n".join(lines)

    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol (6-digit A-share code)"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None,
) -> str:
    """Get balance sheet data for A-share stocks.

    Uses East Money financial report API.
    """
    try:
        ticker = _normalize_symbol(ticker)

        # Get balance sheet data
        df = ak.stock_zdjz_balanace_sheet_em(symbol=ticker)

        if df is None or df.empty:
            return f"No balance sheet data found for symbol '{ticker}'"

        # Convert to CSV string
        csv_string = df.to_csv(index=False)

        header = f"# Balance Sheet data for {ticker} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol (6-digit A-share code)"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None,
) -> str:
    """Get cash flow statement data for A-share stocks.

    Uses East Money financial report API.
    """
    try:
        ticker = _normalize_symbol(ticker)

        # Get cash flow data
        df = ak.stock_zdjz_cash_flow_em(symbol=ticker)

        if df is None or df.empty:
            return f"No cash flow data found for symbol '{ticker}'"

        # Convert to CSV string
        csv_string = df.to_csv(index=False)

        header = f"# Cash Flow Statement for {ticker} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol (6-digit A-share code)"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "annual",
    curr_date: Annotated[str, "current date (not used)"] = None,
) -> str:
    """Get income statement data for A-share stocks.

    Uses East Money financial report API.
    """
    try:
        ticker = _normalize_symbol(ticker)

        # Get income statement data
        df = ak.stock_zdjz_profit_statement_em(symbol=ticker)

        if df is None or df.empty:
            return f"No income statement data found for symbol '{ticker}'"

        # Convert to CSV string
        csv_string = df.to_csv(index=False)

        header = f"# Income Statement for {ticker} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_news(
    symbol: Annotated[str, "ticker symbol (6-digit A-share code)"],
    start_date: Annotated[str, "start date YYYY-mm-dd"],
    end_date: Annotated[str, "end date YYYY-mm-dd"],
) -> str:
    """Get news for a specific A-share stock.

    Uses East Money stock news API.
    """
    try:
        symbol = _normalize_symbol(symbol)

        # Get stock news from East Money
        df = ak.stock_news_em(symbol=symbol)

        if df is None or df.empty:
            return f"No news found for {symbol}"

        # Parse date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_items = []

        for _, row in df.iterrows():
            title = row.get("新闻标题", row.get("title", ""))
            content = row.get("新闻内容", row.get("content", ""))
            pub_date = row.get("发布时间", row.get("pub_date", ""))
            source = row.get("文章来源", row.get("source", "东方财富"))

            if not title:
                continue

            # Try to parse date
            try:
                if isinstance(pub_date, str):
                    pub_dt = datetime.strptime(pub_date[:10], "%Y-%m-%d")
                    if not (start_dt <= pub_dt <= end_dt):
                        continue
            except (ValueError, TypeError):
                pass

            news_items.append(f"### {title} (来源: {source}, {pub_date})\n{content}\n")

        if not news_items:
            return f"No news found for {symbol} between {start_date} and {end_date}"

        result = f"## {symbol} News, from {start_date} to {end_date}:\n\n"
        result += "\n".join(news_items)

        return result

    except Exception as e:
        return f"Error fetching news for {symbol}: {str(e)}"


def get_global_news(
    curr_date: Annotated[str, "current date YYYY-mm-dd"],
    look_back_days: Annotated[int, "days to look back"] = 7,
    limit: Annotated[int, "max number of articles"] = 10,
) -> str:
    """Get global/macro economic news for Chinese market.

    Uses East Money macro news API.
    """
    try:
        # Calculate date range
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - timedelta(days=look_back_days)

        # Get macro news from East Money
        df = ak.stock_macro_china()

        if df is None or df.empty:
            return f"No global news found for {curr_date}"

        news_items = []

        for _, row in df.head(limit).iterrows():
            title = row.get("新闻标题", row.get("title", ""))
            content = row.get("新闻内容", row.get("content", ""))
            pub_date = row.get("发布时间", row.get("pub_date", ""))

            if not title:
                continue

            news_items.append(f"### {title} ({pub_date})\n{content}\n")

        if not news_items:
            return f"No global news found for {curr_date}"

        result = f"## Global/Macro News, from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        result += "\n".join(news_items)

        return result

    except Exception as e:
        return f"Error fetching global news: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol (6-digit A-share code)"],
) -> str:
    """Get insider transactions (大宗交易) for A-share stocks.

    Uses East Money block trade API.
    """
    try:
        ticker = _normalize_symbol(ticker)

        # Get block trade data
        df = ak.stock_large_deal_em(symbol=ticker)

        if df is None or df.empty:
            return f"No insider transaction data found for symbol '{ticker}'"

        # Convert to CSV string
        csv_string = df.to_csv(index=False)

        header = f"# Insider Transactions (大宗交易) for {ticker}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"# Source: akshare (East Money)\n\n"

        return header + csv_string

    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"
