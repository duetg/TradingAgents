import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings (default: MiniMax for A-shares)
    "llm_provider": "minimax",
    "deep_think_llm": "MiniMax-M2.7",
    "quick_think_llm": "MiniMax-M2.5",
    "backend_url": "https://api.minimaxi.com/v1",
    # Set environment variable: MINIMAX_API_KEY=your_key
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration (default: akshare for A-shares)
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "akshare",       # Options: alpha_vantage, yfinance, akshare
        "technical_indicators": "akshare",  # Options: alpha_vantage, yfinance, akshare
        "fundamental_data": "akshare",      # Options: alpha_vantage, yfinance, akshare
        "news_data": "akshare",             # Options: alpha_vantage, yfinance, akshare
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
