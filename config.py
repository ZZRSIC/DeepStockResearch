"""
A股 DeepResearch 配置文件
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """配置类"""
    
    # API 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    
    # Tavily 配置
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    USE_TAVILY = True  # 是否使用 Tavily 网络检索
    
    # Tushare 配置
    TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
    
    # 数据源配置
    USE_TUSHARE = True  # 是否使用 Tushare（需要 token）
    USE_AKSHARE = True  # 是否使用 AKShare（免费）
    USE_BAOSTOCK = True  # 是否使用 BaoStock（免费）
    
    # DeepResearch 配置
    MAX_RESEARCH_ROUNDS = 3  # 最大研究轮次
    TOP_K_STOCKS = 10  # 返回 Top K 股票
    
    # 评分权重
    SCORE_WEIGHTS = {
        "relevance": 0.3,      # 相关性权重
        "fundamental": 0.25,   # 基本面权重
        "valuation": 0.2,      # 估值权重
        "momentum": 0.15,      # 动量权重
        "risk": 0.1           # 风险权重（负向）
    }
    
    # 缓存配置
    ENABLE_CACHE = True
    CACHE_DIR = "./cache"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "./deepresearch.log"
