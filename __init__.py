"""
A股 DeepResearch Pipeline
基于 LLM 的新闻驱动股票推荐系统
"""

from .config import Config
from .news2stock.pipeline import AStockDeepResearchPipeline
from .step_a_news_parser import NewsParser
from .step_b_stock_pool import StockPoolBuilder
from .step_c_deep_research import DeepResearcher
from .step_d_scoring import StockScorer
from .step_e_report import ReportGenerator

__version__ = "1.0.0"
__all__ = [
    "Config",
    "AStockDeepResearchPipeline",
    "NewsParser",
    "StockPoolBuilder",
    "DeepResearcher",
    "StockScorer",
    "ReportGenerator"
]
