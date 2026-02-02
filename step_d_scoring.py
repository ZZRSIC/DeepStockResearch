"""
Step D: 推荐筛选模块
不做任何打分，仅保留候选顺序
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class StockScorer:
    """股票筛选器（无打分）"""
    
    def __init__(self, ts_pro=None, weights: Dict[str, float] = None):
        self.ts_pro = ts_pro
        self.weights = weights or {}
    
    def score_stocks(self, candidate_stocks: List[Dict], 
                    research_results: Dict, parsed_news: Dict) -> List[Dict]:
        """
        不做评分，直接返回候选股票列表
        
        Args:
            candidate_stocks: 候选股票列表
            research_results: 研究结果
            parsed_news: 解析后的新闻
            
        Returns:
            带评分的股票列表
        """
        return candidate_stocks
