"""
Step D: 投资价值打分模块
基于基本面、估值、动量等多维度对股票进行打分
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StockScorer:
    """股票评分器"""
    
    def __init__(self, ts_pro=None, weights: Dict[str, float] = None):
        self.ts_pro = ts_pro
        self.weights = weights or {
            "relevance": 0.3,
            "fundamental": 0.25,
            "valuation": 0.2,
            "momentum": 0.15,
            "risk": 0.1
        }
    
    def score_stocks(self, candidate_stocks: List[Dict], 
                    research_results: Dict, parsed_news: Dict) -> List[Dict]:
        """
        对股票进行综合评分
        
        Args:
            candidate_stocks: 候选股票列表
            research_results: 研究结果
            parsed_news: 解析后的新闻
            
        Returns:
            带评分的股票列表
        """
        scored_stocks = []
        
        for stock in candidate_stocks:
            try:
                scores = self._calculate_scores(stock, research_results, parsed_news)
                
                # 计算综合得分
                total_score = sum(
                    scores.get(key, 0) * weight 
                    for key, weight in self.weights.items()
                )
                
                stock['scores'] = scores
                stock['total_score'] = total_score
                scored_stocks.append(stock)
                
            except Exception as e:
                logger.warning(f"评分失败 {stock.get('name', 'Unknown')}: {e}")
                continue
        
        # 按总分排序
        scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
        
        return scored_stocks
    
    def _calculate_scores(self, stock: Dict, research_results: Dict, 
                         parsed_news: Dict) -> Dict:
        """计算各维度得分"""
        
        scores = {
            "relevance": self._score_relevance(stock, research_results),
            "fundamental": self._score_fundamental(stock),
            "valuation": self._score_valuation(stock),
            "momentum": self._score_momentum(stock),
            "risk": self._score_risk(stock, research_results)
        }
        
        return scores
    
    def _score_relevance(self, stock: Dict, research_results: Dict) -> float:
        """相关性评分（0-100）"""
        base_score = stock.get('relevance_score', 0.5) * 100
        
        # 根据研究结果中的分析调整
        ts_code = stock['ts_code']
        if ts_code in research_results.get('stock_analysis', {}):
            analysis = research_results['stock_analysis'][ts_code]
            
            # 根据影响方向调整
            impact = analysis.get('impact', '').lower()
            if '正面' in impact or 'positive' in impact:
                base_score *= 1.2
            elif '负面' in impact or 'negative' in impact:
                base_score *= 0.8
            
            # 根据确定性调整
            confidence = analysis.get('confidence', '').lower()
            if '高' in confidence or 'high' in confidence:
                base_score *= 1.1
            elif '低' in confidence or 'low' in confidence:
                base_score *= 0.9
        
        return min(base_score, 100)
    
    def _score_fundamental(self, stock: Dict) -> float:
        """基本面评分（0-100）"""
        if not self.ts_pro:
            return 50  # 默认中等分数
        
        try:
            ts_code = stock['ts_code']
            
            # 获取最新财务数据
            income_df = self.ts_pro.income(ts_code=ts_code, 
                                          fields='end_date,revenue,n_income,roe')
            
            if income_df is None or len(income_df) == 0:
                return 50
            
            latest = income_df.iloc[0]
            
            # 基于 ROE、净利润增长等指标评分
            score = 50
            
            # ROE 评分
            roe = latest.get('roe', 0)
            if roe > 15:
                score += 20
            elif roe > 10:
                score += 10
            elif roe > 5:
                score += 5
            
            # 营收和净利润（需要同比数据，这里简化处理）
            revenue = latest.get('revenue', 0)
            n_income = latest.get('n_income', 0)
            
            if revenue > 0 and n_income > 0:
                profit_margin = n_income / revenue * 100
                if profit_margin > 20:
                    score += 15
                elif profit_margin > 10:
                    score += 10
                elif profit_margin > 5:
                    score += 5
            
            return min(score, 100)
            
        except Exception as e:
            logger.debug(f"获取基本面数据失败: {e}")
            return 50
    
    def _score_valuation(self, stock: Dict) -> float:
        """估值评分（0-100）"""
        if not self.ts_pro:
            return 50
        
        try:
            ts_code = stock['ts_code']
            
            # 获取最新估值数据
            daily_basic = self.ts_pro.daily_basic(
                ts_code=ts_code,
                fields='trade_date,pe,pb,ps,total_mv'
            )
            
            if daily_basic is None or len(daily_basic) == 0:
                return 50
            
            latest = daily_basic.iloc[0]
            
            score = 50
            
            # PE 评分（越低越好）
            pe = latest.get('pe', 0)
            if 0 < pe < 15:
                score += 25
            elif 15 <= pe < 30:
                score += 15
            elif 30 <= pe < 50:
                score += 5
            
            # PB 评分（越低越好）
            pb = latest.get('pb', 0)
            if 0 < pb < 2:
                score += 15
            elif 2 <= pb < 4:
                score += 10
            elif 4 <= pb < 6:
                score += 5
            
            return min(score, 100)
            
        except Exception as e:
            logger.debug(f"获取估值数据失败: {e}")
            return 50
    
    def _score_momentum(self, stock: Dict) -> float:
        """动量评分（0-100）"""
        if not self.ts_pro:
            return 50
        
        try:
            ts_code = stock['ts_code']
            
            # 获取最近的日线数据
            daily_df = self.ts_pro.daily(
                ts_code=ts_code,
                fields='trade_date,close,pct_chg'
            )
            
            if daily_df is None or len(daily_df) < 20:
                return 50
            
            # 计算不同周期的收益率
            daily_df = daily_df.sort_values('trade_date')
            
            score = 50
            
            # 近5日涨跌幅
            if len(daily_df) >= 5:
                pct_5d = ((daily_df.iloc[-1]['close'] / daily_df.iloc[-5]['close']) - 1) * 100
                if pct_5d > 5:
                    score += 15
                elif pct_5d > 0:
                    score += 8
                elif pct_5d < -5:
                    score -= 15
            
            # 近20日涨跌幅
            if len(daily_df) >= 20:
                pct_20d = ((daily_df.iloc[-1]['close'] / daily_df.iloc[-20]['close']) - 1) * 100
                if pct_20d > 10:
                    score += 20
                elif pct_20d > 0:
                    score += 10
                elif pct_20d < -10:
                    score -= 20
            
            return max(0, min(score, 100))
            
        except Exception as e:
            logger.debug(f"获取动量数据失败: {e}")
            return 50
    
    def _score_risk(self, stock: Dict, research_results: Dict) -> float:
        """风险评分（0-100，分数越高风险越低）"""
        score = 70  # 基础分
        
        # 根据研究结果中的风险点调整
        ts_code = stock['ts_code']
        if ts_code in research_results.get('stock_analysis', {}):
            analysis = research_results['stock_analysis'][ts_code]
            risks = analysis.get('risks', [])
            
            # 每个风险点扣分
            risk_penalty = len(risks) * 10
            score -= risk_penalty
        
        # 如果有波动率数据，可以进一步调整
        if self.ts_pro:
            try:
                ts_code = stock['ts_code']
                daily_df = self.ts_pro.daily(
                    ts_code=ts_code,
                    fields='trade_date,pct_chg'
                )
                
                if daily_df is not None and len(daily_df) >= 20:
                    # 计算20日波动率
                    volatility = daily_df['pct_chg'].head(20).std()
                    
                    # 高波动扣分
                    if volatility > 5:
                        score -= 20
                    elif volatility > 3:
                        score -= 10
                    
            except Exception as e:
                logger.debug(f"计算波动率失败: {e}")
        
        return max(0, min(score, 100))
