"""
Step E: 输出报告模块
生成带证据摘要的投资建议报告
"""

import logging
from typing import Dict, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
    
    def generate_report(self, parsed_news: Dict, scored_stocks: List[Dict],
                       research_results: Dict) -> Dict:
        """
        生成投资建议报告
        
        Args:
            parsed_news: 解析后的新闻
            scored_stocks: 评分后的股票列表
            research_results: 研究结果
            
        Returns:
            完整的报告数据
        """
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "news_summary": parsed_news['event_summary'],
                "event_type": parsed_news['event_type'],
                "total_candidates": len(scored_stocks),
                "top_k": self.top_k
            },
            "news_analysis": self._format_news_analysis(parsed_news),
            "research_summary": self._format_research_summary(research_results),
            "top_stocks": self._format_top_stocks(scored_stocks[:self.top_k], research_results),
            "risk_disclaimer": self._get_disclaimer()
        }
        
        return report
    
    def _format_news_analysis(self, parsed_news: Dict) -> Dict:
        """格式化新闻分析"""
        return {
            "事件类型": parsed_news['event_type'],
            "事件摘要": parsed_news['event_summary'],
            "关键实体": parsed_news['key_entities'],
            "影响路径": parsed_news['impact_paths'],
            "影响时间框架": parsed_news['time_frame'],
            "受影响产业链环节": parsed_news['affected_segments']
        }
    
    def _format_research_summary(self, research_results: Dict) -> Dict:
        """格式化研究摘要"""
        return {
            "研究问题数量": len(research_results.get('questions', [])),
            "研究问题": research_results.get('questions', []),
            "证据数量": len(research_results.get('evidence', [])),
            "分析股票数量": len(research_results.get('stock_analysis', {})),
            "tavily_call_count": research_results.get('tavily_call_count', 0)
        }
    
    def _format_top_stocks(self, top_stocks: List[Dict], 
                          research_results: Dict) -> List[Dict]:
        """格式化 Top K 股票"""
        formatted_stocks = []
        
        for rank, stock in enumerate(top_stocks, 1):
            ts_code = stock['ts_code']
            analysis = research_results.get('stock_analysis', {}).get(ts_code, {})
            
            stock_info = {
                "排名": rank,
                "股票代码": ts_code,
                "股票名称": stock['name'],
                "所属行业": stock['industry'],
                "综合评分": round(stock['total_score'], 2),
                "各维度评分": {
                    "相关性": round(stock['scores']['relevance'], 2),
                    "基本面": round(stock['scores']['fundamental'], 2),
                    "估值": round(stock['scores']['valuation'], 2),
                    "动量": round(stock['scores']['momentum'], 2),
                    "风险控制": round(stock['scores']['risk'], 2)
                },
                "投资逻辑": {
                    "为什么相关": analysis.get('relevance', stock['relevance_reason']),
                    "潜在影响": analysis.get('impact', '未知'),
                    "影响确定性": analysis.get('confidence', '中等'),
                    "时间框架": analysis.get('timeframe', '中期')
                },
                "主要风险点": analysis.get('risks', ['数据不足，无法评估']),
                "证据数量": analysis.get('evidence_count', 0)
            }
            
            formatted_stocks.append(stock_info)
        
        return formatted_stocks
    
    def _get_disclaimer(self) -> str:
        """获取免责声明"""
        return """
本报告由 AI 系统自动生成，仅供参考，不构成投资建议。
投资有风险，入市需谨慎。请结合自身情况做出投资决策。
数据来源：Tushare、AKShare 等公开数据源。
生成时间可能存在延迟，请以实时数据为准。
        """.strip()
    
    def format_text_report(self, report: Dict) -> str:
        """生成文本格式报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("A股 DeepResearch 投资分析报告".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # 元数据
        meta = report['metadata']
        lines.append(f"生成时间: {meta['generated_at']}")
        lines.append(f"新闻摘要: {meta['news_summary']}")
        lines.append(f"事件类型: {meta['event_type']}")
        lines.append(f"候选股票总数: {meta['total_candidates']}")
        lines.append("")
        
        # 新闻分析
        lines.append("-" * 80)
        lines.append("【新闻分析】")
        lines.append("-" * 80)
        news = report['news_analysis']
        lines.append(f"事件类型: {news['事件类型']}")
        lines.append(f"事件摘要: {news['事件摘要']}")
        lines.append(f"影响时间: {news['影响时间框架']}")
        
        if news['关键实体']['companies']:
            lines.append(f"相关公司: {', '.join(news['关键实体']['companies'])}")
        if news['关键实体']['industries']:
            lines.append(f"相关行业: {', '.join(news['关键实体']['industries'])}")
        
        lines.append("")
        lines.append("影响路径:")
        for idx, path in enumerate(news['影响路径'], 1):
            lines.append(f"  {idx}. {path['path']} ({path['direction']}, 置信度: {path['confidence']})")
        
        lines.append("")
        
        # Top K 股票
        lines.append("-" * 80)
        lines.append(f"【Top {len(report['top_stocks'])} 推荐股票】")
        lines.append("-" * 80)
        
        for stock in report['top_stocks']:
            lines.append("")
            lines.append(f"#{stock['排名']} {stock['股票名称']} ({stock['股票代码']})")
            lines.append(f"    行业: {stock['所属行业']}")
            lines.append(f"    综合评分: {stock['综合评分']}/100")
            lines.append(f"    各维度评分: 相关性={stock['各维度评分']['相关性']}, "
                        f"基本面={stock['各维度评分']['基本面']}, "
                        f"估值={stock['各维度评分']['估值']}, "
                        f"动量={stock['各维度评分']['动量']}, "
                        f"风险={stock['各维度评分']['风险控制']}")
            lines.append("")
            lines.append(f"    【投资逻辑】")
            lines.append(f"    ▸ 相关性: {stock['投资逻辑']['为什么相关']}")
            lines.append(f"    ▸ 潜在影响: {stock['投资逻辑']['潜在影响']}")
            lines.append(f"    ▸ 确定性: {stock['投资逻辑']['影响确定性']}")
            lines.append("")
            lines.append(f"    【风险提示】")
            for risk in stock['主要风险点']:
                lines.append(f"    ⚠ {risk}")
            lines.append(f"    证据支持数量: {stock['证据数量']}")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("【免责声明】")
        lines.append("-" * 80)
        lines.append(report['risk_disclaimer'])
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_report(self, report: Dict, output_path: str, format: str = 'json'):
        """保存报告"""
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            elif format == 'text':
                text_report = self.format_text_report(report)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_report)
            
            logger.info(f"报告已保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
