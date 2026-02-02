"""
A股 DeepResearch Pipeline 主程序
整合所有步骤的完整流程
"""

import logging
from typing import Dict, Optional
from config import Config
from step_a_news_parser import NewsParser
from step_b_stock_pool import StockPoolBuilder
from step_c_deep_research import DeepResearcher
from step_d_scoring import StockScorer
from step_e_report import ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AStockDeepResearchPipeline:
    """A股 DeepResearch 完整流程"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化 Pipeline
        
        Args:
            config: 配置对象，如果不提供则使用默认配置
        """
        self.config = config or Config()
        
        # 初始化各个模块
        logger.info("初始化 DeepResearch Pipeline...")
        
        # Step A: 新闻解析器
        self.news_parser = NewsParser(
            api_key=self.config.OPENAI_API_KEY,
            api_base=self.config.OPENAI_API_BASE,
            model=self.config.MODEL_NAME
        )
        
        # 初始化 Tushare
        ts_pro = None
        if self.config.USE_TUSHARE and self.config.TUSHARE_TOKEN:
            try:
                import tushare as ts
                ts.set_token(self.config.TUSHARE_TOKEN)
                ts_pro = ts.pro_api()
                logger.info("Tushare Pro API 初始化成功")
            except Exception as e:
                logger.warning(f"Tushare 初始化失败: {e}")
        
        # Step B: 股票池构建器（启用 DeepResearch）
        self.stock_pool_builder = StockPoolBuilder(
            use_tushare=self.config.USE_TUSHARE,
            use_akshare=self.config.USE_AKSHARE,
            tushare_token=self.config.TUSHARE_TOKEN,
            api_key=self.config.OPENAI_API_KEY,
            api_base=self.config.OPENAI_API_BASE,
            model=self.config.MODEL_NAME
        )
        
        # Step C: 深度研究器
        self.deep_researcher = DeepResearcher(
            api_key=self.config.OPENAI_API_KEY,
            api_base=self.config.OPENAI_API_BASE,
            model=self.config.MODEL_NAME,
            max_rounds=self.config.MAX_RESEARCH_ROUNDS,
            ts_pro=ts_pro,
            tavily_api_key=self.config.TAVILY_API_KEY if self.config.USE_TAVILY else None
        )
        
        # Step D: 股票评分器
        self.stock_scorer = StockScorer(
            ts_pro=ts_pro,
            weights=self.config.SCORE_WEIGHTS
        )
        
        # Step E: 报告生成器
        self.report_generator = ReportGenerator(
            top_k=self.config.TOP_K_STOCKS
        )
        
        logger.info("Pipeline 初始化完成")
    
    def run(self, news_text: str, save_report: bool = True, 
            output_dir: str = "./output") -> Dict:
        """
        运行完整的 DeepResearch 流程
        
        Args:
            news_text: 输入的新闻文本
            save_report: 是否保存报告
            output_dir: 报告输出目录
            
        Returns:
            完整的分析报告
        """
        logger.info("=" * 80)
        logger.info("开始 A股 DeepResearch 分析流程")
        logger.info("=" * 80)
        
        # Step A: 解析新闻
        logger.info("Step A: 解析新闻...")
        parsed_news = self.news_parser.parse_news(news_text)
        logger.info(f"新闻解析完成 - 事件类型: {parsed_news['event_type']}")
        
        # Step B: 构建候选股票池
        logger.info("Step B: 构建候选股票池...")
        candidate_stocks = self.stock_pool_builder.build_candidate_pool(parsed_news)
        logger.info(f"候选股票池构建完成 - 共 {len(candidate_stocks)} 只股票")
        
        if len(candidate_stocks) == 0:
            logger.warning("未找到相关股票，流程终止")
            return {
                "error": "未找到相关股票",
                "parsed_news": parsed_news
            }
        
        # Step C: 深度研究
        logger.info("Step C: 进行深度研究...")
        research_results = self.deep_researcher.conduct_research(
            parsed_news, candidate_stocks
        )
        logger.info(f"深度研究完成 - 生成 {len(research_results['questions'])} 个研究问题")
        logger.info(f"Tavily API 调用次数: {research_results.get('tavily_call_count', 0)}")
        
        # Step D: 投资价值打分
        logger.info("Step D: 计算投资价值评分...")
        scored_stocks = self.stock_scorer.score_stocks(
            candidate_stocks, research_results, parsed_news
        )
        logger.info(f"评分完成 - Top 1: {scored_stocks[0]['name']} (分数: {scored_stocks[0]['total_score']:.2f})")
        
        # Step E: 生成报告
        logger.info("Step E: 生成投资分析报告...")
        report = self.report_generator.generate_report(
            parsed_news, scored_stocks, research_results
        )
        
        # 保存报告
        if save_report:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存 JSON 格式
            json_path = os.path.join(output_dir, f"report_{timestamp}.json")
            self.report_generator.save_report(report, json_path, format='json')
            
            # 保存文本格式
            text_path = os.path.join(output_dir, f"report_{timestamp}.txt")
            self.report_generator.save_report(report, text_path, format='text')
            
            logger.info(f"报告已保存至: {output_dir}")
        
        logger.info("=" * 80)
        logger.info("分析流程完成")
        logger.info(f"Tavily API 总调用次数: {research_results.get('tavily_call_count', 0)}")
        logger.info("=" * 80)
        
        return report
    
    def run_batch(self, news_list: list, output_dir: str = "./output") -> list:
        """
        批量处理多条新闻
        
        Args:
            news_list: 新闻文本列表
            output_dir: 输出目录
            
        Returns:
            报告列表
        """
        reports = []
        
        for idx, news_text in enumerate(news_list, 1):
            logger.info(f"\n处理第 {idx}/{len(news_list)} 条新闻...")
            try:
                report = self.run(news_text, save_report=True, output_dir=output_dir)
                reports.append(report)
            except Exception as e:
                logger.error(f"处理失败: {e}")
                reports.append({"error": str(e)})
        
        return reports


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A股 DeepResearch 新闻分析系统')
    parser.add_argument('--news', type=str, help='新闻文本（直接输入）')
    parser.add_argument('--news-file', type=str, help='新闻文本文件路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--top-k', type=int, default=10, help='返回 Top K 股票')
    
    args = parser.parse_args()
    
    # 读取新闻
    if args.news:
        news_text = args.news
    elif args.news_file:
        with open(args.news_file, 'r', encoding='utf-8') as f:
            news_text = f.read()
    else:
        print("错误: 请提供 --news 或 --news-file 参数")
        return
    
    # 创建 Pipeline
    config = Config()
    config.TOP_K_STOCKS = args.top_k
    
    pipeline = AStockDeepResearchPipeline(config)
    
    # 运行分析
    report = pipeline.run(news_text, save_report=True, output_dir=args.output)
    
    # 只在终端打印推荐股票列表（完整报告已保存到文件）
    if 'error' not in report:
        print("\n" + "=" * 80)
        print("推荐股票列表".center(80))
        print("=" * 80)
        print(f"\n新闻摘要: {report['metadata']['news_summary']}")
        print(f"Tavily API 调用次数: {report['research_summary'].get('tavily_call_count', 0)}\n")
        print("-" * 80)
        print(f"Top {len(report['top_stocks'])} 推荐股票:")
        print("-" * 80)
        
        for stock in report['top_stocks']:
            print(f"\n#{stock['排名']} {stock['股票名称']} ({stock['股票代码']})")
            print(f"    行业: {stock['所属行业']}")
            print(f"    综合评分: {stock['综合评分']}/100")
            print(f"    投资逻辑: {stock['投资逻辑']['为什么相关']}")
        
        print("\n" + "=" * 80)
        print(f"完整投资分析报告已保存至: {args.output}")
        print("=" * 80 + "\n")
    else:
        print(f"分析失败: {report.get('error')}")


if __name__ == "__main__":
    main()
