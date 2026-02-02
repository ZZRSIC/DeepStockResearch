"""
Step B: 候选股票池模块
基于新闻解析结果，检索相关的 A股股票
"""

import logging
from typing import Dict, List, Set
import pandas as pd

logger = logging.getLogger(__name__)


class StockPoolBuilder:
    """股票池构建器（带 DeepResearch 功能）"""
    
    def __init__(self, use_tushare: bool = True, use_akshare: bool = True,
                 tushare_token: str = None, api_key: str = None,
                 api_base: str = None, model: str = None):
        self.use_tushare = use_tushare
        self.use_akshare = use_akshare
        self.tushare_token = tushare_token
        
        # 初始化 LLM 客户端（用于 DeepResearch）
        self.client = None
        self.model = model
        if api_key and api_base and model:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=api_base)
                logger.info("LLM 客户端初始化成功，将使用 DeepResearch 增强股票池构建")
            except Exception as e:
                logger.warning(f"LLM 客户端初始化失败: {e}，将使用基础方法")
        
        # 初始化数据源
        self.ts_pro = None
        if self.use_tushare and tushare_token:
            try:
                import tushare as ts
                ts.set_token(tushare_token)
                self.ts_pro = ts.pro_api()
                logger.info("Tushare 初始化成功")
            except Exception as e:
                logger.warning(f"Tushare 初始化失败: {e}")
        
        # 缓存股票基础数据
        self.stock_basic_df = None
        self._load_stock_basic()
    
    def _load_stock_basic(self):
        """加载股票基础信息"""
        try:
            if self.ts_pro:
                # 使用 Tushare 获取股票列表
                self.stock_basic_df = self.ts_pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date'
                )
                logger.info(f"加载了 {len(self.stock_basic_df)} 只股票基础信息")
            elif self.use_akshare:
                # 使用 AKShare 作为备选
                import akshare as ak
                self.stock_basic_df = ak.stock_info_a_code_name()
                logger.info(f"使用 AKShare 加载了 {len(self.stock_basic_df)} 只股票")
        except Exception as e:
            logger.error(f"加载股票基础信息失败: {e}")
            # 创建空 DataFrame
            self.stock_basic_df = pd.DataFrame()
    
    def build_candidate_pool(self, parsed_news: Dict) -> List[Dict]:
        """
        根据解析后的新闻构建候选股票池（含产业链深度分析）
        
        Args:
            parsed_news: 解析后的新闻数据
            
        Returns:
            候选股票列表
        """
        candidate_stocks = []
        
        # 0. LLM DeepResearch 增强（如果启用）
        deepresearch_stocks = []
        if self.client:
            logger.info("启动 LLM DeepResearch 进行股票池深度分析...")
            deepresearch_stocks = self._deepresearch_stock_pool(parsed_news)
            logger.info(f"DeepResearch 发现 {len(deepresearch_stocks)} 只潜在相关股票")
        
        # 1. 直接提及的公司
        direct_stocks = self._find_direct_mentioned_stocks(
            parsed_news['key_entities']['companies']
        )
        logger.info(f"找到 {len(direct_stocks)} 只直接相关股票")

        # 1.5 新闻解析候选公司（覆盖优先）
        expanded_candidates = self._find_company_candidates(
            parsed_news.get('company_candidates', [])
        )
        logger.info(f"找到 {len(expanded_candidates)} 只新闻候选股票")
        
        # 2. 产业链分析股票
        chain_stocks = []
        if 'industry_chain_analysis' in parsed_news:
            chain_stocks = self._find_industry_chain_stocks(
                parsed_news['industry_chain_analysis']
            )
            logger.info(f"找到 {len(chain_stocks)} 只产业链相关股票")
        
        # 3. 行业相关股票
        industry_stocks = self._find_industry_stocks(
            parsed_news['key_entities']['industries']
        )
        logger.info(f"找到 {len(industry_stocks)} 只行业相关股票")
        
        # 4. 概念相关股票（如果使用 AKShare）
        concept_stocks = self._find_concept_stocks(parsed_news)
        
        # 5. 地区相关股票
        region_stocks = self._find_region_stocks(
            parsed_news['key_entities']['regions']
        )
        
        # 合并去重（优先级：DeepResearch > 直接提及 > 产业链 > 行业 > 概念 > 地区）
        all_stocks = self._merge_and_deduplicate(
            deepresearch_stocks, direct_stocks, expanded_candidates, chain_stocks,
            industry_stocks, concept_stocks, region_stocks
        )
        
        logger.info(f"构建候选池完成，共 {len(all_stocks)} 只股票")
        return all_stocks
    
    def _deepresearch_stock_pool(self, parsed_news: Dict) -> List[Dict]:
        """
        使用 LLM 进行股票池深度研究
        
        Args:
            parsed_news: 解析后的新闻数据
            
        Returns:
            DeepResearch 发现的股票列表
        """
        if not self.client:
            return []
        
        try:
            # 构建 DeepResearch 提示词
            prompt = self._build_deepresearch_prompt(parsed_news)
            
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个资深的 A股市场分析师和产业链研究专家，擅长深度挖掘新闻事件背后的投资机会。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # 解析 LLM 返回的股票列表
            stocks = []
            for category, stock_list in result.items():
                if isinstance(stock_list, list):
                    for stock_info in stock_list:
                        company_name = stock_info.get('company_name', '')
                        if company_name:
                            # 在数据库中查找该股票
                            matched = self._search_company_stocks(
                                company_name,
                                f"DeepResearch-{stock_info.get('reason', '未知')}",
                                stock_info.get('confidence', 0.8),
                                stock_info.get('impact', '中性')
                            )
                            stocks.extend(matched)
            
            logger.info(f"DeepResearch 分析完成，识别出 {len(stocks)} 只相关股票")
            return stocks
            
        except Exception as e:
            logger.error(f"DeepResearch 失败: {e}")
            return []
    
    def _build_deepresearch_prompt(self, parsed_news: Dict) -> str:
        """构建 DeepResearch 提示词"""
        event_summary = parsed_news.get('event_summary', '')
        event_type = parsed_news.get('event_type', '')
        companies = parsed_news.get('key_entities', {}).get('companies', [])
        industries = parsed_news.get('key_entities', {}).get('industries', [])
        products = parsed_news.get('key_entities', {}).get('products', [])
        
        return f"""你是资深 A股市场分析师，请分析以下新闻事件，目标是**尽可能覆盖所有相关的 A股上市公司**，并分层标注强/中/弱相关，确保每家公司都有清晰的关系与路径。

【新闻事件】
{event_summary}

【覆盖原则（非常重要）】
1. **覆盖优先但需可解释**：尽量多列公司，但每家公司必须有清晰关系，逻辑链不超过 3 步。
2. **强/中/弱分层**：强相关用于直接受益或关键环节；中相关用于供应链/上下游；弱相关用于更长链路或不确定性更高的标的（低置信度）。
3. **公司名称要求**：请尽量输出 A股公司简称；若新闻主体为非 A股，请输出 A股对标/供应链/竞争公司。
4. **置信度数值**：使用 0-1 的数值（最多两位小数），强相关建议 ≥0.75，中相关 0.55-0.74，弱相关 0.35-0.54。

【分析思路（思考过程不需要输出）】
1. 事件的核心是什么？（政策/订单/事故/监管/技术突破/供给冲击）
2. 事件实施**必需**哪些产品/服务/系统/材料/设备？
3. 直接受益、上游供应、下游应用、竞争替代、政策执行/合规服务分别有哪些公司？

针对"{event_summary}"，请按事件类型深入分析并尽量覆盖相关标的：

**如果是政策/补贴类事件（如：育儿补贴、财政拨款、政府采购）**：
- 执行政策需要哪些系统/平台/服务商（财政/社保/支付/运营平台）
- 政策直接扶持的行业龙头与核心零部件
- 产业链关键环节（材料/设备/软件/渠道）

**如果是企业合同/订单类事件**：
- 新闻中直接提到的公司是否为 A股？如不是，列出 A股对标与供应链
- 订单涉及的关键材料/设备/系统供应商
- 承运、施工、运维、集成等配套服务商

**如果是技术突破/产品发布类事件**：
- A股中拥有该技术/工艺/产品的公司
- 应用场景的龙头与核心零部件供应商
- 关键材料/设备/工艺/软件平台

**如果是行业监管/整顿类事件**：
- 合规要求带来的订单与服务需求（检测/认证/环保/安全设备）
- 行业集中度提升的龙头与替代受益者

【筛选与分层标准】
只保留与新闻**有明确关系**的公司，并按强/中/弱分层：
✅ 直接提及或直接受益（强相关）
✅ 事件实施必需的关键材料/设备/系统/服务（强相关）
✅ 上下游关键环节与明确供应关系（中相关）
✅ 竞争对手/替代方案/对标龙头（中相关）
✅ 逻辑链较长但仍可解释（弱相关，低置信度）
❌ 完全无关或无法解释的公司
❌ 逻辑链超过 3 步且没有明确路径的公司

【第四步：输出格式】
{{
    "direct": [
        {{
            "company_name": "A股公司简称（如：恒生电子、比亚迪）",
            "reason": "30字内说明关系",
            "impact": "正面/负面/中性",
            "confidence": 0.88,
            "logic_chain": "新闻→路径→公司",
            "relationship": "直接提及/直接受益"
        }}
    ],
    "supply_chain": [...],
    "downstream": [...],
    "peers_competitors": [...],
    "infrastructure_services": [...],
    "policy_implementation": [...],
    "weak_related": [...]
}}

【典型案例参考（方向性）】

案例1："某地区发放 10亿育儿补贴"
✅ 直接/政策执行：财政系统/社保平台/支付清算/运维服务商
✅ 中相关：育儿补贴直接扶持的行业龙头
⚠️ 逻辑链过长的标的放入弱相关或不输出

案例2："国家补贴新能源汽车"
✅ 直接/强相关：整车龙头、电池龙头
✅ 中相关：电池材料/关键设备/充电基础设施
⚠️ 远端消费链条标为弱相关

案例3："某公司中标 XX 项目"
✅ 若为 A股：直接列入 direct
✅ 若非 A股：列出 A股对标/供应链/配套服务商

【现在请分析】
事件：{event_summary}
事件类型：{event_type}
提及的公司：{', '.join(companies) if companies else '无'}
相关行业：{', '.join(industries) if industries else '无'}

请严格按照上述思路分析，**优先覆盖相关标的并分层输出**，不要输出无法解释关系的公司。"""
    
    def _find_industry_chain_stocks(self, chain_analysis: Dict) -> List[Dict]:
        """
        根据产业链分析查找相关股票
        
        Args:
            chain_analysis: 产业链分析结果
            
        Returns:
            产业链相关股票列表
        """
        stocks = []
        
        if self.stock_basic_df is None or len(self.stock_basic_df) == 0:
            return stocks
        
        # 1. 上游供应商
        for supplier in chain_analysis.get('upstream_suppliers', []):
            for company in supplier.get('companies', []):
                matched_stocks = self._search_company_stocks(
                    company,
                    f"上游供应商-{supplier.get('segment', '未知')}",
                    0.9,
                    supplier.get('impact', '中性')
                )
                stocks.extend(matched_stocks)
        
        # 2. 中游制造商
        for manufacturer in chain_analysis.get('midstream_manufacturers', []):
            for company in manufacturer.get('companies', []):
                matched_stocks = self._search_company_stocks(
                    company,
                    f"中游制造-{manufacturer.get('segment', '未知')}",
                    0.85,
                    manufacturer.get('impact', '中性')
                )
                stocks.extend(matched_stocks)
        
        # 3. 下游应用
        for downstream in chain_analysis.get('downstream_applications', []):
            for company in downstream.get('companies', []):
                matched_stocks = self._search_company_stocks(
                    company,
                    f"下游应用-{downstream.get('segment', '未知')}",
                    0.8,
                    downstream.get('impact', '中性')
                )
                stocks.extend(matched_stocks)
        
        # 4. 竞争对手
        for competitor in chain_analysis.get('competitors', []):
            company = competitor.get('company', '')
            if company:
                matched_stocks = self._search_company_stocks(
                    company,
                    f"竞争对手-{competitor.get('relationship', '未知')}",
                    0.85,
                    competitor.get('impact', '中性')
                )
                stocks.extend(matched_stocks)
        
        # 5. 间接受益方
        for beneficiary in chain_analysis.get('beneficiaries', []):
            company = beneficiary.get('company', '')
            if company:
                matched_stocks = self._search_company_stocks(
                    company,
                    f"间接受益-{beneficiary.get('reason', '未知')}",
                    0.75,
                    '正面'
                )
                stocks.extend(matched_stocks)
        
        return stocks

    def _find_company_candidates(self, company_candidates: List[Dict]) -> List[Dict]:
        """根据新闻解析的候选公司列表查找股票"""
        stocks = []

        if self.stock_basic_df is None or len(self.stock_basic_df) == 0:
            return stocks

        for item in company_candidates or []:
            if not isinstance(item, dict):
                continue

            company_name = item.get('company_name') or item.get('name') or ''
            if not company_name:
                continue

            impact = item.get('impact', '中性')
            confidence = self._normalize_confidence(item.get('confidence'))

            reason_parts = []
            relationship = item.get('relationship')
            if relationship:
                reason_parts.append(relationship)
            reasoning = item.get('reasoning') or item.get('reason')
            if reasoning:
                reason_parts.append(reasoning)

            reason = "新闻候选"
            if reason_parts:
                reason = f"新闻候选-{','.join(reason_parts)}"

            matched_stocks = self._search_company_stocks(
                company_name,
                reason,
                confidence,
                impact
            )
            stocks.extend(matched_stocks)

        return stocks

    def _normalize_confidence(self, confidence) -> float:
        """将置信度统一为 0-1 的浮点数"""
        if isinstance(confidence, (int, float)):
            return max(0.0, min(1.0, float(confidence)))

        if isinstance(confidence, str):
            text = confidence.strip()
            # 尝试解析数值字符串
            try:
                return max(0.0, min(1.0, float(text)))
            except ValueError:
                pass

            if text in {"高", "很高", "较高"}:
                return 0.85
            if text in {"中", "中等", "一般"}:
                return 0.7
            if text in {"低", "较低"}:
                return 0.55

        return 0.7
    
    def _search_company_stocks(self, company_name: str, reason: str,
                               score: float, impact: str) -> List[Dict]:
        """
        搜索特定公司的股票
        
        Args:
            company_name: 公司名称
            reason: 相关原因
            score: 相关性评分
            impact: 影响方向
            
        Returns:
            匹配的股票列表
        """
        stocks = []
        
        if not company_name or self.stock_basic_df is None:
            return stocks
        
        # 在股票名称中搜索（支持模糊匹配）
        matched = self.stock_basic_df[
            self.stock_basic_df['name'].str.contains(company_name, na=False, case=False)
        ]
        
        for _, row in matched.iterrows():
            stocks.append({
                'ts_code': row.get('ts_code', row.get('code', '')),
                'name': row.get('name', ''),
                'industry': row.get('industry', '未知'),
                'relevance_reason': f'{reason}（{impact}影响）',
                'relevance_score': score,
                'impact_direction': impact
            })
        
        return stocks
    
    def _find_direct_mentioned_stocks(self, companies: List[str]) -> List[Dict]:
        """查找直接提及的公司股票"""
        stocks = []
        
        if self.stock_basic_df is None or len(self.stock_basic_df) == 0:
            return stocks
        
        for company in companies:
            # 在股票名称中搜索
            matched = self.stock_basic_df[
                self.stock_basic_df['name'].str.contains(company, na=False, case=False)
            ]
            
            for _, row in matched.iterrows():
                stocks.append({
                    'ts_code': row.get('ts_code', row.get('code', '')),
                    'name': row.get('name', ''),
                    'industry': row.get('industry', '未知'),
                    'relevance_reason': f'直接提及：{company}',
                    'relevance_score': 1.0
                })
        
        return stocks
    
    def _find_industry_stocks(self, industries: List[str]) -> List[Dict]:
        """查找行业相关股票"""
        stocks = []
        
        if self.stock_basic_df is None or len(self.stock_basic_df) == 0:
            return stocks
        
        for industry in industries:
            # 在行业字段中搜索
            matched = self.stock_basic_df[
                self.stock_basic_df['industry'].str.contains(industry, na=False, case=False)
            ]
            
            # 限制每个行业返回的股票数量
            for _, row in matched.head(20).iterrows():
                stocks.append({
                    'ts_code': row.get('ts_code', row.get('code', '')),
                    'name': row.get('name', ''),
                    'industry': row.get('industry', '未知'),
                    'relevance_reason': f'行业相关：{industry}',
                    'relevance_score': 0.7
                })
        
        return stocks
    
    def _find_concept_stocks(self, parsed_news: Dict) -> List[Dict]:
        """查找概念相关股票（使用 AKShare）"""
        stocks = []
        
        if not self.use_akshare:
            return stocks
        
        try:
            import akshare as ak
            
            # 根据事件类型和关键词查找概念板块
            keywords = parsed_news['key_entities']['products']
            
            # 这里可以扩展更多概念板块的映射逻辑
            # 示例：如果提到"芯片"，查找芯片概念股
            
        except Exception as e:
            logger.warning(f"查找概念股失败: {e}")
        
        return stocks
    
    def _find_region_stocks(self, regions: List[str]) -> List[Dict]:
        """查找地区相关股票"""
        stocks = []
        
        if self.stock_basic_df is None or len(self.stock_basic_df) == 0:
            return stocks
        
        for region in regions:
            # 在地区字段中搜索
            if 'area' in self.stock_basic_df.columns:
                matched = self.stock_basic_df[
                    self.stock_basic_df['area'].str.contains(region, na=False, case=False)
                ]
                
                for _, row in matched.head(15).iterrows():
                    stocks.append({
                        'ts_code': row.get('ts_code', row.get('code', '')),
                        'name': row.get('name', ''),
                        'industry': row.get('industry', '未知'),
                        'relevance_reason': f'地区相关：{region}',
                        'relevance_score': 0.5
                    })
        
        return stocks
    
    def _merge_and_deduplicate(self, *stock_lists) -> List[Dict]:
        """合并并去重股票列表"""
        merged = {}
        
        for stock_list in stock_lists:
            for stock in stock_list:
                ts_code = stock['ts_code']
                
                if ts_code not in merged:
                    merged[ts_code] = stock
                else:
                    # 保留相关性更高的原因
                    if stock['relevance_score'] > merged[ts_code]['relevance_score']:
                        merged[ts_code] = stock
        
        return list(merged.values())
