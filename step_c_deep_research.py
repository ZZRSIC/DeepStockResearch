"""
Step C: DeepResearch 多轮检索模块
通过多轮提问、检索、补充证据的方式深度研究
"""

import logging
from typing import Dict, List, Optional
from openai import OpenAI
import json

logger = logging.getLogger(__name__)


class DeepResearcher:
    """深度研究器"""
    
    def __init__(self, api_key: str, api_base: str, model: str,
                 max_rounds: int = 3, ts_pro=None, tavily_api_key: str = None,
                 tavily_max_calls: int = 8):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_rounds = max_rounds
        self.ts_pro = ts_pro
        self.tavily_max_calls = tavily_max_calls
        
        # 初始化 Tavily
        self.tavily_client = None
        self.tavily_call_count = 0  # 添加 Tavily 调用计数器
        self._tavily_cache = {}
        self._tavily_question_done = set()
        self._tavily_budget = 0
        self._tavily_event_evidence = []
        self._tavily_event_included = False
        if tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                logger.info("Tavily 网络检索初始化成功")
            except Exception as e:
                logger.warning(f"Tavily 初始化失败: {e}")
    
    def conduct_research(self, parsed_news: Dict, candidate_stocks: List[Dict]) -> Dict:
        """
        对候选股票进行深度研究
        
        Args:
            parsed_news: 解析后的新闻
            candidate_stocks: 候选股票列表
            
        Returns:
            研究结果
        """
        # 重置调用计数器
        self.tavily_call_count = 0
        self._tavily_cache = {}
        self._tavily_question_done = set()
        self._tavily_event_evidence = []
        self._tavily_event_included = False
        
        research_results = {
            "questions": [],
            "evidence": [],
            "stock_analysis": {},
            "tavily_call_count": 0  # 添加到结果中
        }
        
        # 第一轮：生成研究问题
        questions = self._generate_research_questions(parsed_news, candidate_stocks)
        research_results["questions"] = questions
        self._tavily_budget = self._compute_tavily_budget(questions)
        self._tavily_event_evidence = self._prefetch_event_evidence(parsed_news, candidate_stocks)
        
        # 多轮研究循环
        for round_num in range(self.max_rounds):
            logger.info(f"开始第 {round_num + 1} 轮研究")
            
            # 为每个问题收集证据
            for question in questions:
                evidence = self._collect_evidence(question, parsed_news, candidate_stocks)
                research_results["evidence"].extend(evidence)
            
            # 如果已经收集到足够证据，可以提前结束
            if len(research_results["evidence"]) > 50:
                break
        
        # 对每只股票进行分析
        for stock in candidate_stocks[:30]:  # 限制分析的股票数量
            analysis = self._analyze_stock(
                stock, parsed_news, research_results["evidence"]
            )
            research_results["stock_analysis"][stock['ts_code']] = analysis
        
        # 保存 Tavily 调用次数到结果中
        research_results["tavily_call_count"] = self.tavily_call_count
        logger.info(f"Tavily 总调用次数: {self.tavily_call_count}")
        
        return research_results
    
    def _generate_research_questions(self, parsed_news: Dict, 
                                    candidate_stocks: List[Dict]) -> List[str]:
        """生成研究问题"""
        
        prompt = f"""基于以下新闻事件和候选股票，生成 5-7 个高质量的深度研究问题。

【新闻事件】
事件摘要：{parsed_news['event_summary']}
事件类型：{parsed_news['event_type']}
影响路径：{json.dumps(parsed_news['impact_paths'], ensure_ascii=False)}

【候选股票】
{', '.join([s['name'] for s in candidate_stocks[:10]])}

【研究问题生成指南】
请围绕"找出真正受益的投资标的"这一核心目标，生成具有针对性的研究问题。

问题应该聚焦于：
1. **因果关系验证**：事件如何传导到具体公司的业务和财务？
2. **产业链定位**：哪些环节是关键受益点？
3. **竞争格局分析**：谁是真正的龙头和受益者？
4. **证据支持**：有什么数据、公告、研报支持这个逻辑？
5. **风险识别**：主要的不确定性和风险在哪里？

【优质问题示例】
针对"育儿补贴发放"新闻：
✅ 好问题："育儿补贴的发放渠道是什么？需要哪些技术系统支持？哪些公司提供相关服务？"
✅ 好问题："历史上类似的财政补贴政策实施时，哪些公司获得了实质性订单增长？"
✅ 好问题："补贴资金规模有多大？对相关公司的业绩贡献能达到什么量级？"

❌ 差问题："补贴会增加消费吗？"（太宽泛，传导路径不清晰）
❌ 差问题："哪些公司受益？"（太笼统，没有研究价值）

【输出格式】
**重要：必须严格按照以下 JSON 格式返回，使用 "questions" 作为键名：**
{{
    "questions": [
        "问题1：具体、可验证、有针对性的研究问题",
        "问题2：...",
        "问题3：...",
        "问题4：...",
        "问题5：...",
        "问题6：...",
        "问题7：..."
    ]
}}

请生成 5-7 个高质量研究问题。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 A股研究分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 容错处理：支持多种 JSON 格式
            questions = []
            if isinstance(result, list):
                # 格式1: 直接返回数组 ["问题1", "问题2", ...]
                questions = result
            elif isinstance(result, dict):
                # 格式2: 对象格式，尝试多个可能的键名
                questions = (result.get("questions") or
                           result.get("问题列表") or
                           result.get("items") or
                           result.get("question_list") or
                           list(result.values())[0] if result.values() else [])
            
            # 确保 questions 是列表
            if not isinstance(questions, list):
                questions = []
            
            logger.info(f"生成了 {len(questions)} 个研究问题")
            
            # 如果仍然没有问题，记录原始响应以便调试
            if len(questions) == 0:
                logger.warning(f"LLM 返回了空问题列表，原始响应: {response.choices[0].message.content[:200]}")
            
            return questions
            
        except Exception as e:
            logger.error(f"生成研究问题失败: {e}")
            # 返回默认问题
            return [
                "新闻中提及的公司有哪些是 A股上市公司？",
                "受影响的产业链环节有哪些？",
                "相关行业的龙头公司是谁？",
                "历史上类似事件的市场反应如何？",
                "主要风险点是什么？"
            ]
    
    def _collect_evidence(self, question: str, parsed_news: Dict,
                         candidate_stocks: List[Dict]) -> List[Dict]:
        """为问题收集证据"""
        evidence_list = []
        
        # 0. 事件级通用证据（仅追加一次，避免重复）
        if self._tavily_event_evidence and not self._tavily_event_included:
            evidence_list.extend(self._tavily_event_evidence)
            self._tavily_event_included = True

        # 1. 使用 Tavily 网络检索（优先级最高）
        if self._should_use_tavily(question, parsed_news, candidate_stocks):
            tavily_evidence = self._get_tavily_evidence(question, parsed_news, candidate_stocks)
            evidence_list.extend(tavily_evidence)
        
        # 2. 从 Tushare 获取相关新闻
        news_evidence = self._get_news_evidence(question, candidate_stocks)
        evidence_list.extend(news_evidence)
        
        # 3. 从公告中获取证据（如果有 API）
        # announcement_evidence = self._get_announcement_evidence(question, candidate_stocks)
        # evidence_list.extend(announcement_evidence)
        
        # 4. 使用 LLM 推理证据
        reasoning_evidence = self._get_reasoning_evidence(question, parsed_news)
        evidence_list.extend(reasoning_evidence)
        
        return evidence_list
    
    def _get_tavily_evidence(self, question: str, parsed_news: Dict,
                            candidate_stocks: List[Dict]) -> List[Dict]:
        """使用 Tavily 进行网络检索获取证据"""
        evidence = []
        
        if not self.tavily_client:
            return evidence
        
        try:
            if self.tavily_call_count >= self._tavily_budget:
                return evidence

            # 智能选择相关股票
            relevant_stocks = self._select_relevant_stocks_for_question(
                question, candidate_stocks, parsed_news
            )
            
            # 构建多个搜索查询策略
            search_queries = self._build_search_queries(
                question, parsed_news, relevant_stocks
            )
            
            # 对每个查询进行搜索（优先 1 次，结果不足再追加）
            for idx, search_query in enumerate(search_queries[:2]):  # 最多2个查询
                if self.tavily_call_count >= self._tavily_budget:
                    break

                logger.info(f"Tavily 搜索 {idx+1}/{len(search_queries[:2])}: {search_query}")
                results = self._tavily_search(search_query)

                for result in results:
                    evidence.append({
                        "type": "web_search",
                        "source": "Tavily",
                        "title": result.get('title', ''),
                        "content": result.get('content', ''),
                        "url": result.get('url', ''),
                        "score": result.get('score', 0),
                        "published_date": result.get('published_date', ''),
                        "query_strategy": idx + 1,  # 记录使用的查询策略
                        "relevance": "high" if result.get('score', 0) > 0.8 else "medium"
                    })

                # 如果结果质量足够，停止追加查询
                if self._is_good_results(results):
                    break
            
            logger.info(f"Tavily 返回 {len(evidence)} 条网络证据")
            self._tavily_question_done.add(question)
            
        except Exception as e:
            logger.warning(f"Tavily 检索失败: {e}")
        
        return evidence

    def _tavily_search(self, query: str) -> List[Dict]:
        """带缓存的 Tavily 搜索，减少重复调用"""
        if not self.tavily_client or not query:
            return []

        normalized = self._normalize_query(query)
        if normalized in self._tavily_cache:
            return self._tavily_cache[normalized]

        if self.tavily_call_count >= self._tavily_budget:
            return []

        self.tavily_call_count += 1  # 增加调用计数
        response = self.tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=4,  # 单次多拿一点结果，减少调用次数
            include_domains=[
                "sina.com.cn",
                "eastmoney.com",
                "10jqka.com.cn",
                "cnstock.com",
                "stcn.com",
                "jrj.com.cn",
                "cs.com.cn",
                "finance.sina.com.cn",
                "stock.stockstar.com"
            ],
            exclude_domains=["youtube.com", "twitter.com"]
        )

        results = response.get('results', []) if response else []
        self._tavily_cache[normalized] = results
        return results

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.lower().split())

    def _is_good_results(self, results: List[Dict]) -> bool:
        if not results:
            return False
        if any(r.get('score', 0) >= 0.78 for r in results):
            return True
        return len(results) >= 3

    def _compute_tavily_budget(self, questions: List[str]) -> int:
        """根据问题数量动态分配 Tavily 调用预算"""
        if not self.tavily_client:
            return 0
        base = max(3, len(questions))
        return min(self.tavily_max_calls, base)

    def _prefetch_event_evidence(self, parsed_news: Dict,
                                 candidate_stocks: List[Dict]) -> List[Dict]:
        """事件级一次性检索，作为通用证据"""
        if not self.tavily_client:
            return []
        if self.tavily_call_count >= self._tavily_budget:
            return []

        event_summary = parsed_news.get('event_summary', '')
        key_entities = parsed_news.get('key_entities', {})
        companies = key_entities.get('companies', [])
        products = key_entities.get('products', [])
        industries = key_entities.get('industries', [])
        regulators = key_entities.get('regulators', [])
        keywords = parsed_news.get('search_keywords', [])

        terms = [event_summary]
        terms.extend(companies[:2])
        terms.extend(products[:2])
        terms.extend(industries[:2])
        terms.extend(regulators[:1])
        terms.extend(keywords[:2])
        query = " ".join([t for t in terms if t])

        if not query.strip():
            return []

        results = self._tavily_search(query)
        evidence = []
        for result in results:
            evidence.append({
                "type": "web_search",
                "source": "Tavily",
                "title": result.get('title', ''),
                "content": result.get('content', ''),
                "url": result.get('url', ''),
                "score": result.get('score', 0),
                "published_date": result.get('published_date', ''),
                "query_strategy": 0,
                "relevance": "high" if result.get('score', 0) > 0.8 else "medium"
            })
        return evidence

    def _should_use_tavily(self, question: str, parsed_news: Dict,
                           candidate_stocks: List[Dict]) -> bool:
        """判断是否需要调用 Tavily，控制调用次数但不牺牲准确度"""
        if not self.tavily_client:
            return False
        if self.tavily_call_count >= self._tavily_budget:
            return False
        if question in self._tavily_question_done:
            return False

        # 需要外部证据的关键词
        must_keywords = [
            "公告", "中标", "订单", "签约", "销量", "营收", "财报", "业绩",
            "政策", "监管", "标准", "补贴", "价格", "产能", "产量",
            "招标", "核准", "项目", "扩产", "合作", "投资", "并购",
            "停产", "事故", "处罚", "整改", "复工"
        ]
        if any(kw in question for kw in must_keywords):
            return True

        # 数字类问题通常需要外部证据
        if any(ch.isdigit() for ch in question):
            return True

        # 如果问题包含具体公司名，倾向使用 Tavily 验证
        stock_names = [s.get('name', '') for s in candidate_stocks[:8] if s.get('name')]
        key_entities = parsed_news.get('key_entities', {})
        entity_names = []
        for entities in key_entities.values():
            entity_names.extend(entities)
        name_pool = set(stock_names + entity_names)
        if any(name and name in question for name in name_pool):
            return True

        # 若已有事件级通用证据，可跳过低优先级问题
        return False
    
    def _select_relevant_stocks_for_question(self, question: str,
                                            candidate_stocks: List[Dict],
                                            parsed_news: Dict) -> List[Dict]:
        """根据问题智能选择相关股票"""
        # 问题关键词分类
        question_lower = question.lower()
        
        # 龙头/竞争格局类问题 - 选择排名靠前的股票
        if any(kw in question_lower for kw in ['龙头', '竞争', '格局', '领先', '头部']):
            return candidate_stocks[:5]
        
        # 产业链/上下游类问题 - 选择不同环节的股票
        elif any(kw in question_lower for kw in ['产业链', '上游', '下游', '中游', '供应链']):
            # 尝试从不同行业选择
            selected = []
            seen_industries = set()
            for stock in candidate_stocks[:15]:
                if stock.get('industry') not in seen_industries:
                    selected.append(stock)
                    seen_industries.add(stock.get('industry'))
                if len(selected) >= 5:
                    break
            return selected if selected else candidate_stocks[:5]
        
        # 低估/潜力类问题 - 选择中后段的股票
        elif any(kw in question_lower for kw in ['低估', '潜力', '被忽视', '黑马']):
            return candidate_stocks[5:15] if len(candidate_stocks) > 5 else candidate_stocks
        
        # 公司/实体类问题 - 从新闻关键实体中提取相关股票
        elif any(kw in question_lower for kw in ['公司', '企业', '实体', '上市']):
            key_entities = parsed_news.get('key_entities', {})
            entity_names = []
            for entity_type, entities in key_entities.items():
                entity_names.extend(entities)
            
            # 匹配包含关键实体的股票
            matched_stocks = []
            for stock in candidate_stocks[:20]:
                if any(entity in stock['name'] for entity in entity_names):
                    matched_stocks.append(stock)
            
            return matched_stocks[:8] if matched_stocks else candidate_stocks[:5]
        
        # 默认：返回前8只股票
        else:
            return candidate_stocks[:8]
    
    def _build_search_queries(self, question: str, parsed_news: Dict,
                             relevant_stocks: List[Dict]) -> List[str]:
        """构建多个搜索查询策略"""
        queries = []
        event_summary = parsed_news['event_summary']
        stock_names = [s['name'] for s in relevant_stocks]
        search_keywords = parsed_news.get('search_keywords', [])
        key_entities = parsed_news.get('key_entities', {})
        products = key_entities.get('products', [])
        industries = key_entities.get('industries', [])
        companies = key_entities.get('companies', [])
        
        # 策略1: 问题 + 关键词（不带股票名）
        terms = [question]
        if search_keywords:
            terms.extend(search_keywords[:2])
        elif companies:
            terms.extend(companies[:2])
        elif products:
            terms.extend(products[:2])
        elif event_summary:
            terms.append(event_summary)
        query1 = " ".join([t for t in terms if t])
        if query1.strip():
            queries.append(query1)
        
        # 策略2: 问题 + 行业（不带股票名）
        if len(relevant_stocks) > 0 or industries:
            # 提取行业信息
            industries = list(set([s.get('industry', '') for s in relevant_stocks[:3] if s.get('industry')])) or industries
            industry_str = ' '.join(industries[:2]) if industries else ''
            query2 = f"{question} {industry_str}".strip()
            if query2:
                queries.append(query2)
        
        # 策略3: 事件摘要 + 关键词补充（仅在前两条为空时使用）
        if len(queries) == 0 and event_summary:
            extra_terms = [event_summary]
            extra_terms.extend(search_keywords[:2])
            extra_terms.extend(products[:2])
            query3 = " ".join([t for t in extra_terms if t])
            if query3:
                queries.append(query3)
        
        # 去重并返回
        return list(dict.fromkeys(queries))  # 保持顺序的去重
    
    def _get_news_evidence(self, question: str, candidate_stocks: List[Dict]) -> List[Dict]:
        """从新闻中获取证据"""
        evidence = []
        
        if not self.ts_pro:
            return evidence
        
        try:
            # 获取最近的新闻（Tushare pro.news 接口）
            # 注意：这个接口可能需要较高权限
            for stock in candidate_stocks[:5]:  # 限制查询数量
                try:
                    news_df = self.ts_pro.news(
                        src='sina',
                        start_date='20240101',
                        end_date='20241231'
                    )
                    
                    # 过滤包含股票名称的新闻
                    if news_df is not None and len(news_df) > 0:
                        for _, row in news_df.head(3).iterrows():
                            evidence.append({
                                "type": "news",
                                "source": "新浪财经",
                                "content": row.get('content', row.get('title', '')),
                                "date": row.get('datetime', ''),
                                "relevance": "medium"
                            })
                except Exception as e:
                    logger.debug(f"获取 {stock['name']} 新闻失败: {e}")
                    
        except Exception as e:
            logger.warning(f"获取新闻证据失败: {e}")
        
        return evidence
    
    def _get_reasoning_evidence(self, question: str, parsed_news: Dict) -> List[Dict]:
        """通过 LLM 推理获取证据"""
        prompt = f"""基于以下新闻信息，回答问题并提供推理依据。

新闻摘要：{parsed_news['event_summary']}
事件类型：{parsed_news['event_type']}
关键实体：{json.dumps(parsed_news['key_entities'], ensure_ascii=False)}
影响路径：{json.dumps(parsed_news['impact_paths'], ensure_ascii=False)}

问题：{question}

请提供：
1. 直接回答
2. 推理依据
3. 可信度评估（高/中/低）

以 JSON 格式返回。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 A股市场分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return [{
                "type": "reasoning",
                "question": question,
                "answer": result.get("answer", ""),
                "reasoning": result.get("reasoning", ""),
                "confidence": result.get("confidence", "medium"),
                "source": "LLM推理"
            }]
            
        except Exception as e:
            logger.error(f"推理证据生成失败: {e}")
            return []
    
    def _analyze_stock(self, stock: Dict, parsed_news: Dict,
                      evidence: List[Dict]) -> Dict:
        """分析单只股票的相关性（不做全方位打分）"""
        
        # 收集相关证据
        relevant_evidence = [
            e for e in evidence
            if stock['name'] in str(e) or stock['ts_code'] in str(e)
        ]
        
        prompt = f"""基于新闻和证据，分析股票与新闻事件的相关性。

股票：{stock['name']} ({stock['ts_code']})
行业：{stock['industry']}
初步相关性：{stock['relevance_reason']}

新闻事件：{parsed_news['event_summary']}
事件类型：{parsed_news['event_type']}
证据数量：{len(relevant_evidence)}

请专注于相关性分析，回答以下问题：
1. 该股票与新闻事件的关联路径是什么？（直接相关/产业链相关/间接相关）
2. 关联的具体原因和逻辑链条
3. 相关性强度评估（强/中/弱）
4. 支持该相关性判断的关键证据摘要

注意：只需分析相关性，不需要做投资价值评分、买卖建议等综合评估。

以 JSON 格式返回，包含字段：
- relation_path: 关联路径
- relation_reason: 关联原因
- relation_strength: 相关性强度（强/中/弱）
- key_evidence: 关键证据列表
- evidence_count: 证据数量
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的股票关联性分析师，专注于分析股票与新闻事件的相关性，不做投资评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            analysis["evidence_count"] = len(relevant_evidence)
            
            # 确保返回结构完整
            return {
                "relation_path": analysis.get("relation_path", "未知"),
                "relation_reason": analysis.get("relation_reason", stock['relevance_reason']),
                "relation_strength": analysis.get("relation_strength", "中"),
                "key_evidence": analysis.get("key_evidence", []),
                "evidence_count": len(relevant_evidence)
            }
            
        except Exception as e:
            logger.error(f"股票相关性分析失败: {e}")
            return {
                "relation_path": "分析失败",
                "relation_reason": stock['relevance_reason'],
                "relation_strength": "未知",
                "key_evidence": [],
                "evidence_count": len(relevant_evidence)
            }
