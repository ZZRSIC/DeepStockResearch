"""
Step A: 新闻解析模块
使用 LLM 将新闻解析为结构化 JSON
"""

import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class NewsParser:
    """新闻解析器"""
    
    def __init__(self, api_key: str, api_base: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
    
    def parse_news(self, news_text: str) -> Dict:
        """
        解析新闻为结构化数据
        
        Args:
            news_text: 新闻文本
            
        Returns:
            结构化的新闻数据
        """
        prompt = self._build_parse_prompt(news_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的 A股市场分析师，擅长从新闻中提取结构化信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"新闻解析成功: {result.get('event_type', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"新闻解析失败: {str(e)}")
            return self._get_default_result()
    
    def _build_parse_prompt(self, news_text: str) -> str:
        """构建解析提示词"""
        return f"""请作为专业的 A股产业链分析师，目标是在**保持与新闻强相关**的前提下，**尽可能覆盖所有相关的 A股公司**，并分层标注相关强度。

新闻内容：
{news_text}

请按照以下结构输出（必须是有效的 JSON 格式）：
{{
    "event_type": "事件类型（从以下选择：政策发布/业绩公告/重大事故/并购重组/供给冲击/科技突破/行业变革/监管变化/其他）",
    "event_summary": "事件摘要（1-2句话）",
    "key_entities": {{
        "companies": ["新闻中**直接提及**的公司名称（仅限直接提及）"],
        "products": ["相关产品或服务"],
        "industries": ["相关行业"],
        "regions": ["相关地区"],
        "regulators": ["相关监管部门"]
    }},
    "industry_chain_analysis": {{
        "upstream_suppliers": [
            {{
                "segment": "上游环节名称（如：原材料/核心零部件/设备/软件平台等）",
                "description": "该环节的详细描述",
                "companies": ["该环节尽可能多的 A股公司简称"],
                "impact": "正面/负面/中性",
                "reasoning": "为何受影响的推理"
            }}
        ],
        "midstream_manufacturers": [
            {{
                "segment": "中游环节名称（如：组装/制造/系统集成等）",
                "description": "该环节的详细描述",
                "companies": ["该环节尽可能多的 A股公司简称"],
                "impact": "正面/负面/中性",
                "reasoning": "为何受影响的推理"
            }}
        ],
        "downstream_applications": [
            {{
                "segment": "下游环节名称（如：终端应用/销售渠道/运营服务等）",
                "description": "该环节的详细描述",
                "companies": ["该环节尽可能多的 A股公司简称"],
                "impact": "正面/负面/中性",
                "reasoning": "为何受影响的推理"
            }}
        ],
        "competitors": [
            {{
                "company": "竞争对手公司简称（A股为主）",
                "relationship": "直接竞争/替代品竞争/潜在竞争",
                "impact": "正面/负面/中性",
                "reasoning": "竞争关系分析"
            }}
        ],
        "beneficiaries": [
            {{
                "company": "间接受益公司简称（A股为主）",
                "reason": "受益原因（如：技术溢出/市场扩大/成本下降/政策执行/合规需求等）",
                "confidence": "高/中/低"
            }}
        ]
    }},
    "impact_paths": [
        {{
            "path": "影响路径描述（如：需求增加/成本下降/供给受限/合规风险/竞争格局变化等）",
            "direction": "正面/负面/中性",
            "confidence": "高/中/低",
            "reasoning": "影响路径的推理依据"
        }}
    ],
    "time_frame": "影响时间框架（短期/中期/长期）",
    "affected_segments": ["受影响的产业链环节（上游原材料/中游制造/下游应用等）"],
    "company_candidates": [
        {{
            "company_name": "A股公司简称（优先输出 A股，避免只给海外公司名）",
            "relationship": "直接提及/上游供应/中游制造/下游应用/渠道/竞争/替代/配套服务/政策执行/合规/基础设施",
            "segment": "上游/中游/下游/配套/其他",
            "impact": "正面/负面/中性",
            "confidence": "高/中/低",
            "reasoning": "10-20字解释",
            "logic_chain": "新闻→路径→公司"
        }}
    ],
    "search_keywords": ["同义词/技术名/产品名/政策名/材料/设备/应用场景/简称别名等"]
}}

深度思考要求：
1. **覆盖优先但需可解释**：每家公司必须有清晰的关系与路径，逻辑链不超过 3 步；不确定的放入低置信度。
2. 对于**直接提及的公司**，除了写入 key_entities.companies，还要扩展其：
   - 上游供应商、核心材料/设备/软件平台
   - 下游客户/应用场景/运营服务
   - 直接竞争对手/替代方案
   - 政策执行/合规/基础设施服务商
3. **尽量输出 A股公司简称**。若新闻主体为非 A股（如海外公司/机构），请给出 A股对标或供应链公司。
4. industry_chain_analysis 的 companies 字段尽可能多列出 A股公司；若缺乏确定性，可在 company_candidates 中标注为低置信度。
5. search_keywords 用于后续概念/行业检索，请包含产品别名、核心技术名、材料/设备关键词、政策或监管术语。

请严格按照此格式输出，确保是有效的 JSON。"""
    
    def _get_default_result(self) -> Dict:
        """返回默认结果"""
        return {
            "event_type": "其他",
            "event_summary": "解析失败",
            "key_entities": {
                "companies": [],
                "products": [],
                "industries": [],
                "regions": [],
                "regulators": []
            },
            "industry_chain_analysis": {
                "upstream_suppliers": [],
                "midstream_manufacturers": [],
                "downstream_applications": [],
                "competitors": [],
                "beneficiaries": []
            },
            "impact_paths": [],
            "time_frame": "未知",
            "affected_segments": [],
            "company_candidates": [],
            "search_keywords": []
        }
