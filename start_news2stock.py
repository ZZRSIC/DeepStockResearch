import sys
import os
import types
import importlib.util
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# 创建FastAPI应用实例
app = FastAPI(
    title="实时新闻处理API",
    description="处理新闻内容并返回分析结果",
    version="1.0.0"
)

# 定义请求体模型
class NewsRequest(BaseModel):
    news: str
    news_id: str
    created_at: Optional[datetime] = None


_news2stock_pipeline = None


def _bootstrap_config_module() -> None:
    if "config" in sys.modules:
        return
    try:
        import config  # noqa: F401
        return
    except Exception:
        pass

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    config_module = types.ModuleType("config")

    class Config:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
        USE_TAVILY = os.getenv("USE_TAVILY", "true").lower() in ("1", "true", "yes")
        TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
        USE_TUSHARE = os.getenv("USE_TUSHARE", "true").lower() in ("1", "true", "yes")
        USE_AKSHARE = os.getenv("USE_AKSHARE", "true").lower() in ("1", "true", "yes")
        USE_BAOSTOCK = os.getenv("USE_BAOSTOCK", "true").lower() in ("1", "true", "yes")
        MAX_RESEARCH_ROUNDS = int(os.getenv("MAX_RESEARCH_ROUNDS", "3"))
        TOP_K_STOCKS = int(os.getenv("TOP_K_STOCKS", "10"))
        SCORE_WEIGHTS = {
            "relevance": 0.3,
            "fundamental": 0.25,
            "valuation": 0.2,
            "momentum": 0.15,
            "risk": 0.1,
        }
        ENABLE_CACHE = True
        CACHE_DIR = "./cache"
        LOG_LEVEL = "INFO"
        LOG_FILE = "./deepresearch.log"

    config_module.Config = Config
    sys.modules["config"] = config_module


def _load_news2stock_pipeline():
    global _news2stock_pipeline
    if _news2stock_pipeline is not None:
        return _news2stock_pipeline

    base_dir = os.path.dirname(os.path.abspath(__file__))
    news2stock_dir = os.path.join(base_dir, "news2stock")
    if not os.path.isdir(news2stock_dir):
        raise RuntimeError("未找到 news2stock 目录")

    if news2stock_dir not in sys.path:
        sys.path.insert(0, news2stock_dir)

    _bootstrap_config_module()

    pipeline_path = os.path.join(news2stock_dir, "pipeline.py")
    spec = importlib.util.spec_from_file_location("news2stock_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 news2stock/pipeline.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _news2stock_pipeline = module.AStockDeepResearchPipeline()
    return _news2stock_pipeline


@app.post("/process_news")
async def process_news(request: NewsRequest):
    """处理新闻内容的API端点"""
    try:
        pipeline = _load_news2stock_pipeline()
        result = pipeline.run(request.news, save_report=False)
        return {
            "success": True,
            "news_id": request.news_id,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "news_id": request.news_id,
            "result": {"error": str(e)}
        }


if __name__ == "__main__":
    uvicorn.run(
        app="start_news2stock:app",
        host="0.0.0.0",
        port=8283,
        reload=False
    )
