# News2Stock 接口说明

本项目提供一个简化的新闻→股票关联分析接口（`/process_news`），用于将新闻输入并返回相关股票结果。  
当前对外只保留 `start_news2stock.py` 入口，便于拷贝到其他工程直接运行。

## 环境要求

- Python 3.10+（推荐 3.10/3.11）
- 已配置 OpenAI API Key

## 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 环境变量

必须：

```bash
export OPENAI_API_KEY="your_api_key"
```

可选：

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export TUSHARE_TOKEN="your_tushare_token"
```

## 启动服务

```bash
python start_news2stock.py
```

或：

```bash
uvicorn start_news2stock:app --host 0.0.0.0 --port 8283
```

## 接口说明

### POST /process_news

**请求体（JSON）：**

```json
{
  "news": "比亚迪发布新一代刀片电池技术，能量密度提升30%，成本下降20%。",
  "news_id": "news_001",
  "created_at": "2026-02-02T10:00:00"
}
```

**响应（JSON）：**

```json
{
  "success": true,
  "news_id": "news_001",
  "result": {
    "metadata": {
      "generated_at": "2026-02-02T17:43:11.229951",
      "news_summary": "比亚迪发布新一代刀片电池技术，能量密度提升30%，成本下降20%。",
      "event_type": "科技突破",
      "total_candidates": 8,
      "top_k": 10
    },
    "top_stocks": [
      {
        "排名": 1,
        "股票代码": "002594.SZ",
        "股票名称": "比亚迪",
        "所属行业": "汽车整车"
      },
      {
        "排名": 2,
        "股票代码": "300750.SZ",
        "股票名称": "宁德时代",
        "所属行业": "电气设备"
      }
    ]
  }
}
```

## 迁移到其他工程

若需要拷贝到 `/Users/sic/code/realtime-news`，仅需复制：

- `start_news2stock.py`
- `news2stock/`

并确保目标工程已配置依赖与环境变量。
