# coding: utf-8
"""集中配置：环境变量必须在任何 llama_index / HF 导入之前设置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---- 项目根目录 ----
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- 环境变量（必须在 llama_index / HF 相关导入之前）----
# Hugging Face 镜像，加速模型下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# HF 缓存重定向到项目内目录（沙箱禁止写入用户 AppData 目录）
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".cache" / "hf"))
# 禁用 nltk 3.10+ 的导入安全机制
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

# 加载 .env
load_dotenv(BASE_DIR / ".env")

# ---- 路径 ----
DOCS_DIR = BASE_DIR / "docs"
STORAGE_DIR = BASE_DIR / "storage1"          # 索引持久化目录（复用已建好的索引）
MANIFEST_PATH = BASE_DIR / "manifest.json"   # 文件清单（P1 增量对账用）
HF_CACHE_MODELS = BASE_DIR / ".cache" / "hf_models"

# ---- 模型配置 ----
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
LLM_MODEL = "deepseek-v3"
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_MAX_TOKENS = 2048

EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_N = 5
SIMILARITY_TOP_K = 10
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ---- 服务配置（P3/P4 使用）----
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "4"))  # 并发闸：DashScope 并发上限
LLM_QUEUE_TIMEOUT = float(os.getenv("LLM_QUEUE_TIMEOUT", "60"))  # 并发闸排队超时（秒），超时返回 429
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "3600"))  # 查询缓存 TTL（秒）
DRAIN_TIMEOUT = float(os.getenv("DRAIN_TIMEOUT", "15"))  # 优雅关闭：在途请求排空超时（秒）

# 文档支持的扩展名（watchdog 白名单）
SUPPORTED_EXTS = {".txt", ".pdf", ".docx", ".md"}
