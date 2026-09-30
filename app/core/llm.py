# coding: utf-8
"""LLM 与 Embedding 配置（从原脚本 setup_llm_and_embedding 迁移而来）。"""
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app import config


def setup_llm_and_embedding():
    """配置 LLM 与 Embedding。

    LLM 用 DashScope（deepseek-v3）；Embedding 用本地 HuggingFace 模型，
    不消耗 DashScope Embedding 额度。
    """
    if not config.DASHSCOPE_API_KEY:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    llm = DashScope(
        model_name=config.LLM_MODEL,
        api_key=config.DASHSCOPE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        top_p=config.LLM_TOP_P,
        max_tokens=config.LLM_MAX_TOKENS,
    )

    # 本地 Embedding：cache_folder 指向项目内目录，避免写入沙箱受限的用户目录
    embed_model = HuggingFaceEmbedding(
        model_name=config.EMBED_MODEL_NAME,
        cache_folder=str(config.HF_CACHE_MODELS),
    )

    return llm, embed_model
