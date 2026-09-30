# coding: utf-8
"""检索与查询引擎组装：向量检索 + 重排序 + refine 合成。

提示词模板来自 security/prompts.py（隔离版，P2）。
"""
import logging

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from app import config
from app.security.prompts import TEXT_QA_TEMPLATE, REFINE_TEMPLATE

logger = logging.getLogger("insurance-qa.retriever")

# 重排序器单例：模型权重只加载一次，build_query_engine 重建时复用，避免 ~8s 重载
_reranker: SentenceTransformerRerank | None = None


def _get_reranker() -> SentenceTransformerRerank:
    """懒加载重排序器单例。首次调用加载模型权重，后续重建 query_engine 时复用。"""
    global _reranker
    if _reranker is None:
        _reranker = SentenceTransformerRerank(
            model=config.RERANKER_MODEL,
            top_n=config.RERANKER_TOP_N,
        )
    return _reranker


def build_query_engine(index):
    """构建增强版查询引擎：向量+BM25 混合召回 + 重排序 + refine 合成。"""
    # 向量检索（语义召回）
    vector_retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)

    # BM25 关键词检索（从 docstore 取当前全部 chunk，保证与索引同步）
    all_nodes = [n for n in index.docstore.docs.values() if isinstance(n, TextNode)]
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=config.SIMILARITY_TOP_K,
    )

    # RRF 融合（向量 + BM25，无额外 LLM 调用）
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=config.SIMILARITY_TOP_K,
    )

    postprocessors = []
    try:
        postprocessors.append(_get_reranker())
    except Exception as e:
        # 重排序器不可用时不影响主流程，降级为无重排序并留痕
        logger.warning("重排序器不可用，已降级为无重排序: %s", e)
        postprocessors = []

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=TEXT_QA_TEMPLATE,
        refine_template=REFINE_TEMPLATE,
    )

    return RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )
