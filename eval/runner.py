# coding: utf-8
"""检索评测：对比 A/B/C 三种检索配置的质量。

A = 纯向量检索（基线）
B = 向量 + BM25 + RRF 融合（召回层）
C = B + bge-reranker 精排（排序层）

指标：
  Recall@10、HitRate@10 → 对比 A vs B（召回增益）
  MRR、Precision@5       → 对比 B vs C（精排增益）

只跑检索，不调 LLM 合成（快、不消耗 DashScope 配额）。

eval.json 格式（数组，每项）：
  {
    "query": "用户问题",
    "relevant_files": ["相关文件名1.txt", "相关文件2.pdf"]
  }
  relevant_files 用 docs/ 下的 basename，可多个。

用法：
  .venv\\Scripts\\python.exe -m eval.runner
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 环境变量与主服务一致（HF 镜像 + 本地缓存，避免沙箱写入受限目录）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'hf')
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever

from app import config
from app.core.llm import setup_llm_and_embedding
from app.core.indexer import IndexManager

EVAL_PATH = Path(__file__).parent / "eval.json"
TOP_K = config.SIMILARITY_TOP_K
RERANK_TOP_N = config.RERANKER_TOP_N


def _file_of(node):
    """从 node 取所属文件名（basename），匹配 eval.json 的 relevant_files。"""
    fp = (node.metadata or {}).get("file_path") or (node.metadata or {}).get("filename")
    return os.path.basename(fp) if fp else ""


def _score(nodes, relevant_files, k):
    """计算单 query 的 Recall@k / Hit / MRR / Precision@k。"""
    hit_files = [_file_of(n) for n in nodes[:k]]
    hit_relevant = [f for f in hit_files if f in relevant_files]
    # 命中的相关文件数（去重，同文件多 chunk 只算一次）
    hit_rel_set = set(hit_relevant)
    recall = len(hit_rel_set) / len(relevant_files) if relevant_files else 0.0
    hit = 1.0 if hit_relevant else 0.0
    mrr = 0.0
    for i, f in enumerate(hit_files, 1):
        if f in relevant_files:
            mrr = 1.0 / i
            break
    precision = len(hit_relevant) / k if k else 0.0
    return {"recall": recall, "hit": hit, "mrr": mrr, "precision": precision}


def _lift(b, a):
    """相对提升百分比。"""
    return f"+{(b - a) / a * 100:.1f}%" if a > 0 else "N/A"


def _print_report(results):
    n = len(results["A"])
    print(f"\n{'=' * 70}")
    print(f"评测结果（{n} 个 query）")
    print(f"{'=' * 70}")
    print(f"{'配置':<22}{'Recall@10':<13}{'HitRate@10':<13}{'MRR':<10}{'Precision':<10}")
    print("-" * 70)
    means = {}
    for cfg, name in [("A", "纯向量"), ("B", "向量+BM25+RRF"), ("C", "B+reranker")]:
        recall = sum(r["recall"] for r in results[cfg]) / n
        hit = sum(r["hit"] for r in results[cfg]) / n
        mrr = sum(r["mrr"] for r in results[cfg]) / n
        prec = sum(r["precision"] for r in results[cfg]) / n
        means[cfg] = {"recall": recall, "hit": hit, "mrr": mrr, "precision": prec}
        print(f"{name:<22}{recall:<13.3f}{hit:<13.3f}{mrr:<10.3f}{prec:<10.3f}")

    print(f"\n{'提升对比':<22}{'Recall':<13}{'HitRate':<13}{'MRR':<10}{'Precision':<10}")
    print("-" * 70)
    ra, rb = means["A"]["recall"], means["B"]["recall"]
    ha, hb = means["A"]["hit"], means["B"]["hit"]
    mb, mc = means["B"]["mrr"], means["C"]["mrr"]
    pb, pc = means["B"]["precision"], means["C"]["precision"]
    print(f"{'B vs A（召回增益）':<22}{_lift(rb, ra):<13}{_lift(hb, ha):<13}{'-':<10}{'-':<10}")
    print(f"{'C vs B（精排增益）':<22}{'-':<13}{'-':<13}{_lift(mc, mb):<10}{_lift(pc, pb):<10}")


def main():
    print("初始化 LLM 与 Embedding（仅配置，不调用 LLM）...")
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    print("加载索引...")
    index = IndexManager().load_or_build()

    # 构建复用的检索器（一次构建，多 query 复用，不每 query 重建）
    print(f"构建检索器（top_k={TOP_K}, rerank_top_n={RERANK_TOP_N}）...")
    vec_A = index.as_retriever(similarity_top_k=TOP_K)  # A 基线：纯向量

    vec_B = index.as_retriever(similarity_top_k=TOP_K)
    all_nodes = [n for n in index.docstore.docs.values() if isinstance(n, TextNode)]
    bm25_B = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=TOP_K)
    rrf_B = QueryFusionRetriever(retrievers=[vec_B, bm25_B], similarity_top_k=TOP_K)

    reranker = SentenceTransformerRerank(model=config.RERANKER_MODEL, top_n=RERANK_TOP_N)

    # 加载评测集
    with open(EVAL_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    print(f"加载评测集：{len(cases)} 个 query\n")

    results = {"A": [], "B": [], "C": []}
    for i, case in enumerate(cases, 1):
        query = case["query"]
        relevant = set(case["relevant_files"])
        print(f"[{i}/{len(cases)}] {query}")

        nodes_A = vec_A.retrieve(query)
        nodes_B = rrf_B.retrieve(query)
        # C：B 的召回结果过 reranker 精排
        nodes_C = reranker.postprocess_nodes(nodes_B, query_str=query)

        results["A"].append(_score(nodes_A, relevant, TOP_K))
        results["B"].append(_score(nodes_B, relevant, TOP_K))
        results["C"].append(_score(nodes_C, relevant, RERANK_TOP_N))

    _print_report(results)


if __name__ == "__main__":
    main()
