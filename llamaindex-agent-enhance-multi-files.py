#!/usr/bin/env python
# coding: utf-8

import os
# 设置 Hugging Face 镜像（必须在其他导入之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 将 HF 缓存重定向到项目内目录（沙箱禁止写入用户 AppData 目录）
os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'hf')
# 禁用 nltk 3.10+ 的导入安全机制
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
import asyncio
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer


# ==================== 步骤 1：配置 LLM 和 Embedding ====================
def setup_llm_and_embedding():
    """配置 LLM 和 Embedding，使用 DashScope"""
    api_key = os.getenv('DASHSCOPE_API_KEY')

    if not api_key:
        raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

    # 使用 DashScope LLM
    llm = DashScope(
        model_name="deepseek-v3",
        api_key=api_key,
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,  # 增加输出长度
    )

    # 使用本地 HuggingFace Embedding（不消耗 DashScope 额度）
    # BAAI/bge-small-zh-v1.5：中文检索效果好，体积小，与重排序器同源
    # cache_folder 指向项目内目录，避免写入沙箱受限的用户目录
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'hf_models'),
    )

    return llm, embed_model


# ==================== 步骤 2：自定义提示词模板 ====================
def get_custom_prompts():
    """创建自定义的提示词模板，提升回答质量"""

    # 文本问答提示词（用于 query_engine）
    text_qa_template = PromptTemplate(
        """你是一个专业的AI助手，专门回答关于保险产品的问题。

背景信息：
---------------------
{context_str}
---------------------

基于以上背景信息，请回答以下问题。

要求：
1. 如果背景信息中有相关内容，请基于这些信息给出准确、详细的回答
2. 如果背景信息不足以回答问题，请明确说明"根据提供的文档，我无法完整回答这个问题"
3. 回答要结构清晰，重点突出
4. 不要编造或猜测文档中没有的信息

问题：{query_str}

请用中文回答："""
    )

    # 精炼提示词（用于 refine 模式）
    refine_template = PromptTemplate(
        """你是一个专业的AI助手，正在优化之前的回答。

原始回答：
{existing_answer}

新的背景信息：
{context_msg}

请根据新的背景信息，优化和完善之前的回答。如果新信息没有改变原始回答，请保持原样。
如果发现需要补充或修正的内容，请更新回答。

问题：{query_str}

优化后的回答："""
    )

    return text_qa_template, refine_template


# ==================== 步骤 3：加载文档并创建索引 ====================
def load_documents_and_create_index(file_dir: str = './docs'):
    """加载文档文件夹中的所有文件并创建向量索引（增强版）"""
    persist_dir = "./storage1"

    # 检查索引是否已存在
    if os.path.exists(persist_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            print("✅ 从存储加载索引成功")
            return index
        except Exception as e:
            print(f"⚠️ 加载索引失败: {e}，将重新创建索引")

    # 如果索引不存在，创建新索引
    if not os.path.exists(file_dir):
        print(f"❌ 文档目录 {file_dir} 不存在")
        return None

    # 读取文档
    reader = SimpleDirectoryReader(file_dir)
    documents = reader.load_data()

    if not documents:
        print("❌ 没有找到任何文档")
        return None

    print(f"📚 加载了 {len(documents)} 个文档")

    # 使用更好的文本分割器
    text_splitter = SentenceSplitter(
        chunk_size=512,  # 更小的块，提高检索精度
        chunk_overlap=50,  # 适度重叠
        separator=" ",  # 按空格分割
    )

    # 创建向量索引
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        show_progress=True,
    )

    # 保存索引
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"💾 索引已保存到 {persist_dir}")

    return index


# ==================== 步骤 4：创建增强版查询引擎 ====================
def create_enhanced_query_engine(index, llm):
    """创建增强版的查询引擎，包含重排序和更好的合成策略"""

    # 获取自定义提示词
    text_qa_template, refine_template = get_custom_prompts()

    # 1. 创建检索器
    retriever = index.as_retriever(
        similarity_top_k=10,
    )

    # 2. 创建重排序器
    try:
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=5,
        )
        print("✅ 重排序器加载成功")
        postprocessors = [reranker]
    except Exception as e:
        print(f"⚠️ 重排序器加载失败: {e}，将不使用重排序")
        postprocessors = []

    # 3. 创建响应合成器（使用 refine 模式）
    response_synthesizer = get_response_synthesizer(
        response_mode="refine",
        text_qa_template=text_qa_template,
        refine_template=refine_template,
    )

    # 4. 直接构建 RetrieverQueryEngine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )

    return query_engine


# ==================== 步骤 5：创建混合检索器（可选） ====================
def create_hybrid_retriever(index, documents):
    """创建混合检索器，结合向量检索和BM25关键词检索"""

    # 向量检索器
    vector_retriever = index.as_retriever(
        similarity_top_k=10,
    )

    # BM25 关键词检索器（需要原始文档）
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=documents,
        similarity_top_k=10,
    )

    # 路由检索器（可以同时使用两种检索方式）
    hybrid_retriever = RouterRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        # 可以配置选择策略
    )

    return hybrid_retriever


# ==================== 步骤 6：创建智能体 ====================
def create_agent(index, llm, documents=None):
    """创建增强版 ReAct 智能体"""

    # 创建增强版查询引擎
    query_engine = create_enhanced_query_engine(index, llm)

    # 定义系统提示词（更详细的指令）
    system_instruction = '''你是一个专业的保险知识助手，擅长回答关于雇主责任险等保险产品的问题。

你的工作方式：
1. 当用户提出问题时，你会从文档中检索相关信息
2. 基于检索到的信息，给出准确、详细的回答
3. 如果文档信息不足，会明确告知用户
4. 回答要结构清晰，使用要点或分段方式

注意：
- 只使用文档中的信息，不要编造
- 用中文回复
- 保持专业、友好的语气'''

    # 创建检索工具
    def retrieve_documents(query: str) -> str:
        """从文档中检索相关信息"""
        response = query_engine.query(query)
        return str(response)

    retrieve_tool = FunctionTool.from_defaults(
        fn=retrieve_documents,
        name="retrieve_insurance_info",
        description="从保险产品文档中检索相关信息",
    )

    # 创建智能体
    agent = ReActAgent(
        tools=[retrieve_tool],
        llm=llm,
        system_prompt=system_instruction,
        verbose=False,  # 设为 True 可查看思考过程
    )

    return agent, query_engine


# ==================== 步骤 7：主函数 ====================
async def main():
    """主函数 - 增强版"""

    # 1. 配置 LLM 和 Embedding
    print("🚀 初始化系统...")
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # 2. 加载文档并创建索引
    print("📂 加载文档...")
    index = load_documents_and_create_index()
    if index is None:
        print("❌ 无法创建索引，程序退出")
        return

    # 3. 创建智能体
    print("🤖 创建智能体...")
    agent, query_engine = create_agent(index, llm)

    # 4. 执行查询
    query = "介绍下雇主责任险"
    print(f"\n❓ 用户查询: {query}\n")

    # ===== 显示召回的文档内容（增强版） =====
    print("\n" + "=" * 50)
    print("📋 召回的文档内容（重排序前）")
    print("=" * 50)

    # 获取检索结果（用于展示）
    retriever = index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve(query)

    if retrieved_nodes:
        for i, node in enumerate(retrieved_nodes[:5]):  # 只显示前5个
            print(f"\n📄 文档片段 {i + 1}:")
            text_preview = node.text[:300].encode('gbk', errors='replace').decode('gbk')
            print(f"内容: {text_preview}...")
            if hasattr(node, 'score'):
                print(f"相似度分数: {node.score:.4f}")
    else:
        print("没有召回任何文档内容")
    print("=" * 50 + "\n")

    # ===== 使用智能体回答问题 =====
    print("💬 智能体回答:")
    print("-" * 50)

    try:
        response = await agent.run(query)
        response_str = str(response).encode('gbk', errors='replace').decode('gbk')
        print(response_str)
    except Exception as e:
        print(f"❌ 回答生成失败: {e}")

    print("\n" + "=" * 50)
    print("✨ 回答完成")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())