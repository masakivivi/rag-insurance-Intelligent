# coding: utf-8
"""FastAPI 服务入口（P0：lifespan + 路由装配；P3：访问日志/优雅 drain/统一异常兜底）。"""
import asyncio
import logging

from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# 注册 starlette 的 HTTPException（父类）才能覆盖路由 404；fastapi.HTTPException 是其子类，按 mro 同样命中
from starlette.exceptions import HTTPException

# 必须先导入 config 以设置 HF 环境变量，再导入 llama_index 相关模块
from app import config  # noqa: F401  确保环境变量先就位
from llama_index.core import Settings

from app.api.errors import InsuranceQAError
from app.api.middleware import AccessLogDrainMiddleware, DrainTracker
from app.cache import QueryCache
from app.core.llm import setup_llm_and_embedding
from app.core.indexer import IndexManager
from app.core.retriever import build_query_engine
from app.core.watcher import Watcher
from app.api.routes import router

logger = logging.getLogger("insurance-qa")


def setup_logging():
    """配置日志：控制台 + 滚动文件落盘。

    必须在 uvicorn 完成 dictConfig 之后调用（即 lifespan 启动阶段），
    否则 root 配置会被 uvicorn 默认配置覆盖或不被设置。

    - root 级别强制 INFO（uvicorn 不一定设置 root 级别，默认 WARNING 会过滤掉业务 INFO）
    - 控制台 + 文件 handler 均幂等追加，避免与 uvicorn 既有 handler 重复
    - 文件：logs/app.log，单文件 10MB，保留 5 份，UTF-8
    - 屏蔽 uvicorn.access 默认访问日志，用自研 insurance-qa.access（含 duration）替代
    """
    log_dir = config.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 控制台 handler（幂等：uvicorn 可能已加 StreamHandler，避免重复输出）
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)

    # 文件 handler（幂等：避免 reload/重复 startup 重复加）
    has_file = any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    if not has_file:
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # 用自研访问日志（含 duration）替代 uvicorn 默认 access log，避免重复
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # 第三方库 INFO 噪声（HF 模型下载/加载请求）降到 WARNING，保持业务日志干净
    for name in ("httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期：启动初始化，关闭清理。"""
    setup_logging()
    logger.info("启动：初始化 LLM 与 Embedding")
    llm, embed_model = setup_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    app.state.llm = llm
    app.state.embed_model = embed_model

    logger.info("启动：加载/构建索引")
    index_manager = IndexManager()
    index = index_manager.load_or_build()
    app.state.index = index
    app.state.index_manager = index_manager

    logger.info("启动：构建查询引擎")
    app.state.query_engine = build_query_engine(index)

    # P4：查询缓存 + LLM 并发闸
    app.state.cache = QueryCache()
    await app.state.cache.connect()
    app.state.llm_semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)
    # 索引增量变更后全局清空 qa:* 缓存（watchdog / 手动 sync 共用此回调）
    async def _on_index_changed():
        # 增量变更后：清缓存 + 重建查询引擎（刷新 BM25 内存索引）
        await app.state.cache.clear_all()
        try:
            app.state.query_engine = build_query_engine(app.state.index)
            logger.info("查询引擎已重建（BM25 nodes 刷新）")
        except Exception as e:
            logger.warning("查询引擎重建失败，沿用旧引擎: %s", e)

    index_manager.on_index_changed = _on_index_changed

    # 启动 watchdog 实时监听 + 初始同步（处理停机期间的变化）
    app.state.watcher = Watcher(index_manager, asyncio.get_running_loop())
    app.state.watcher.start()
    await index_manager.sync_index()

    logger.info("启动完成")
    yield

    # 优雅关闭：先拒绝新请求并排空在途请求，再卸载组件
    tracker = getattr(app.state, "tracker", None)
    if tracker is not None:
        tracker.begin_shutdown()
        remaining = await tracker.wait_drained(config.DRAIN_TIMEOUT)
        if remaining > 0:
            logger.warning(
                "排空超时（%.1fs），仍有 %d 个在途请求，强制关闭",
                config.DRAIN_TIMEOUT,
                remaining,
            )
    if getattr(app.state, "watcher", None) is not None:
        await app.state.watcher.stop()
    if getattr(app.state, "cache", None) is not None:
        await app.state.cache.close()
    logger.info("关闭：清理完成")


def create_app() -> FastAPI:
    app = FastAPI(title="保险智能问答系统", lifespan=lifespan)

    # 访问日志 + drain 中间件（最外层，包裹所有路由与异常处理）
    tracker = DrainTracker()
    app.state.tracker = tracker
    app.add_middleware(AccessLogDrainMiddleware, tracker=tracker)

    app.include_router(router)

    # 统一异常处理：自定义异常 → 结构化错误响应 {code, message}
    @app.exception_handler(InsuranceQAError)
    async def _qa_exception_handler(request: Request, exc: InsuranceQAError):
        logger.warning("请求异常: %s code=%s", exc, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": str(exc)},
        )

    # HTTPException（含 drain 期间的 503）统一为结构化响应
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        code_map = {503: "SHUTTING_DOWN", 429: "RATE_LIMITED"}
        code = code_map.get(exc.status_code, f"HTTP_{exc.status_code}")
        logger.info("HTTP %d %s: %s", exc.status_code, request.url.path, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": code, "message": exc.detail},
        )

    # 兜底：未捕获异常 → 结构化 500，避免返回默认 HTML 堆栈
    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        logger.exception("未捕获异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"code": "INTERNAL_ERROR", "message": "服务内部错误"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
