# coding: utf-8
"""API 路由：/ask（安全管道）、/health、/index/sync（增量同步）。

P2 接入 sanitize → 隔离检索合成 → filter；P4 接入缓存与并发闸。
"""
import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import config
from app.security.sanitizer import sanitize
from app.security.filter import filter_output
from app.api.errors import SecurityError

router = APIRouter()
sec_logger = logging.getLogger("insurance-qa.security")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[str] = []


@router.get("/health")
async def health(request: Request):
    """健康检查：实际探测各组件状态。"""
    state = request.app.state
    index = getattr(state, "index", None)
    index_ok = index is not None and len(index.docstore.docs) > 0
    watcher = getattr(state, "watcher", None)
    watcher_ok = watcher is not None and getattr(watcher, "_observer", None) is not None
    cache = getattr(state, "cache", None)
    components = {
        "llm": "ok" if getattr(state, "llm", None) and config.DASHSCOPE_API_KEY else "down",
        "embed_model": "ok" if getattr(state, "embed_model", None) else "down",
        "index": "ok" if index_ok else "down",
        "query_engine": "ok" if getattr(state, "query_engine", None) else "down",
        "watcher": "ok" if watcher_ok else "down",
        "cache": "ok" if cache is not None and cache.available else "down",
        "storage": "ok" if config.STORAGE_DIR.exists() else "down",
        "manifest": "ok" if config.MANIFEST_PATH.exists() else "down",
    }
    status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return {"status": status, "components": components}


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    """问答入口：输入清洗 → 缓存命中检查 → 并发闸 → 隔离检索合成 → 输出过滤 → 写缓存。"""
    # 1. 输入清洗 + 注入检测
    sq = sanitize(req.query)
    if sq.is_injection:
        sec_logger.warning("注入命中，已拒绝: pattern=%s raw=%r", sq.matched_pattern, sq.raw[:100])
        raise SecurityError("检测到注入模式，已拒绝")

    cache = request.app.state.cache
    semaphore = request.app.state.llm_semaphore

    # 2. 缓存命中秒回（不占用 LLM 并发闸）
    cached = await cache.get(sq.sanitized)
    if cached is not None:
        return AskResponse(
            query=req.query, answer=cached["answer"], sources=cached["sources"]
        )

    query_engine = request.app.state.query_engine
    # 3. 并发闸：限制 LLM 并发，超限排队；排队超时直接拒绝，避免请求无限挂
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=config.LLM_QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="当前并发已满，请稍后重试")
    try:
        response = await query_engine.aquery(sq.sanitized)
    finally:
        semaphore.release()
    # 4. 输出过滤
    answer = filter_output(str(response))
    sources = (
        [n.node.text[:200] for n in response.source_nodes]
        if response.source_nodes
        else []
    )
    # 5. 写缓存（无检索命中用短 TTL，由 cache 内部判断）
    await cache.set(sq.sanitized, answer, sources)
    return AskResponse(query=req.query, answer=answer, sources=sources)


@router.post("/index/sync")
async def index_sync(request: Request):
    """手动触发增量同步。"""
    index_manager = request.app.state.index_manager
    stats = await index_manager.sync_index()
    return {"status": "synced", "stats": stats}
