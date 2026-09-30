# coding: utf-8
"""Redis 查询缓存（P4）：按规范化 query 的 sha256 前缀作 key，命中秒回。

设计要点（见 design.md 3.4.1）：
- key = qa:{sha256(normalize(query))[:16]}，normalize = 去标点 + 折叠空白 + 小写
- value = {"answer": ..., "sources": [...]}（JSON）
- 写入策略：回答成功后写缓存；无检索命中的负向回答用短 TTL（避免重复空跑）
- 失效：索引增量同步产生变更后，由 IndexManager.on_index_changed 回调全局清空 qa:*
- 降级：Redis 不可用时缓存静默关闭，不影响问答主流程
"""
import hashlib
import json
import logging
import re

import redis.asyncio as redis

from app import config

logger = logging.getLogger("insurance-qa.cache")

# 中英文标点 + 空白，规范化时一并去除
_PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:\"'()\[\]{}<>《》【】\s]+")
# 负向回答（无检索命中）短 TTL，避免重复空跑
_SHORT_TTL = 60


def normalize(query: str) -> str:
    """规范化 query 作为缓存 key 基础：去标点 + 折叠空白 + 小写。"""
    return _PUNCT_RE.sub("", query).lower()


def cache_key(query: str) -> str:
    """生成缓存 key。"""
    digest = hashlib.sha256(normalize(query).encode("utf-8")).hexdigest()[:16]
    return f"qa:{digest}"


class QueryCache:
    """Redis 查询缓存封装。Redis 不可用时优雅降级。"""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        """连接 Redis；失败则降级关闭（不抛异常，保证服务可启动）。"""
        try:
            self._client = redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._client.ping()
            logger.info("Redis 缓存已连接: %s", config.REDIS_URL)
        except Exception as e:
            logger.warning("Redis 不可用，缓存降级关闭: %s", e)
            self._client = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Redis 缓存已关闭")

    async def get(self, query: str) -> dict | None:
        """读缓存；失败时返回 None（视为未命中）。"""
        if self._client is None:
            return None
        try:
            raw = await self._client.get(cache_key(query))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("缓存读失败，视为未命中: %s", e)
            return None

    async def set(self, query: str, answer: str, sources: list[str]) -> None:
        """写缓存；无检索命中用短 TTL，否则用配置 TTL。"""
        if self._client is None:
            return
        ttl = _SHORT_TTL if not sources else config.QUERY_CACHE_TTL
        payload = json.dumps(
            {"answer": answer, "sources": sources}, ensure_ascii=False
        )
        try:
            await self._client.set(cache_key(query), payload, ex=ttl)
        except Exception as e:
            logger.warning("缓存写失败，跳过: %s", e)

    async def clear_all(self) -> int:
        """全局失效：清空所有 qa:* 缓存。返回清除条数。"""
        if self._client is None:
            return 0
        deleted = 0
        try:
            async for key in self._client.scan_iter(match="qa:*", count=200):
                await self._client.delete(key)
                deleted += 1
            if deleted:
                logger.info("缓存已全局失效：清除 %d 条", deleted)
        except Exception as e:
            logger.warning("缓存清空失败: %s", e)
        return deleted
