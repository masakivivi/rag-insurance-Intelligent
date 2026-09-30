# coding: utf-8
"""HTTP 中间件（P3）：访问日志 + 优雅排空（drain）。

- 访问日志：记录 method/path/status/duration_ms（替代 uvicorn 默认 access log）
- 优雅 drain：关闭期间拒绝新请求（503），并等待在途请求完成后再卸载组件
"""
import asyncio
import logging
import time

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("insurance-qa.access")


class DrainTracker:
    """在途请求计数 + 关闭标记，用于优雅排空。"""

    def __init__(self) -> None:
        self._inflight = 0
        self._shutting_down = False
        # 归零事件：inflight==0 时置位，用于 wait_drained 等待
        self._zero_event = asyncio.Event()
        self._zero_event.set()

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def acquire(self) -> None:
        """请求进入；关闭期间直接拒绝（运维探活的 /health 除外）。"""
        if self._shutting_down:
            raise HTTPException(status_code=503, detail="服务正在关闭，暂不接受新请求")
        self._inflight += 1
        if self._inflight == 1:
            self._zero_event.clear()

    def release(self) -> None:
        """请求结束：计数 -1，归零时置位事件。"""
        if self._inflight > 0:
            self._inflight -= 1
        if self._inflight == 0:
            self._zero_event.set()

    def begin_shutdown(self) -> None:
        """进入关闭流程：拒绝新请求。"""
        self._shutting_down = True
        if self._inflight == 0:
            self._zero_event.set()

    async def wait_drained(self, timeout: float) -> int:
        """等待在途请求归零，超时后返回剩余数量。"""
        try:
            await asyncio.wait_for(self._zero_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self._inflight


class AccessLogDrainMiddleware(BaseHTTPMiddleware):
    """访问日志 + drain 中间件。"""

    def __init__(self, app: ASGIApp, tracker: DrainTracker) -> None:
        super().__init__(app)
        self._tracker = tracker

    async def dispatch(self, request: Request, call_next):
        # /health 关闭期间也允许探活（运维/编排系统需要判断就绪）
        is_health = request.url.path == "/health"
        acquired = False
        start = time.perf_counter()
        status_code = 500
        try:
            if not is_health:
                self._tracker.acquire()  # 关闭期间抛 HTTPException(503)
                acquired = True
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except HTTPException as e:
            status_code = e.status_code
            raise
        finally:
            if acquired:
                self._tracker.release()
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "%s %s %s %dms",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
