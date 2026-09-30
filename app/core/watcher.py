# coding: utf-8
"""watchdog 实时监听（P1）。

设计要点：
- 监听 docs/ 文件变化，全局防抖 1.5s（编辑器保存会触发多次 modified，需合并）。
- 防抖静默后触发 index_manager.sync_index()，由其统一做 diff+增量（移动=旧删+新增，sync 自动覆盖）。
- 事件过滤：扩展名白名单 + 忽略临时文件。
- watchdog 回调跑在独立线程，用 call_soon_threadsafe 转回事件循环调度防抖任务。
"""
import asyncio
import logging
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import config

logger = logging.getLogger("insurance-qa.watcher")

_DEBOUNCE_SEC = 1.5


class _Handler(FileSystemEventHandler):
    """watchdog 事件处理器，转发给 Watcher。"""

    def __init__(self, watcher):
        super().__init__()
        self._watcher = watcher

    def _emit(self, path: str, event_type: str):
        if path:
            self._watcher._on_event(path, event_type)

    def on_created(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "created")

    def on_modified(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "modified")

    def on_deleted(self, event):
        if not event.is_directory:
            self._emit(event.src_path, "deleted")

    def on_moved(self, event):
        if not event.is_directory:
            # 移动 = 旧路径删除 + 新路径新增，sync_index 自动覆盖，这里仅触发
            self._emit(event.dest_path, "moved")


class Watcher:
    """实时监听 docs/ 变化，防抖后触发增量同步。"""

    def __init__(self, index_manager, loop: asyncio.AbstractEventLoop):
        self._im = index_manager
        self._loop = loop
        self._observer: Observer | None = None
        self._pending: asyncio.Task | None = None  # 防抖中的同步任务

    def start(self):
        if not config.DOCS_DIR.exists():
            logger.warning("docs 目录不存在，watcher 不启动")
            return
        handler = _Handler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(config.DOCS_DIR), recursive=True)
        self._observer.start()
        logger.info("watchdog 监听已启动: %s", config.DOCS_DIR)

    async def stop(self):
        # 停掉 observer，不再产生新事件
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        # 取消尚未触发的防抖同步（已运行的 sync 在线程中完成，不强行打断）
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
            try:
                await self._pending
            except asyncio.CancelledError:
                pass
            self._pending = None
        logger.info("watchdog 监听已停止")

    # ------------------------------------------------------------------
    # 事件过滤 + 防抖 + 触发同步
    # ------------------------------------------------------------------
    @staticmethod
    def _is_relevant(path: str) -> bool:
        name = os.path.basename(path)
        # 忽略临时文件与 Office 锁文件
        if name.startswith("~$") or name.startswith(".") or name.endswith((".tmp", ".swp", ".crdownload")):
            return False
        if os.path.splitext(name)[1].lower() not in config.SUPPORTED_EXTS:
            return False
        return True

    def _on_event(self, path: str, event_type: str):
        """watchdog 线程回调：过滤后把调度丢回事件循环。"""
        if not self._is_relevant(path):
            return
        self._loop.call_soon_threadsafe(self._schedule_sync, path, event_type)

    def _schedule_sync(self, path: str, event_type: str):
        # 全局防抖：任意相关事件都重置单一计时器
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = self._loop.create_task(self._debounced_sync(path, event_type))

    async def _debounced_sync(self, path: str, event_type: str):
        try:
            await asyncio.sleep(_DEBOUNCE_SEC)
        except asyncio.CancelledError:
            return
        name = os.path.basename(path)
        logger.info("watchdog 触发增量同步（%s, %s）", event_type, name)
        try:
            await self._im.sync_index()
        except Exception as e:
            logger.exception("watchdog 触发的同步失败: %s", e)
