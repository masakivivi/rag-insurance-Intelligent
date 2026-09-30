# coding: utf-8
"""索引管理（P1：增量同步 + manifest 对账）。

设计要点：
- manifest.json 记录 每个文件 → {hash, node_ids}，作为文档目录与向量索引的对账单。
- 删除直接按 node_ids 调 delete_nodes，与 LlamaIndex 内部 ref_doc_id 方案解耦，最稳。
- 三入口（启动 / 手动 / watchdog）走同一套 sync_index 逻辑。
- manifest 与 storage 绑定；manifest 缺失视为损坏 → 显式重建（恢复路径，非运行时 fallback）。
"""
import asyncio
import hashlib
import json
import logging
import os
from os.path import getmtime, getsize, relpath

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from app import config

logger = logging.getLogger("insurance-qa.index")


class IndexManager:
    """管理向量索引的加载、构建与增量同步。"""

    def __init__(self):
        self.index = None
        self._splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separator=" ",
        )
        self._manifest = {"files": {}, "version": 1}
        # 写索引的进程内锁：与 watcher、手动 sync、优雅关闭共用
        self.lock = asyncio.Lock()
        # 索引变更回调（async）：sync 产生变更后触发，用于全局清空查询缓存
        self.on_index_changed = None

    # ------------------------------------------------------------------
    # 加载 / 构建
    # ------------------------------------------------------------------
    def load_or_build(self):
        """加载已有索引；manifest 缺失则从 docstore 生成（首次迁移），storage 损坏则显式重建。"""
        if config.STORAGE_DIR.exists():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(config.STORAGE_DIR))
                self.index = load_index_from_storage(storage_context)
                if config.MANIFEST_PATH.exists():
                    self._manifest = self._load_manifest()
                    logger.info("索引与 manifest 加载成功")
                else:
                    # 首次迁移：已有索引健康但 manifest 不存在，从 docstore 生成
                    self._bootstrap_manifest()
                    self._save_manifest()
                    logger.info("索引加载成功，manifest 已从 docstore 生成")
                return self.index
            except Exception as e:
                logger.warning("加载索引失败: %s，将显式重建", e)
                self.index = None

        return self._build_from_docs()

    def _build_from_docs(self):
        """从 docs/ 全量构建索引 + 生成 manifest（首次或损坏恢复时调用）。"""
        if not config.DOCS_DIR.exists():
            raise FileNotFoundError(f"文档目录不存在: {config.DOCS_DIR}")

        documents = SimpleDirectoryReader(str(config.DOCS_DIR)).load_data()
        if not documents:
            raise ValueError("文档目录为空，无法构建索引")

        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[self._splitter],
            show_progress=True,
        )
        self.index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
        # 从构建好的 docstore 反向生成 manifest
        self._bootstrap_manifest()
        self._save_manifest()
        logger.info("索引重建完成，manifest 已生成")
        return self.index

    # ------------------------------------------------------------------
    # manifest 读写
    # ------------------------------------------------------------------
    def _load_manifest(self) -> dict:
        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self):
        with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2)

    def _bootstrap_manifest(self):
        """从已有 docstore 反向构建 manifest（file → node_ids）。

        用 docstore.docs 按 node.ref_doc_id 分组，不依赖不确定的 ref_doc_ids API。
        """
        files = {}
        docstore = self.index.docstore
        groups: dict[str, list[str]] = {}
        for node_id, node in docstore.docs.items():
            ref = getattr(node, "ref_doc_id", None)
            if ref:
                groups.setdefault(ref, []).append(node_id)
        for ref_doc_id, node_ids in groups.items():
            first_node = docstore.docs.get(node_ids[0])
            if first_node is None:
                continue
            fp = (first_node.metadata or {}).get("file_path")
            if not fp:
                continue
            rel = self._rel(fp)
            files[rel] = {
                "hash": self._file_hash(fp) if os.path.exists(fp) else None,
                "size": os.path.getsize(fp) if os.path.exists(fp) else None,
                "mtime": os.path.getmtime(fp) if os.path.exists(fp) else None,
                "ref_doc_id": ref_doc_id,
                "node_ids": node_ids,
            }
        self._manifest = {"files": files, "version": 1}

    # ------------------------------------------------------------------
    # 文件扫描与对账
    # ------------------------------------------------------------------
    @staticmethod
    def _rel(abs_path: str) -> str:
        return relpath(abs_path, str(config.BASE_DIR)).replace("\\", "/")

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    def _scan_docs(self) -> dict:
        """扫描 docs/，返回 {rel_path: {size, mtime, hash}}。"""
        result = {}
        for root, _dirs, files in os.walk(str(config.DOCS_DIR)):
            for name in files:
                if os.path.splitext(name)[1].lower() not in config.SUPPORTED_EXTS:
                    continue
                ap = os.path.join(root, name)
                rel = self._rel(ap)
                result[rel] = {
                    "abs": ap,
                    "size": getsize(ap),
                    "mtime": getmtime(ap),
                }
        return result

    def _diff(self, scan: dict):
        """对比扫描结果与 manifest，返回 (new, modified, deleted, unchanged)。"""
        manifest_files = self._manifest["files"]
        new, modified, deleted, unchanged = [], [], [], []
        for rel, info in scan.items():
            if rel not in manifest_files:
                new.append(rel)
                continue
            entry = manifest_files[rel]
            # 快速判别：size/mtime 未变则视为未变；变化时算 hash 精确确认
            if entry.get("size") == info["size"] and entry.get("mtime") == info["mtime"]:
                unchanged.append(rel)
            else:
                h = self._file_hash(info["abs"])
                if entry.get("hash") == h:
                    unchanged.append(rel)
                else:
                    info["hash"] = h
                    modified.append(rel)
        for rel in manifest_files:
            if rel not in scan:
                deleted.append(rel)
        return new, modified, deleted, unchanged

    # ------------------------------------------------------------------
    # 增量同步（核心）
    # ------------------------------------------------------------------
    def _sync_sync(self) -> dict:
        """同步逻辑（同步实现，由 async sync_index 在线程中调用）。"""
        scan = self._scan_docs()
        new, modified, deleted, unchanged = self._diff(scan)
        changed = False

        for rel in deleted:
            self._delete_file(rel)
            changed = True
        for rel in new:
            self._insert_file(scan[rel]["abs"], rel)
            changed = True
        for rel in modified:
            # 先删旧节点，再插新
            self._delete_file(rel)
            self._insert_file(scan[rel]["abs"], rel, hash_=scan[rel]["hash"])
            changed = True

        if changed:
            self.index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
            self._save_manifest()
            logger.info("增量同步完成：新增=%d 修改=%d 删除=%d 未变=%d", len(new), len(modified), len(deleted), len(unchanged))
        else:
            logger.info("增量同步完成：无变更（%d 个文件未变）", len(unchanged))

        return {"new": len(new), "modified": len(modified), "deleted": len(deleted), "unchanged": len(unchanged)}

    async def sync_index(self) -> dict:
        """增量同步入口（启动 / 手动 / watchdog 共用）。

        产生变更后，触发 on_index_changed 回调（如全局清空查询缓存）。
        """
        async with self.lock:
            stats = await asyncio.to_thread(self._sync_sync)
        # 在锁外触发缓存失效回调（不阻塞后续 sync）
        if self.on_index_changed is not None and (
            stats["new"] or stats["modified"] or stats["deleted"]
        ):
            try:
                await self.on_index_changed()
            except Exception as e:
                logger.warning("缓存失效回调失败: %s", e)
        return stats

    # ------------------------------------------------------------------
    # 单文件增量动作
    # ------------------------------------------------------------------
    def _insert_file(self, abs_path: str, rel: str, hash_: str | None = None):
        """读取单个文件 → 切分 → 插入索引，并记录到 manifest。"""
        documents = SimpleDirectoryReader(input_files=[abs_path]).load_data()
        if not documents:
            logger.warning("文件无内容，跳过: %s", rel)
            return
        nodes = self._splitter.get_nodes_from_documents(documents)
        if not nodes:
            logger.warning("文件解析后无有效文本 chunk，跳过: %s", rel)
            return
        self.index.insert_nodes(nodes, show_progress=False)
        node_ids = [n.node_id for n in nodes]
        ref_doc_id = nodes[0].ref_doc_id
        self._manifest["files"][rel] = {
            "hash": hash_ or self._file_hash(abs_path),
            "size": os.path.getsize(abs_path),
            "mtime": os.path.getmtime(abs_path),
            "ref_doc_id": ref_doc_id,
            "node_ids": node_ids,
        }
        logger.info("已索引文件: %s（%d 个 chunk）", rel, len(node_ids))

    def _delete_file(self, rel: str):
        """按 manifest 中的 node_ids 删除该文件的全部节点。"""
        entry = self._manifest["files"].get(rel)
        if not entry:
            return
        node_ids = entry.get("node_ids") or []
        if node_ids:
            self.index.delete_nodes(node_ids, deleting_from_docstore=True)
        del self._manifest["files"][rel]
        logger.info("已删除文件索引: %s（%d 个 chunk）", rel, len(node_ids))
