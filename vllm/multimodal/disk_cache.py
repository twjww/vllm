# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed persistence layer for the multimodal shared-memory cache.

On server startup, :class:`MultiModalDiskCache` can replay a previously saved
index back into the shared-memory ring buffer so that the very first requests
already benefit from the IPC cache (warm-start).  During normal operation
every new item is persisted asynchronously to disk so that the cache survives
process restarts.

Layout on disk::

    {cache_dir}/
        index.json              # { mm_hash: relative_path } (atomically written)
        objects/
            ab/
                ab1234…msgpack  # serialised MultiModalKwargsItem (tensor data)
        prompt_updates/
            ab/
                ab1234…pkl      # list[ResolvedPromptUpdate]  (pickle)

Files are sharded into two-character prefix sub-directories to avoid hitting
filesystem limits on single-directory entry counts.
"""

import json
import os
import cloudpickle
import queue
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_object_storage import MsgpackSerde
    from vllm.multimodal.inputs import MultiModalKwargsItem
    from vllm.multimodal.processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)

# Sentinel object used to signal the background worker to stop.
_STOP_SENTINEL: Any = object()

_INDEX_FILENAME = "index.json"
_OBJECTS_DIR = "objects"
_PROMPT_UPDATES_DIR = "prompt_updates"


def _shard_dir(base: Path, mm_hash: str) -> Path:
    """Return the two-character shard sub-directory for *mm_hash*."""
    return base / mm_hash[:2]


def _object_path(base: Path, mm_hash: str) -> Path:
    return _shard_dir(base / _OBJECTS_DIR, mm_hash) / f"{mm_hash}.msgpack"


def _prompt_updates_path(base: Path, mm_hash: str) -> Path:
    return _shard_dir(base / _PROMPT_UPDATES_DIR, mm_hash) / f"{mm_hash}.pkl"


class MultiModalDiskCache:
    """Thread-safe disk KV store that persists :class:`MultiModalKwargsItem`
    objects and their associated ``ResolvedPromptUpdate`` lists.

    Writes are handled by a single background daemon thread so the hot
    request path is never blocked.  All writes use an atomic
    write-then-rename pattern so the index is never left in a corrupt state.

    Args:
        cache_dir: Root directory for the disk cache.  Created on demand.
        max_items: Maximum number of items to keep on disk.  ``0`` means
            unlimited.  When the limit is exceeded the oldest entries
            (insertion order) are evicted from disk.
        serde: A :class:`MsgpackSerde` instance used for serialising /
            deserialising :class:`MultiModalKwargsItem` objects.
    """

    def __init__(
        self,
        cache_dir: str,
        max_items: int = 0,
        *,
        serde: "MsgpackSerde | None" = None,
    ) -> None:
        self._root = Path(cache_dir)
        self._max_items = max_items

        # Lazily import to avoid circular dependencies at module import time.
        if serde is None:
            from vllm.distributed.device_communicators.shm_object_storage import (
                MsgpackSerde,
            )

            serde = MsgpackSerde()
        self._serde = serde

        # In-memory copy of the on-disk index: mm_hash -> relative path str.
        # Maintained in insertion order (Python 3.7+ guarantee).
        self._index: dict[str, str] = {}
        self._index_lock = threading.Lock()

        # Async write queue.  Items are (mm_hash, item, prompt_updates) tuples
        # or the _STOP_SENTINEL.
        self._write_queue: queue.Queue[Any] = queue.Queue()

        # Create directory structure.
        (self._root / _OBJECTS_DIR).mkdir(parents=True, exist_ok=True)
        (self._root / _PROMPT_UPDATES_DIR).mkdir(parents=True, exist_ok=True)

        # Load existing index from disk (if any).
        self._load_index()

        # Start background writer thread.
        self._worker_thread = threading.Thread(
            target=self._worker,
            name="mm-disk-cache-writer",
            daemon=True,
        )
        self._worker_thread.start()
        logger.debug(
            "MultiModalDiskCache initialised at '%s' "
            "(max_items=%d, existing_items=%d)",
            self._root,
            self._max_items,
            len(self._index),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_index(self) -> dict[str, str]:
        """Return a snapshot of the current index (mm_hash → relative path).

        The returned dict is a *copy*; it is safe to iterate while new items
        are being written in the background.
        """
        with self._index_lock:
            return dict(self._index)

    def load_item(self, mm_hash: str) -> "MultiModalKwargsItem":
        """Deserialise and return the :class:`MultiModalKwargsItem` stored for
        *mm_hash*.

        Raises:
            KeyError: If *mm_hash* is not found in the index.
            Exception: If the file is missing or its contents are corrupt.
        """
        path = _object_path(self._root, mm_hash)
        raw = path.read_bytes()
        # MsgpackSerde.serialize returns a list[bytes]; we stored them
        # concatenated together with a leading length table so we can
        # reconstruct the original list.  See _serialise_item().
        chunks = self._deserialise_chunks(raw)
        return self._serde.mm_decoder.decode(chunks)  # type: ignore[return-value]

    def load_prompt_updates(
        self, mm_hash: str
    ) -> "list[ResolvedPromptUpdate]":
        """Deserialise and return the prompt updates stored for *mm_hash*.

        Raises:
            KeyError: If *mm_hash* is not found in the index.
            Exception: If the file is missing or its contents are corrupt.
        """
        path = _prompt_updates_path(self._root, mm_hash)
        return cloudpickle.loads(path.read_bytes())  # type: ignore[return-value]

    def save_item_async(
        self,
        mm_hash: str,
        item: "MultiModalKwargsItem",
        prompt_updates: "Sequence[ResolvedPromptUpdate]",
    ) -> None:
        """Enqueue *item* and *prompt_updates* for background disk write.

        This method returns immediately and never blocks the caller.
        """
        self._write_queue.put((mm_hash, item, prompt_updates))

    def flush(self) -> None:
        """Drain the write queue and stop the background worker.

        Should be called during server shutdown to ensure all pending writes
        are flushed to disk before the process exits.
        """
        self._write_queue.put(_STOP_SENTINEL)
        self._worker_thread.join()
        logger.debug("MultiModalDiskCache flushed and worker stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        index_path = self._root / _INDEX_FILENAME
        if not index_path.exists():
            return
        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._index = data
                logger.debug(
                    "Loaded %d entries from disk cache index.", len(self._index)
                )
        except Exception:
            logger.warning(
                "Failed to load disk cache index from '%s'; starting empty.",
                index_path,
                exc_info=True,
            )
            self._index = {}

    def _save_index_atomic(self) -> None:
        """Atomically overwrite index.json with the current in-memory index."""
        index_path = self._root / _INDEX_FILENAME
        # Write to a sibling temp file then rename for atomicity.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._root, prefix=".index_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
            os.replace(tmp_path, index_path)
        except Exception:
            logger.warning(
                "Failed to atomically update disk cache index.", exc_info=True
            )
            # Clean up temp file on failure.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _serialise_item(self, item: "MultiModalKwargsItem") -> bytes:
        """Serialise *item* to bytes using MsgpackSerde.

        ``MsgpackSerde.serialize`` returns ``(data, nbytes, metadata, md_bytes)``
        where *data* is a ``list[bytes]`` for :class:`MultiModalKwargsItem`.
        We store the result as::

            [4-byte n_chunks][4-byte len_0]…[4-byte len_{n-1}][chunk_0]…[chunk_{n-1}]

        This allows reconstruction without storing a separate length table
        file.
        """
        data, _nbytes, _metadata, _md_bytes = self._serde.serialize(item)
        if isinstance(data, (bytes, bytearray, memoryview)):
            chunks: list[bytes] = [bytes(data)]
        else:
            chunks = [bytes(c) for c in data]

        n = len(chunks)
        header = n.to_bytes(4, "little")
        lengths = b"".join(len(c).to_bytes(4, "little") for c in chunks)
        return header + lengths + b"".join(chunks)

    def _deserialise_chunks(self, raw: bytes) -> list[bytes]:
        """Inverse of :meth:`_serialise_item`; returns the list of chunks."""
        n = int.from_bytes(raw[:4], "little")
        offset = 4
        lengths: list[int] = []
        for _ in range(n):
            lengths.append(int.from_bytes(raw[offset : offset + 4], "little"))
            offset += 4
        chunks: list[bytes] = []
        for length in lengths:
            chunks.append(raw[offset : offset + length])
            offset += length
        return chunks

    def _write_item(
        self,
        mm_hash: str,
        item: "MultiModalKwargsItem",
        prompt_updates: "Sequence[ResolvedPromptUpdate]",
    ) -> None:
        """Serialise and write *item* + *prompt_updates* to disk, then update
        the index.  Runs exclusively in the background worker thread."""
        try:
            # Serialise tensor data.
            obj_path = _object_path(self._root, mm_hash)
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            raw_item = self._serialise_item(item)
            self._atomic_write_bytes(obj_path, raw_item)

            # Serialise prompt updates.
            pu_path = _prompt_updates_path(self._root, mm_hash)
            pu_path.parent.mkdir(parents=True, exist_ok=True)
            raw_pu = cloudpickle.dumps(list(prompt_updates))
            self._atomic_write_bytes(pu_path, raw_pu)

            # Update in-memory index and persist it atomically.
            rel_path = str(obj_path.relative_to(self._root))
            with self._index_lock:
                if mm_hash not in self._index:
                    self._index[mm_hash] = rel_path
                    self._evict_if_needed()
                self._save_index_atomic()

        except Exception:
            logger.warning(
                "Failed to write mm item '%s' to disk cache.", mm_hash, exc_info=True
            )

    def _evict_if_needed(self) -> None:
        """Remove the oldest entries when *max_items* is exceeded.

        Must be called with :attr:`_index_lock` held.
        """
        if self._max_items <= 0 or len(self._index) <= self._max_items:
            return

        n_evict = len(self._index) - self._max_items
        evicted = list(self._index.keys())[:n_evict]
        for mm_hash in evicted:
            del self._index[mm_hash]
            self._delete_files(mm_hash)
        logger.debug("Evicted %d old entries from disk cache.", n_evict)

    def _delete_files(self, mm_hash: str) -> None:
        """Remove the object and prompt-update files for *mm_hash*."""
        for path in (
            _object_path(self._root, mm_hash),
            _prompt_updates_path(self._root, mm_hash),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """Write *data* to *path* atomically via a sibling temp file."""
        tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Background thread: drain the write queue until stopped."""
        while True:
            item = self._write_queue.get()
            if item is _STOP_SENTINEL:
                break
            mm_hash, mm_item, prompt_updates = item
            # Skip if already on disk.
            with self._index_lock:
                already_saved = mm_hash in self._index
            if not already_saved:
                self._write_item(mm_hash, mm_item, prompt_updates)

