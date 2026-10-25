"""Datasheet and design text search: a local SQLite FTS5 index, mirrored to Moss.

The local index always holds every chunk, so search works offline and Moss can be rebuilt
from it. With Moss credentials set, chunks are also pushed to one Moss index, which is loaded
into this process for hybrid (semantic + keyword) queries; a Moss failure falls back to the
local index and says so. Search results are reading material with citations. They are never
a source of coordinates, nets or pin maps; those come from the design store.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MOSS_INDEX = "hardware-troubleshooter-docs"
_JOB_TIMEOUT_S = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY, kind TEXT, mpn TEXT, title TEXT, path TEXT, pages INTEGER,
  ingested_at REAL, moss_synced INTEGER DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5 (
  chunk_id UNINDEXED, source_id UNINDEXED, kind UNINDEXED, mpn, page UNINDEXED, text,
  tokenize = 'porter unicode61'
);
"""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    kind: str  # datasheet | component
    mpn: str
    page: int | None
    text: str

    def metadata(self) -> dict[str, str]:
        # Moss metadata values are strings.
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "mpn": self.mpn,
            "page": "" if self.page is None else str(self.page),
        }


def pdf_pages(path: str | Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion needs the docs extra: pip install '.[docs]'") from exc
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def chunk_pages(source_id: str, kind: str, mpn: str, pages: list[str]) -> list[Chunk]:
    chunks = []
    for number, text in enumerate(pages, start=1):
        text = re.sub(r"[ \t]+", " ", text).strip()
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_CHARS)
            if end < len(text):  # break at a paragraph or sentence when one is near
                floor = start + CHUNK_CHARS // 2
                cut = max(text.rfind("\n", floor, end), text.rfind(". ", floor, end))
                end = cut + 1 if cut > 0 else end
            body = text[start:end].strip()
            if body:
                chunk_id = f"{source_id}:p{number}:{start}"
                chunks.append(Chunk(chunk_id, source_id, kind, mpn, number, body))
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


class LocalIndex:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    def replace(self, source: dict, chunks: list[Chunk]) -> None:
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE source_id = ?", (source["source_id"],))
            self.db.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    source["source_id"],
                    source["kind"],
                    source["mpn"],
                    source["title"],
                    source["path"],
                    source["pages"],
                    time.time(),
                ),
            )
            self.db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                [(c.chunk_id, c.source_id, c.kind, c.mpn, c.page, c.text) for c in chunks],
            )

    def chunk_ids(self, source_id: str) -> set[str]:
        rows = self.db.execute("SELECT chunk_id FROM chunks WHERE source_id = ?", (source_id,))
        return {r[0] for r in rows.fetchall()}

    def mark_synced(self, source_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE sources SET moss_synced = 1 WHERE source_id = ?", (source_id,))

    def sources(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM sources ORDER BY ingested_at DESC").fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, mpn: str | None, top_k: int) -> list[dict]:
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = (
            "SELECT chunk_id, source_id, kind, mpn, page, text, bm25(chunks) AS rank "
            "FROM chunks WHERE chunks MATCH ?"
        )
        args: list = [match]
        if mpn:
            sql, args = sql + " AND mpn = ?", args + [mpn]
        rows = self.db.execute(sql + " ORDER BY rank LIMIT ?", args + [top_k]).fetchall()
        return [
            {
                "chunk_id": r["chunk_id"],
                "source_id": r["source_id"],
                "kind": r["kind"],
                "mpn": r["mpn"],
                "page": r["page"],
                "score": -float(r["rank"]),
                "text": r["text"],
            }
            for r in rows
        ]


class MossIndex:
    """One Moss index, loaded into this process. Calls are async; failures raise."""

    def __init__(self, project_id: str, project_key: str, name: str = MOSS_INDEX):
        from inferedge_moss import MossClient

        self.client = MossClient(project_id, project_key)
        self.name = name
        self.loaded = False

    async def upsert(self, chunks: list[Chunk]) -> None:
        from inferedge_moss import DocumentInfo, MutationOptions

        docs = [DocumentInfo(id=c.chunk_id, text=c.text, metadata=c.metadata()) for c in chunks]
        names = {index.name for index in await self.client.list_indexes()}
        if self.name in names:
            result = await self.client.add_docs(self.name, docs, MutationOptions(upsert=True))
        else:
            result = await self.client.create_index(self.name, docs)
        await self._wait(result.job_id)
        await self.load()

    async def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            result = await self.client.delete_docs(self.name, chunk_ids)
            await self._wait(result.job_id)

    async def load(self) -> None:
        await self.client.load_index(self.name)
        self.loaded = True

    async def search(self, query: str, mpn: str | None, top_k: int) -> list[dict]:
        from inferedge_moss import QueryOptions

        if not self.loaded:
            await self.load()
        flt = {"field": "mpn", "condition": {"$eq": mpn}} if mpn else None
        result = await self.client.query(self.name, query, QueryOptions(top_k=top_k, filter=flt))
        out = []
        for doc in result.docs:
            meta = doc.metadata or {}
            page = meta.get("page", "")
            out.append(
                {
                    "chunk_id": doc.id,
                    "source_id": meta.get("source_id", ""),
                    "kind": meta.get("kind", ""),
                    "mpn": meta.get("mpn", ""),
                    "page": int(page) if page.isdigit() else None,
                    "score": float(doc.score),
                    "text": doc.text,
                }
            )
        return out

    async def _wait(self, job_id: str) -> None:
        if not job_id:
            return
        deadline = time.monotonic() + _JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            status = await self.client.get_job_status(job_id)
            value = getattr(status.status, "value", str(status.status)).lower()
            if value == "completed":
                return
            if value == "failed":
                raise RuntimeError(f"Moss job {job_id} failed: {status.error}")
            await asyncio.sleep(1.0)
        raise TimeoutError(f"Moss job {job_id} didn't finish in {_JOB_TIMEOUT_S:.0f} s")


class DocsLibrary:
    def __init__(self, local: LocalIndex, moss: MossIndex | None):
        self.local = local
        self.moss = moss

    @property
    def backend(self) -> str:
        return "moss" if self.moss else "local"

    async def add(self, source: dict, chunks: list[Chunk]) -> dict:
        # Moss upserts by id, so chunks an earlier ingestion had and this one lacks are deleted.
        previous = self.local.chunk_ids(source["source_id"])
        self.local.replace(source, chunks)
        out = {"source_id": source["source_id"], "chunks": len(chunks), "moss": "not_configured"}
        if self.moss is not None:
            try:
                await self.moss.upsert(chunks)
                await self.moss.delete(sorted(previous - {c.chunk_id for c in chunks}))
                self.local.mark_synced(source["source_id"])
                out["moss"] = "synced"
            except Exception as exc:  # the local index still has it; report and carry on
                LOG.exception("Moss sync failed")
                out["moss"] = f"sync_failed:{type(exc).__name__}: {exc}"
        return out

    async def search(self, query: str, mpn: str | None, top_k: int) -> dict:
        if self.moss is not None:
            try:
                return {
                    "backend": "moss",
                    "results": await self.moss.search(query, mpn, top_k),
                }
            except Exception as exc:
                LOG.exception("Moss query failed; using the local index")
                return {
                    "backend": "local",
                    "fallback_reason": f"moss_failed:{type(exc).__name__}",
                    "results": self.local.search(query, mpn, top_k),
                }
        return {"backend": "local", "results": self.local.search(query, mpn, top_k)}
