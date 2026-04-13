from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from resume_rag.ingestion.resume_text import tokenize_for_bm25

logger = logging.getLogger(__name__)

def chunk_uid_for_document(doc: Document) -> str:
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    uid = meta.get("chunk_uid")
    if uid:
        return str(uid)
    rid = meta.get("id", "unknown")
    return f"{rid}::__{hash(doc.page_content) & 0xFFFFFFFF:x}"

class BM25ChunkIndex:
    def __init__(self) -> None:
        self._uids: List[str] = []
        self._tokenized: List[List[str]] = []
        self._documents: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def clear(self) -> None:
        self._uids.clear()
        self._tokenized.clear()
        self._documents.clear()
        self._bm25 = None

    def __len__(self) -> int:
        return len(self._documents)

    def _rebuild_bm25(self) -> None:
        if not self._tokenized:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokenized)

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            self._append_one(doc)
        self._rebuild_bm25()
        logger.info("BM25 index size: %s chunks", len(self._documents))

    def _append_one(self, doc: Document) -> None:
        uid = chunk_uid_for_document(doc)
        toks = tokenize_for_bm25(doc.page_content or "")
        if not toks:
            toks = ["empty"]
        self._uids.append(uid)
        self._tokenized.append(toks)
        self._documents.append(doc)

    def search(self, query: str, k: int) -> List[Document]:
        if not query or not query.strip() or not self._bm25:
            return []
        q = tokenize_for_bm25(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._documents[i] for i in ranked_idx]

    def search_ranked_uids(self, query: str, k: int) -> List[str]:
        if not query or not query.strip() or not self._bm25:
            return []
        q = tokenize_for_bm25(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._uids[i] for i in ranked_idx]

    def uid_to_document(self) -> Dict[str, Document]:
        return dict(zip(self._uids, self._documents))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "uids": self._uids,
            "tokenized": self._tokenized,
            "metadatas": [d.metadata for d in self._documents],
            "contents": [d.page_content for d in self._documents],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index to %s", path)

    def load(self, path: Path) -> bool:
        if not path.is_file():
            return False
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.clear()
        self._uids = list(payload["uids"])
        self._tokenized = list(payload["tokenized"])
        for meta, content in zip(payload["metadatas"], payload["contents"]):
            self._documents.append(Document(page_content=content, metadata=dict(meta)))
        self._rebuild_bm25()
        logger.info("Loaded BM25 index from %s (%s chunks)", path, len(self._documents))
        return True

    def rebuild_from_chroma(self, vectorstore) -> None:
        """Rebuild from a LangChain Chroma vectorstore collection."""
        coll = getattr(vectorstore, "_collection", None)
        if coll is None:
            logger.error("Chroma collection missing; cannot rebuild BM25")
            return
        batch = coll.get(include=["documents", "metadatas"])
        docs_raw = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        self.clear()
        for content, meta in zip(docs_raw, metas):
            if content is None:
                continue
            doc = Document(page_content=str(content), metadata=dict(meta or {}))
            self._append_one(doc)
        self._rebuild_bm25()
        logger.info("Rebuilt BM25 from Chroma (%s chunks)", len(self._documents))