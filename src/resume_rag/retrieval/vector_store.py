from typing import Any, List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from resume_rag.retrieval.embeddings import build_azure_embeddings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config_manager):
        self.config = config_manager
        self.embeddings = build_azure_embeddings(config_manager)
        self.vectorstore = Chroma(
            collection_name="resume_collection",
            embedding_function=self.embeddings,
            persist_directory=config_manager.chroma_persist_dir
        )
        logger.info(f"Initialized vector store at {config_manager.chroma_persist_dir}")

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to vector store"""
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            self.vectorstore.add_documents(batch)
            logger.info(f"Added batch {batch_num}/{total_batches} ({len(batch)} documents)")

        logger.info(f"Successfully added {len(documents)} documents to vector store")

    def _chroma_collection(self):
        return getattr(self.vectorstore, "_collection", None)

    def search(self, query: str, k: int = 10, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Perform similarity search."""
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if k <= 0:
            logger.warning("Invalid k value, using default k=10")
            k = 10

        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
        return results

    def search_with_scores(self, query: str, k: int = 10) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        if not query or not query.strip():
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(results)} documents with scores")
        return results

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete all chunks whose metadata ``id`` matches a logical document id."""
        if not doc_ids:
            logger.warning("No document IDs provided for deletion")
            return False

        valid_ids = [doc_id for doc_id in doc_ids if doc_id and doc_id.strip()]
        if not valid_ids:
            logger.error("No valid document IDs provided")
            return False

        coll = self._chroma_collection()
        if coll is None:
            logger.error("Chroma collection handle missing; cannot delete by metadata id")
            return False

        if len(valid_ids) == 1:
            coll.delete(where={"id": valid_ids[0]})
        else:
            coll.delete(where={"id": {"$in": valid_ids}})
        logger.info(f"Deleted chunks for {len(valid_ids)} logical document id(s)")
        return True

    def get_collection_stats(self) -> Dict[str, Any]:
        """Collection size from Chroma only (does not call the embedding API)."""
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is not None and hasattr(collection, "count"):
            n = collection.count()
            return {
                "status": "active",
                "document_count": n,
            }
        return {"status": "unknown", "document_count": 0}