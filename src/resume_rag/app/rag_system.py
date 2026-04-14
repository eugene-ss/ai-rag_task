import logging
from typing import Any, Dict, List, Optional

from langchain_openai import AzureChatOpenAI

from resume_rag.app.answer_generator import AnswerGenerator
from resume_rag.config.settings import ConfigManager
from resume_rag.domain.models import (
    EvaluationResults,
    Permission,
    User,
)
from resume_rag.evaluation.evaluator import Evaluator
from resume_rag.ingestion.loader import DocumentLoader
from resume_rag.prompts.prompt_manager import PromptManager
from resume_rag.retrieval.bm25_index import BM25ChunkIndex
from resume_rag.retrieval.hybrid_retriever import HybridRetriever
from resume_rag.retrieval.vector_store import VectorStore
from resume_rag.security.access_control import AccessControl

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config = ConfigManager(config_path)
        prompts_dir = str(self.config.project_root / "prompts")
        self.prompt_manager = PromptManager(prompts_dir=prompts_dir)

        self.document_loader = DocumentLoader(self.config)
        self.vector_store = VectorStore(self.config)
        self.access_control = AccessControl(self.config)

        llm_config = self.config.get_llm_config()
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.endpoint_url,
            api_key=self.config.api_key,
            azure_deployment=llm_config.deployment_name,
            api_version=llm_config.api_version,
            temperature=llm_config.temperature,
        )

        self.evaluator = Evaluator(self.llm, self.config, self.prompt_manager)
        self.bm25_index = BM25ChunkIndex()
        self._bm25_path = self.config.vector_db_dir / self.config.app_settings.hybrid_search.bm25_persist_filename
        self._retriever = HybridRetriever(
            self.config, self.vector_store, self.bm25_index, self.access_control
        )
        self._answer_gen = AnswerGenerator(self.llm, self.config)
        self._init_hybrid_from_disk_or_chroma()

        logger.info("RAG System initialized successfully")

    def _init_hybrid_from_disk_or_chroma(self) -> None:
        if not self.config.app_settings.hybrid_search.enabled:
            return
        if self.bm25_index.load(self._bm25_path):
            return
        logger.warning("BM25 pickle not found; rebuilding from Chroma (if any)")
        self.bm25_index.rebuild_from_chroma(self.vector_store.vectorstore)
        if len(self.bm25_index) > 0:
            self.bm25_index.save(self._bm25_path)

    def _sync_bm25_full_rebuild(self) -> None:
        if not self.config.app_settings.hybrid_search.enabled:
            return
        self.bm25_index.rebuild_from_chroma(self.vector_store.vectorstore)
        if len(self.bm25_index) > 0:
            self.bm25_index.save(self._bm25_path)

    def load_dataset(self, csv_filename: str = "Resume.csv", load_pdfs: bool = False) -> int:
        documents = self.document_loader.load_dataset(csv_filename, load_pdfs)

        if not documents:
            logger.warning("No documents loaded")
            return 0

        self.vector_store.add_documents(documents)
        if self.config.app_settings.hybrid_search.enabled:
            if len(self.bm25_index) == 0:
                self.bm25_index.add_documents(documents)
            else:
                self.bm25_index.upsert_documents(documents)
            self.bm25_index.save(self._bm25_path)

        logger.info("Successfully loaded %s documents", len(documents))
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 10,
        user: Optional[User] = None,
    ):
        return self._retriever.search(query, k=k, user=user)

    def generate_answer(
        self,
        query: str,
        retrieved_docs,
        user: Optional[User] = None,
    ) -> str:
        return self._answer_gen.generate_answer(
            query, retrieved_docs, user, self.access_control
        )

    def update_document(
        self,
        doc_id: str,
        new_content: str,
        metadata: Dict[str, Any],
        user: Optional[User] = None,
    ):
        if user and not self.access_control.check_permission(user, Permission.WRITE):
            logger.warning("User %s denied document update permission", user.user_id)
            return False

        if not doc_id or not doc_id.strip():
            logger.error("Invalid document ID for update")
            return False

        if not new_content or not new_content.strip():
            logger.error("Invalid content for document update")
            return False

        from langchain_core.documents import Document

        from resume_rag.domain.models import DocumentMetadata

        merged = {**metadata, "id": doc_id.strip()}
        validated_metadata = DocumentMetadata(**merged)
        new_doc = Document(
            page_content=new_content,
            metadata=validated_metadata.model_dump(exclude_none=True),
        )

        if not self.vector_store.delete_documents([doc_id]):
            logger.error("Could not remove existing chunks for document %s", doc_id)
            return False
        chunks = self.document_loader.chunk_documents([new_doc])
        if not chunks:
            logger.error("Update produced no chunks after splitting")
            return False
        self.vector_store.add_documents(chunks)
        if self.config.app_settings.hybrid_search.enabled:
            self.bm25_index.upsert_documents(chunks)
            self.bm25_index.save(self._bm25_path)

        if user:
            self.access_control.log_access(user, "update_document", doc_id, True)

        logger.info("Successfully updated document %s", doc_id)
        return True

    def delete_documents(self, doc_ids: List[str], user: Optional[User] = None):
        if user and not self.access_control.check_permission(user, Permission.DELETE):
            logger.warning("User %s denied document deletion permission", user.user_id)
            return False

        if not doc_ids:
            logger.error("No document IDs provided for deletion")
            return False

        valid_ids = [doc_id for doc_id in doc_ids if doc_id and doc_id.strip()]
        if not valid_ids:
            logger.error("No valid document IDs provided")
            return False

        if not self.vector_store.delete_documents(valid_ids):
            if user:
                self.access_control.log_access(
                    user, "delete_documents", f"{len(valid_ids)} documents", False
                )
            return False

        if self.config.app_settings.hybrid_search.enabled:
            for did in valid_ids:
                self.bm25_index.remove_by_doc_id(did)
            if len(self.bm25_index) > 0:
                self.bm25_index.save(self._bm25_path)

        if user:
            self.access_control.log_access(
                user, "delete_documents", f"{len(valid_ids)} documents", True
            )

        logger.info("Successfully deleted %s documents", len(valid_ids))
        return True

    def run_evaluation(self, user: Optional[User] = None) -> EvaluationResults:
        if user and not self.access_control.check_permission(user, Permission.ANALYZE):
            raise PermissionError("User does not have analyze permission")

        eval_queries = self.document_loader.get_evaluation_queries()

        if not eval_queries:
            raise ValueError("No evaluation queries available")

        results = self.evaluator.run_evaluation(self, eval_queries, user=user)

        self._print_evaluation_summary(results)

        if user:
            self.access_control.log_access(
                user, "run_evaluation", f"{len(eval_queries)} queries", True
            )

        return results

    def run_evaluation_per_role(
        self,
        roles: Optional[List[str]] = None,
    ) -> Dict[str, EvaluationResults]:
        """Run evaluation once per role to verify access control impact."""
        from resume_rag.domain.models import Role

        if roles is None:
            roles = [r.value for r in Role]

        per_role: Dict[str, EvaluationResults] = {}
        for role_name in roles:
            try:
                role = Role(role_name)
            except ValueError:
                logger.warning("Unknown role %s, skipping", role_name)
                continue

            dept_map = {"hr_manager": "HR", "recruiter": "HR", "analyst": "IT"}
            user = User(
                user_id=f"eval_{role_name}",
                role=role,
                department=dept_map.get(role_name),
            )

            if not self.access_control.check_permission(user, Permission.ANALYZE):
                if role != Role.ADMIN:
                    logger.info("Role %s lacks ANALYZE; skipping", role_name)
                    continue

            logger.info("Running evaluation as role=%s", role_name)
            try:
                result = self.run_evaluation(user=user)
                per_role[role_name] = result
            except PermissionError:
                logger.info("Role %s denied evaluation", role_name)

        return per_role

    def _print_evaluation_summary(self, results: EvaluationResults):
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 50)

        for metric, value in results.summary.items():
            if metric.startswith("avg_"):
                metric_name = metric[4:].replace("_", " ").title()
                print(f"{metric_name:.<30} {value:.3f}")

        labeled = results.summary.get("labeled_query_count", 0)
        total = results.summary.get("total_query_count", results.total_queries)
        print(f"\nTotal Queries Evaluated: {int(total)} (Labeled: {int(labeled)})")
        print("=" * 50)

    def get_system_stats(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_collection_stats()
        hy = self.config.app_settings.hybrid_search
        return {
            "vector_store": vector_stats,
            "vector_db_directory": str(self.config.vector_db_dir),
            "results_directory": str(self.config.results_dir),
            "data_directory": str(self.config.data_dir),
            "hybrid_search_enabled": hy.enabled,
            "bm25_chunks": len(self.bm25_index),
            "structured_output_enabled": self.config.app_settings.structured_output.enabled,
        }
