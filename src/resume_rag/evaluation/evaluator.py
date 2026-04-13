import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from resume_rag.domain.models import (
    EvaluationMetrics,
    EvaluationQuery,
    EvaluationQualityScores,
    EvaluationResults,
    SearchResult,
)
from resume_rag.llm.json_utils import loads_json_stripped

logger = logging.getLogger(__name__)

def load_eval_labels(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        logger.info("No eval labels file at %s; using heuristic relevance only", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, List[str]] = {}
    for row in data.get("labels", []):
        q = row.get("query")
        if not q or "relevant_resume_ids" not in row:
            continue
        out[str(q)] = [str(x) for x in row["relevant_resume_ids"]]
    logger.info("Loaded %s labeled evaluation queries from %s", len(out), path)
    return out

class Evaluator:
    def __init__(self, llm, config_manager, prompt_manager):
        self.llm = llm
        self.config = config_manager
        self.prompt_manager = prompt_manager
        self.eval_config = config_manager.get_evaluation_config()

    @staticmethod
    def precision_at_k(retrieved: List[SearchResult], relevant_ids: List[str], k: int) -> float:
        """Calculate Precision."""
        if not retrieved or k <= 0:
            return 0.0

        k = min(k, len(retrieved))
        top_k = retrieved[:k]

        relevant_count = sum(
            1 for result in top_k
            if result.document.metadata.id in relevant_ids
        )
        return relevant_count / k

    def recall_at_k(self, retrieved: List[SearchResult], relevant_ids: List[str], k: int) -> float:
        """Calculate Recall."""
        if not relevant_ids or not retrieved or k <= 0:
            return 0.0

        k = min(k, len(retrieved))
        top_k = retrieved[:k]

        relevant_count = sum(
            1 for result in top_k
            if result.document.metadata.id in relevant_ids
        )
        return relevant_count / len(relevant_ids)

    @staticmethod
    def _format_numbered_excerpts(
        retrieved: List[SearchResult], max_n: int, chars: int
    ) -> Tuple[str, int]:
        lines: List[str] = []
        n = min(max_n, len(retrieved))
        for i in range(n):
            text = (retrieved[i].document.page_content or "")[:chars]
            lines.append(f"### Excerpt {i + 1}\n{text}")
        return "\n\n".join(lines), n

    @staticmethod
    def _avg_relevance_from_parsed(parsed: EvaluationQualityScores, n_excerpts: int) -> float:
        if n_excerpts <= 0:
            return 0.0
        by_idx = {r.index: r.score for r in parsed.relevance_scores}
        vals = [float(by_idx.get(i, 0.5)) for i in range(1, n_excerpts + 1)]
        return float(np.mean(vals))

    def _evaluation_quality(
        self, query: str, context: str, answer: str, numbered_excerpts: str
    ) -> EvaluationQualityScores:
        messages = self.prompt_manager.get_messages(
            "evaluate",
            query=query,
            context=context,
            answer=answer,
            numbered_excerpts=numbered_excerpts,
        )
        try:
            structured_llm = self.llm.with_structured_output(EvaluationQualityScores)
            parsed = structured_llm.invoke(messages)
            if isinstance(parsed, EvaluationQualityScores):
                return parsed
            return EvaluationQualityScores.model_validate(dict(parsed))
        except Exception as e:
            logger.warning("Structured evaluate prompt failed (%s); JSON fallback", e)
            response = self.llm.invoke(messages)
            raw = (response.content or "").strip()
            try:
                data = loads_json_stripped(raw)
                return EvaluationQualityScores.model_validate(data)
            except Exception:
                return EvaluationQualityScores(
                    faithfulness=0.5,
                    groundedness=0.5,
                    relevance_scores=[],
                )

    def evaluate_query(
        self,
        query: str,
        retrieved: List[SearchResult],
        answer: str,
        relevant_ids: List[str],
    ) -> EvaluationMetrics:
        metrics_data: Dict = {"query": query}

        for k in self.eval_config.precision_k_values:
            metrics_data[f"precision_at_{k}"] = self.precision_at_k(retrieved, relevant_ids, k)

        for k in self.eval_config.recall_k_values:
            metrics_data[f"recall_at_{k}"] = self.recall_at_k(retrieved, relevant_ids, k)

        answer_eval = self._answer_text_for_eval(answer)

        if retrieved and answer:
            top_docs = retrieved[:3]
            context = "\n".join(
                [doc.document.page_content[:500] for doc in top_docs]
            )
            numbered_block, n_excerpts = self._format_numbered_excerpts(
                retrieved, max_n=5, chars=500
            )
            if n_excerpts == 0:
                metrics_data["faithfulness"] = 0.0
                metrics_data["groundedness"] = 0.0
                metrics_data["avg_relevance"] = 0.0
            else:
                parsed = self._evaluation_quality(
                    query, context, answer_eval, numbered_block
                )
                metrics_data["faithfulness"] = float(parsed.faithfulness)
                metrics_data["groundedness"] = float(parsed.groundedness)
                metrics_data["avg_relevance"] = self._avg_relevance_from_parsed(
                    parsed, n_excerpts
                )
        else:
            metrics_data["faithfulness"] = 0.0
            metrics_data["groundedness"] = 0.0
            metrics_data["avg_relevance"] = 0.0

        return EvaluationMetrics(**metrics_data)

    @staticmethod
    def _answer_text_for_eval(answer: str) -> str:
        s = (answer or "").strip()
        if not s.startswith("{"):
            return s
        try:
            data = json.loads(s)
            if isinstance(data, dict) and data.get("summary"):
                return str(data["summary"])
        except json.JSONDecodeError:
            pass
        return s

    @staticmethod
    def identify_relevant_docs_heuristic(
        retrieved: List[SearchResult], eval_query: EvaluationQuery
    ) -> List[str]:
        relevant_ids: List[str] = []
        keywords = eval_query.keywords or []

        for result in retrieved:
            doc_category = result.document.metadata.category
            doc_content = (result.document.page_content or "").lower()
            skills_meta = getattr(result.document.metadata, "skills", None) or ""
            skills_lower = str(skills_meta).lower()

            category_match = doc_category in eval_query.relevant_categories
            keyword_hits = sum(1 for kw in keywords if kw.lower() in doc_content)
            skill_hits = sum(1 for kw in keywords if kw.lower() in skills_lower) if skills_lower else 0

            if category_match and (keyword_hits > 0 or not keywords):
                relevant_ids.append(result.document.metadata.id)
            elif category_match and skill_hits > 0:
                relevant_ids.append(result.document.metadata.id)
            elif keyword_hits >= 2:
                relevant_ids.append(result.document.metadata.id)

        return list(dict.fromkeys(relevant_ids))

    def run_evaluation(self, rag_system, eval_queries: List[EvaluationQuery]) -> EvaluationResults:
        logger.info("Starting evaluation on %s queries", len(eval_queries))

        labels_path = self.config.data_dir / self.eval_config.eval_labels_path
        labels_map = load_eval_labels(labels_path)

        all_metrics = []

        for i, eval_query in enumerate(eval_queries):
            logger.info("Evaluating query %s/%s: %s", i + 1, len(eval_queries), eval_query.query)

            retrieved = rag_system.search(
                eval_query.query, k=self.eval_config.max_docs_for_evaluation
            )

            if not retrieved:
                logger.warning("No results for query: %s", eval_query.query)
                continue

            answer = rag_system.generate_answer(eval_query.query, retrieved[:3])

            if eval_query.query in labels_map:
                relevant_ids = list(dict.fromkeys(labels_map[eval_query.query]))
                logger.info("Using labeled relevant_resume_ids (count=%s)", len(relevant_ids))
            else:
                relevant_ids = self.identify_relevant_docs_heuristic(retrieved, eval_query)

            metrics = self.evaluate_query(
                eval_query.query,
                retrieved,
                answer,
                relevant_ids,
            )
            all_metrics.append(metrics)

            logger.info(
                "Results - P@1: %.3f, R@5: %.3f, Faithfulness: %.3f",
                metrics.precision_at_1,
                metrics.recall_at_5,
                metrics.faithfulness,
            )

        if not all_metrics:
            raise ValueError("No queries were successfully evaluated")

        summary = {}
        metric_names = [
            "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
            "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
            "faithfulness", "groundedness", "avg_relevance",
        ]

        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in all_metrics]
            summary[f"avg_{metric_name}"] = float(np.mean(values)) if values else 0.0
            summary[f"std_{metric_name}"] = float(np.std(values)) if values else 0.0

        logger.info("Evaluation completed. Successful: %s", len(all_metrics))

        self._save_evaluation_results(all_metrics, summary)

        return EvaluationResults(
            summary=summary,
            individual_results=all_metrics,
            total_queries=len(all_metrics),
        )

    def _save_evaluation_results(self, all_metrics: List[EvaluationMetrics], summary: Dict[str, float]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.config.evaluation_results_dir) / f"evaluation_{timestamp}.json"

        results_data = {
            "timestamp": timestamp,
            "summary": summary,
            "individual_results": [metrics.model_dump() for metrics in all_metrics],
            "total_queries": len(all_metrics),
        }

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Detailed evaluation results saved to %s", results_file)