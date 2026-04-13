import json
import logging
from textwrap import dedent
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from resume_rag.config.settings import ConfigManager
from resume_rag.domain.models import Permission, RAGStructuredAnswer, SearchResult, User
from resume_rag.llm.json_utils import loads_json_stripped
from resume_rag.security.access_control import AccessControl

logger = logging.getLogger(__name__)

RESUME_ANSWER_SYSTEM = (
    "You are an **expert HR analyst**. Ground every factual claim in the resume excerpts below. "
    "If the excerpts do not support an assertion, say so explicitly."
)

RESUME_ANSWER_USER_TEMPLATE = """**Query**: {query}

**Resume excerpts**:
{documents}

{output_spec}"""

OUTPUT_SPEC_STRUCTURED = dedent(
    """
    Output format: respond with a JSON object with keys:
    - summary
    - candidates (array of objects with resume_id, category, evidence_snippet, relevance_note)
    - confidence (low, medium, high)
    """
).strip()

OUTPUT_SPEC_PROSE = dedent(
    """
    Provide a detailed, professional response with specific examples drawn from the excerpts above.
    """
).strip()

class AnswerGenerator:
    def __init__(self, llm, config: ConfigManager) -> None:
        self.llm = llm
        self.config = config

    @staticmethod
    def _resume_answer_messages(
        query: str, documents_text: str, output_spec: str
    ) -> List[BaseMessage]:
        user_content = RESUME_ANSWER_USER_TEMPLATE.format(
            query=query,
            documents=documents_text,
            output_spec=output_spec,
        )
        return [
            SystemMessage(content=RESUME_ANSWER_SYSTEM),
            HumanMessage(content=user_content),
        ]

    @staticmethod
    def build_context_budget(retrieved_docs: List[SearchResult], max_chars: int) -> str:
        parts: List[str] = []
        used = 0
        for sr in retrieved_docs:
            meta = sr.document.metadata
            rid = getattr(meta, "id", "")
            cat = getattr(meta, "category", "")
            headline = getattr(meta, "headline", None) or ""
            header = f"--- resume_id={rid} category={cat}"
            if headline:
                header += f" headline={headline[:80]}"
            header += " ---\n"
            allowance = max_chars - used - len(header) - 4
            if allowance < 120:
                break
            body = (sr.document.page_content or "")[:allowance]
            block = header + body
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[SearchResult],
        user: Optional[User],
        access_control: AccessControl,
    ) -> str:
        if not query or not query.strip():
            return "Invalid query provided"

        if user and not isinstance(user, User):
            return "Invalid user"

        if user and not access_control.check_permission(user, Permission.READ):
            return "Access denied: Insufficient permissions"

        if not retrieved_docs:
            return "No relevant documents found to generate an answer"

        so = self.config.app_settings.structured_output
        documents_text = self.build_context_budget(
            retrieved_docs, so.max_context_chars
        )

        if so.enabled:
            messages = self._resume_answer_messages(
                query, documents_text, OUTPUT_SPEC_STRUCTURED
            )
            try:
                structured_llm = self.llm.with_structured_output(RAGStructuredAnswer)
                parsed = structured_llm.invoke(messages)
                if isinstance(parsed, RAGStructuredAnswer):
                    out = parsed.model_dump_json(indent=2)
                else:
                    out = json.dumps(parsed, indent=2)
            except Exception as e:
                logger.warning("Structured output failed (%s); using JSON prompt fallback", e)
                base_user = RESUME_ANSWER_USER_TEMPLATE.format(
                    query=query,
                    documents=documents_text,
                    output_spec=OUTPUT_SPEC_STRUCTURED,
                )
                extra = (
                    base_user
                    + "\n\nReturn a single JSON object with keys: summary (string), "
                    "candidates (array of {resume_id, category, evidence_snippet, relevance_note}), "
                    "confidence (string)."
                )
                fb_msgs: List[BaseMessage] = [
                    SystemMessage(content=RESUME_ANSWER_SYSTEM),
                    HumanMessage(content=extra),
                ]
                response = self.llm.invoke(fb_msgs)
                raw = (response.content or "").strip()
                data = loads_json_stripped(raw)
                out = RAGStructuredAnswer.model_validate(data).model_dump_json(indent=2)
        else:
            messages = self._resume_answer_messages(
                query, documents_text, OUTPUT_SPEC_PROSE
            )
            response = self.llm.invoke(messages)
            out = response.content or ""

        if user:
            access_control.log_access(user, "generate_answer", query, True)

        return out
