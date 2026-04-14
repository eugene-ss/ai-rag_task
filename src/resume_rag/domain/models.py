from typing import List, Dict, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator

class Role(str, Enum):
    ADMIN = "admin"
    HR_MANAGER = "hr_manager"
    RECRUITER = "recruiter"
    ANALYST = "analyst"

class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ANALYZE = "analyze"

class User(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier")
    role: Role = Field(..., description="User role")
    department: Optional[str] = Field(None, description="User department")
    allowed_categories: Optional[Set[str]] = Field(None, description="Explicitly allowed categories")

    class Config:
        use_enum_values = True

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Document unique identifier")
    category: str = Field(..., description="Resume category")
    source: str = Field(..., description="Document source (csv, pdf, table, image_description)")
    file_path: Optional[str] = Field(None, description="Original file path")
    original_index: Optional[int] = Field(None, description="Original CSV index")
    headline: Optional[str] = Field(None, description="Short title line from resume")
    skills: Optional[str] = Field(
        None,
        description="Comma-separated skills for metadata filters / display",
    )
    chunk_index: Optional[int] = Field(None, description="0-based chunk within logical resume")
    total_chunks: Optional[int] = Field(None, description="Chunk count for this resume id")
    chunk_uid: Optional[str] = Field(None, description="Stable id for hybrid fusion (id:chunk_idx)")
    owner_id: Optional[str] = Field(None, description="User ID of the document owner/uploader")
    access_list: Optional[List[str]] = Field(None, description="User IDs explicitly granted access")
    source_type: Optional[str] = Field(None, description="Content type: text, table, image_description")

class ResumeDocument(BaseModel):
    page_content: str = Field(..., min_length=1, description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")

class SearchResult(BaseModel):
    document: ResumeDocument = Field(..., description="Retrieved document")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    method: str = Field(
        ...,
        description="How the hit was ranked: dense vector retrieval or hybrid (dense + BM25 fusion).",
    )

class EvaluationQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Evaluation query")
    relevant_categories: List[str] = Field(default_factory=list, description="Expected relevant categories")
    keywords: List[str] = Field(default_factory=list, description="Expected keywords")

class EvaluationMetrics(BaseModel):
    precision_at_1: float = Field(..., ge=0.0, le=1.0)
    precision_at_3: float = Field(..., ge=0.0, le=1.0)
    precision_at_5: float = Field(..., ge=0.0, le=1.0)
    precision_at_10: float = Field(..., ge=0.0, le=1.0)
    recall_at_1: float = Field(..., ge=0.0, le=1.0)
    recall_at_3: float = Field(..., ge=0.0, le=1.0)
    recall_at_5: float = Field(..., ge=0.0, le=1.0)
    recall_at_10: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    avg_relevance: float = Field(..., ge=0.0, le=1.0)
    query: str = Field(..., description="Original query")
    has_labels: bool = Field(default=False, description="Whether gold labels were available")

class EvaluationResults(BaseModel):
    summary: Dict[str, float] = Field(..., description="Average metrics")
    individual_results: List[EvaluationMetrics] = Field(..., description="Per-query results")
    total_queries: int = Field(..., ge=0, description="Total number of evaluated queries")

class LLMConfig(BaseModel):
    model: str = Field(..., description="Model name")
    api_version: str = Field(..., description="API version")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: int = Field(..., gt=0, description="Maximum tokens")
    deployment_name: str = Field(..., description="Deployment name")

class EmbeddingConfig(BaseModel):
    model: str = Field(..., description="Embedding model name")
    chunk_size: int = Field(..., gt=0, description="Chunk size for embeddings")
    deployment_name: str = Field(..., description="Deployment name")

class TextSplitterConfig(BaseModel):
    chunk_size: int = Field(..., gt=0, description="Text chunk size")
    chunk_overlap: int = Field(..., ge=0, description="Chunk overlap size")

    @model_validator(mode="after")
    def overlap_less_than_chunk_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

class StorageConfig(BaseModel):
    chroma_db_path: str = Field(..., description="ChromaDB storage path")
    evaluation_results_path: str = Field(..., description="Evaluation results path")
    logs_path: str = Field(..., description="Logs storage path")

class AppConfig(BaseModel):
    name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    log_level: str = Field(..., description="Logging level")

class AccessControlConfig(BaseModel):
    department_categories: Dict[str, List[str]] = Field(..., description="Department to categories mapping")

class EvaluationConfig(BaseModel):
    precision_k_values: List[int] = Field(..., description="K values for precision calculation")
    recall_k_values: List[int] = Field(..., description="K values for recall calculation")
    max_docs_for_evaluation: int = Field(..., gt=0, description="Maximum documents for evaluation")
    eval_labels_path: str = Field(
        default="dataset/eval_labels.json",
        description="Optional JSON with per-query labeled relevant resume IDs",
    )


class HybridSearchConfig(BaseModel):
    enabled: bool = Field(default=False, description="BM25 + dense RRF fusion")
    sparse_k: int = Field(default=50, ge=1, description="BM25 candidate count")
    rrf_k: int = Field(default=60, ge=1, description="RRF rank constant")
    fusion: str = Field(default="rrf", description="rrf or weighted")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_persist_filename: str = Field(
        default="bm25_index.pkl",
        description="Pickle file under vector DB dir (saved after load_dataset)",
    )


class StructuredOutputConfig(BaseModel):
    enabled: bool = Field(default=True, description="JSON schema answer via structured LLM output")
    max_context_chars: int = Field(default=12000, ge=500, description="Budget for retrieved chunks in prompt")


class DocumentProcessingConfig(BaseModel):
    normalize_text: bool = Field(default=True, description="Collapse whitespace in resume text")
    extract_headline_skills: bool = Field(default=True, description="Fill headline/skills metadata")


class CitedCandidate(BaseModel):
    resume_id: str
    category: str = ""
    evidence_snippet: str = ""
    relevance_note: str = ""

class RAGStructuredAnswer(BaseModel):
    summary: str
    candidates: List[CitedCandidate] = Field(default_factory=list)
    confidence: str = "medium"


class ExcerptRelevanceScore(BaseModel):
    index: int = Field(..., ge=1, description="Excerpt number (1-based)")
    score: float = Field(..., ge=0.0, le=1.0)

class EvaluationQualityScores(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How completely the answer addresses all aspects of the query",
    )
    relevance_scores: List[ExcerptRelevanceScore] = Field(
        default_factory=list,
        description="One entry per numbered excerpt, query relevance 0-1",
    )

class ApplicationSettings(BaseModel):
    app: AppConfig
    llm: LLMConfig
    embeddings: EmbeddingConfig
    text_splitter: TextSplitterConfig
    storage: StorageConfig
    access_control: AccessControlConfig
    evaluation: EvaluationConfig
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)

    model_config = ConfigDict(env_file=".env")