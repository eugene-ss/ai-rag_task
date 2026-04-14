import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from resume_rag.domain.models import DocumentMetadata, EvaluationQuery
from resume_rag.ingestion.multimodal_pdf import extract_pdf_elements, get_available_backend
from resume_rag.ingestion.resume_text import (
    extract_headline,
    extract_skills_line,
    normalize_resume_text,
)

logger = logging.getLogger(__name__)

def assign_chunk_metadata(chunks: List[Document]) -> List[Document]:
    if not chunks:
        return chunks
    by_id: defaultdict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        meta = c.metadata if isinstance(c.metadata, dict) else {}
        rid = str(meta.get("id", i))
        by_id[rid].append(i)
    out = list(chunks)
    for rid, indices in by_id.items():
        n = len(indices)
        for j, idx in enumerate(indices):
            ch = out[idx]
            md = dict(ch.metadata) if isinstance(ch.metadata, dict) else {}
            md["chunk_index"] = j
            md["total_chunks"] = n
            md["chunk_uid"] = f"{rid}:{j}"
            out[idx] = Document(page_content=ch.page_content, metadata=md)
    return out

class DocumentLoader:
    def __init__(self, config_manager, vision_llm=None):
        self.config = config_manager
        self.vision_llm = vision_llm
        text_config = config_manager.get_text_splitter_config()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_config.chunk_size,
            chunk_overlap=text_config.chunk_overlap,
        )
        logger.info("PDF backend: %s", get_available_backend())

    def _doc_processing(self):
        return self.config.app_settings.document_processing

    def load_dataset(self, csv_filename: str = "Resume.csv", load_pdfs: bool = False) -> List[Document]:
        # Load resume dataset from CSV and PDF files
        csv_path = self.config.data_dir / csv_filename

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        documents = []
        df = pd.read_csv(csv_path)
        logger.info(f"Loading {len(df)} resumes from {csv_path}")
        dp = self._doc_processing()

        for idx, row in df.iterrows():
            if "Resume_str" in row and pd.notna(row["Resume_str"]):
                text = str(row["Resume_str"])
                if dp.normalize_text:
                    text = normalize_resume_text(text)

                headline = None
                skills = None
                if dp.extract_headline_skills:
                    headline = extract_headline(text)
                    skills = extract_skills_line(text)

                try:
                    oi = int(idx)
                except (TypeError, ValueError):
                    oi = None
                metadata = DocumentMetadata(
                    id=str(row.get("ID", idx)),
                    category=str(row.get("Category", "Unknown")),
                    source="csv",
                    original_index=oi,
                    headline=headline,
                    skills=skills,
                )

                doc = Document(
                    page_content=text,
                    metadata=metadata.model_dump(exclude_none=True),
                )
                documents.append(doc)

            if load_pdfs and "ID" in row and "Category" in row:
                pdf_path = self.config.data_dir / str(row["Category"]) / f"{row['ID']}.pdf"
                if pdf_path.exists():
                    pdf_docs = self._load_pdf(pdf_path, row)
                    documents.extend(pdf_docs)

        chunked = self.chunk_documents(documents)
        return assign_chunk_metadata(chunked)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        text_docs = []
        passthrough_docs = []
        for doc in documents:
            st = (doc.metadata or {}).get("source_type", "text")
            if st in ("table", "image_description"):
                passthrough_docs.append(doc)
            else:
                text_docs.append(doc)

        chunked_docs = self.text_splitter.split_documents(text_docs) if text_docs else []
        chunked_docs.extend(passthrough_docs)

        logger.info(
            "Created %d chunks from %d documents (text-split=%d, passthrough=%d)",
            len(chunked_docs), len(documents),
            len(chunked_docs) - len(passthrough_docs), len(passthrough_docs),
        )
        return assign_chunk_metadata(chunked_docs)

    def _load_pdf(self, pdf_path: Path, row: pd.Series) -> List[Document]:
        raw_docs = extract_pdf_elements(str(pdf_path), vision_llm=self.vision_llm)
        dp = self._doc_processing()
        out: List[Document] = []

        for doc in raw_docs:
            text = doc.page_content or ""
            source_type = (doc.metadata or {}).get("source_type", "text")

            if source_type == "text" and dp.normalize_text:
                text = normalize_resume_text(text)

            headline = None
            skills = None
            if source_type == "text" and dp.extract_headline_skills:
                headline = extract_headline(text)
                skills = extract_skills_line(text)

            metadata = DocumentMetadata(
                id=str(row["ID"]),
                category=str(row["Category"]),
                source="pdf",
                source_type=source_type,
                file_path=str(pdf_path),
                headline=headline,
                skills=skills,
            )
            doc.page_content = text
            doc.metadata = metadata.model_dump(exclude_none=True)
            out.append(doc)

        return out

    def get_evaluation_queries(self) -> List[EvaluationQuery]:
        queries_data = [
            {
                "query": "Find Python developers with machine learning experience",
                "relevant_categories": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
                "keywords": ["python", "machine learning", "ml", "ai", "tensorflow", "pytorch"],
            },
            {
                "query": "Healthcare professionals with patient care experience",
                "relevant_categories": ["HEALTHCARE", "FITNESS"],
                "keywords": ["patient", "care", "medical", "nursing", "healthcare", "clinical"],
            },
            {
                "query": "Financial analysts with Excel and data analysis skills",
                "relevant_categories": ["FINANCE", "ACCOUNTANT", "BANKING"],
                "keywords": ["excel", "financial", "analysis", "accounting", "data", "spreadsheet"],
            },
            {
                "query": "Software engineers with web development experience",
                "relevant_categories": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
                "keywords": ["web development", "javascript", "html", "css", "react", "angular", "vue"],
            },
            {
                "query": "Sales professionals with B2B experience",
                "relevant_categories": ["SALES", "BUSINESS-DEVELOPMENT"],
                "keywords": ["sales", "b2b", "business development", "client", "revenue", "crm"],
            },
            {
                "query": "HR managers with recruitment and talent acquisition background",
                "relevant_categories": ["HR", "BUSINESS-DEVELOPMENT"],
                "keywords": ["recruitment", "talent acquisition", "hiring", "onboarding", "hr"],
            },
            {
                "query": "Teachers with curriculum development and classroom management skills",
                "relevant_categories": ["TEACHER"],
                "keywords": ["curriculum", "teaching", "classroom", "education", "instruction"],
            },
            {
                "query": "Graphic designers with UI/UX and Adobe Creative Suite expertise",
                "relevant_categories": ["DESIGNER", "ARTS", "DIGITAL-MEDIA"],
                "keywords": ["ui", "ux", "adobe", "photoshop", "illustrator", "figma", "design"],
            },
            {
                "query": "Accountants with auditing and tax compliance experience",
                "relevant_categories": ["ACCOUNTANT", "FINANCE"],
                "keywords": ["audit", "tax", "compliance", "accounting", "gaap", "cpa"],
            },
            {
                "query": "Construction project managers with safety certification",
                "relevant_categories": ["CONSTRUCTION", "ENGINEERING"],
                "keywords": ["construction", "project management", "safety", "osha", "site"],
            },
            {
                "query": "Data engineers with cloud computing and ETL pipeline experience",
                "relevant_categories": ["INFORMATION-TECHNOLOGY", "ENGINEERING"],
                "keywords": ["data engineering", "etl", "cloud", "aws", "azure", "spark", "pipeline"],
            },
            {
                "query": "Legal professionals with corporate law and contract negotiation skills",
                "relevant_categories": ["ADVOCATE"],
                "keywords": ["law", "legal", "contract", "litigation", "corporate", "compliance"],
            },
            {
                "query": "Fitness trainers with nutrition certification and wellness coaching",
                "relevant_categories": ["FITNESS", "HEALTHCARE"],
                "keywords": ["fitness", "nutrition", "training", "wellness", "coaching", "certified"],
            },
            {
                "query": "Banking professionals with risk management and regulatory compliance",
                "relevant_categories": ["BANKING", "FINANCE"],
                "keywords": ["banking", "risk", "regulatory", "compliance", "financial", "credit"],
            },
            {
                "query": "Aviation engineers with maintenance and safety inspection background",
                "relevant_categories": ["AVIATION", "ENGINEERING"],
                "keywords": ["aviation", "aircraft", "maintenance", "safety", "inspection", "faa"],
            },
        ]

        return [EvaluationQuery(**query_data) for query_data in queries_data]