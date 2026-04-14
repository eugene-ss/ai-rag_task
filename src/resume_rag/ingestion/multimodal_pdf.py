"""Layout-aware PDF extraction with table and image handling"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_HAS_UNSTRUCTURED = False
_HAS_PYMUPDF = False

try:
    from unstructured.partition.pdf import partition_pdf  # type: ignore
    _HAS_UNSTRUCTURED = True
except ImportError:
    pass

try:
    import fitz  # PyMuPDF  # type: ignore
    _HAS_PYMUPDF = True
except ImportError:
    pass

def _extract_with_unstructured(pdf_path: str) -> List[Tuple[str, str]]:
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
    )
    results: List[Tuple[str, str]] = []
    for el in elements:
        category = getattr(el, "category", "")
        text = str(el).strip()
        if not text:
            continue

        if category == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            if html:
                text = _html_table_to_markdown(html)
            results.append((text, "table"))
        elif category == "Image":
            results.append((text, "image_description"))
        else:
            results.append((text, "text"))
    return results

def _html_table_to_markdown(html: str) -> str:
    import re
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    md_rows: List[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        md_rows.append("| " + " | ".join(cleaned) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cleaned)) + " |")
    return "\n".join(md_rows) if md_rows else html

def _extract_with_pymupdf(pdf_path: str) -> List[Tuple[str, str]]:
    doc = fitz.open(pdf_path)
    results: List[Tuple[str, str]] = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        text = page.get_text("text")
        if text and text.strip():
            results.append((text.strip(), "text"))

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                results.append((
                    f"[Image on page {page_num + 1}, index {img_index}; "
                    f"base64 length={len(b64)}]",
                    "image_description",
                ))
                pix = None
            except Exception as exc:
                logger.debug("Could not extract image %s on page %s: %s", img_index, page_num, exc)

    doc.close()
    return results

def _extract_with_pypdf(pdf_path: str) -> List[Tuple[str, str]]:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    results: List[Tuple[str, str]] = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            results.append((text.strip(), "text"))
    return results

def extract_pdf_elements(
    pdf_path: str,
    vision_llm=None,
) -> List[Document]:

    # Extract structured elements from a PDF
    path_str = str(pdf_path)

    if _HAS_UNSTRUCTURED:
        logger.info("Using unstructured for PDF extraction: %s", path_str)
        elements = _extract_with_unstructured(path_str)
    elif _HAS_PYMUPDF:
        logger.info("Using PyMuPDF for PDF extraction: %s", path_str)
        elements = _extract_with_pymupdf(path_str)
    else:
        logger.info("Falling back to pypdf for PDF extraction: %s", path_str)
        elements = _extract_with_pypdf(path_str)

    docs: List[Document] = []
    for content, source_type in elements:
        if not content.strip():
            continue

        if source_type == "image_description" and vision_llm is not None:
            content = _describe_image_with_llm(content, vision_llm)

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "pdf",
                "source_type": source_type,
                "file_path": path_str,
            },
        ))

    logger.info(
        "Extracted %d elements from %s (text=%d, table=%d, image=%d)",
        len(docs), path_str,
        sum(1 for _, t in elements if t == "text"),
        sum(1 for _, t in elements if t == "table"),
        sum(1 for _, t in elements if t == "image_description"),
    )
    return docs

def _describe_image_with_llm(image_info: str, vision_llm) -> str:
    try:
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content=[
            {"type": "text", "text": (
                "Describe this image from a resume PDF concisely. "
                "Focus on any skills, certifications, charts, or data shown."
            )},
            {"type": "text", "text": image_info},
        ])
        response = vision_llm.invoke([msg])
        desc = (response.content or "").strip()
        return desc if desc else image_info
    except Exception as exc:
        logger.warning("Vision LLM description failed: %s", exc)
        return image_info

def get_available_backend() -> str:
    if _HAS_UNSTRUCTURED:
        return "unstructured"
    if _HAS_PYMUPDF:
        return "pymupdf"
    return "pypdf"