# RAG Task

### Context

A **RAG** system for **searching and answering questions over a dataset of resumes** (HR / recruiting / etc.). It indexes resume text, retrieves the most relevant profiles for a user question and generates a short answer with the **role-based access** to job categories.

The system supports teams that need to **explore a structured resume approach without manually reading every file**. A recruiter or analyst asks questions; the system finds relevant resume chunks, ranks them and produces an **LLM-generated answer. Access control** limits which job categories each user role may see.

---

## Getting started

### Prerequisites

- **Python 3.11+**
- **UV package manager**
- **Azure OpenAI-compatible API**: base URL and API key (see `.env.example`). Need working deployments for **chat** and **embeddings** matching names in [`config/app_config.yaml`](config/app_config.yaml) (`llm.deployment_name`, `embeddings.deployment_name`).
- **Resume data**: by default the app expects **`dataset/Resume.csv`** at the repository root (or whatever you set as `DATA_DIR` in `.env`). The CSV should include at least **`ID`**, **`Category`**, and **`Resume_str`** (resume text). PDFs can live under `dataset/<CATEGORY>/<ID>.pdf` when using `--load-pdfs`.

---

### Installation

```bash
# Install UV package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd <repository_name>

# Install dependencies (installs the `resume_rag` package from `src/` into the project env)
uv sync

# (Optional) Install multi-modal PDF extraction dependencies
uv sync --extra multimodal

# Setup environment
cp .env.example .env   # edit .env with your API credentials

# Verify CLI (run from repository root so config/ and evaluation prompts/ resolve)
uv run python -m resume_rag.main --help
```

---

## Usage

Run commands from the **repository root** unless you pass absolute paths. The app loads [`config/app_config.yaml`](config/app_config.yaml) relative to that root; evaluation loads markdown from [`prompts/`](prompts/) (answer-generation text is inlined in [`src/resume_rag/app/answer_generator.py`](src/resume_rag/app/answer_generator.py)).

`uv sync` makes the `resume_rag` package importable; the entry point is defined in [`pyproject.toml`](pyproject.toml) as `resume_rag.main:main`.

### Indexing

- The **first indexing run** embeds resumes and persists Chroma under `VECTOR_DB_DIR` (see [Vector database location](#vector-database-location)); it can take a while.
- Re-running indexing is **idempotent**: chunks already present in Chroma (matched by `chunk_uid`) are skipped automatically, so duplicate vectors are not created.
- When using `--query`, `--evaluate`, `--visualize`, or `--eval-per-role`, dataset loading is **auto-skipped** unless you pass `--force-reload`.

### Common workflows

```bash
# Initial index from CSV (with optional PDFs per row under dataset/<Category>/<ID>.pdf)
uv run python -m resume_rag.main --csv-file Resume.csv
uv run python -m resume_rag.main --csv-file Resume.csv --load-pdfs

# Query against an existing index
uv run python -m resume_rag.main --query "Who is Accountant?"

# Same query, with a role that affects category filtering (admin | hr | recruiter | analyst)
uv run python -m resume_rag.main --query "Financial analyst" --user-type hr

# Run evaluation (writes JSON under results/evaluation_results/)
uv run python -m resume_rag.main --evaluate

# Run evaluation once per role to compare access control impact
uv run python -m resume_rag.main --eval-per-role

# Generate visualization charts from the latest evaluation results
uv run python -m resume_rag.main --visualize

# Generate charts from a specific evaluation file, with a comparison run
uv run python -m resume_rag.main --visualize results/evaluation_results/evaluation_20250101_120000.json \
  --compare-with results/evaluation_results/evaluation_20250102_120000.json

# Alternate config file (path relative to repo root unless absolute)
uv run python -m resume_rag.main --config custom_config.yaml --csv-file Resume.csv
```

### Document Maintenance

Update and delete operations work **without rebuilding** the entire vector store or BM25 index. Only affected chunks are removed/re-indexed.

```bash
# Delete logical documents by metadata id (comma-separated)
uv run python -m resume_rag.main --skip-load --delete-docs "id1,id2"

# Replace one document's text (file or inline), then re-chunk and re-index
uv run python -m resume_rag.main --skip-load --update-doc RESUME_ID --update-file /path/to/new_text.txt
uv run python -m resume_rag.main --skip-load --update-doc RESUME_ID --update-text "New resume body..." \
  --update-category INFORMATION-TECHNOLOGY --update-source csv
```

### Flags (summary)

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML app config (default `config/app_config.yaml`) |
| `--csv-file` | CSV name inside `DATA_DIR` (default `Resume.csv`) |
| `--load-pdfs` | Also load `dataset/<Category>/<ID>.pdf` via multi-modal extraction |
| `--skip-load` | Do not reload CSV/PDF into the vector index |
| `--force-reload` | Force full re-index even when using `--query` or `--evaluate` |
| `--query` | Run a single search + answer |
| `--evaluate` | Run the evaluation suite |
| `--eval-per-role` | Run evaluation once per role (admin, hr_manager, analyst) |
| `--visualize [PATH]` | Generate bar charts from evaluation results (latest or specified file) |
| `--compare-with PATH` | Second evaluation JSON for side-by-side comparison (with `--visualize`) |
| `--user-type` | Test user role for access control (`admin`, `hr`, `recruiter`, `analyst`) |
| `--delete-docs` | Comma-separated logical document IDs to remove from Chroma |
| `--update-doc` | Logical document ID to replace (with `--update-file` or `--update-text`) |
| `--update-category` / `--update-source` | Metadata when using `--update-doc` |

---

## Evaluating results

### `--evaluate`

For each query:

1. Retrieves up to **`max_docs_for_evaluation`** candidates (from [`config/app_config.yaml`](config/app_config.yaml) > `evaluation`, default 10).
2. Builds an answer from the top retrieved chunks.
3. Computes **Precision@K** and **Recall@K** for K in `precision_k_values` / `recall_k_values` (default 1, 3, 5, 10). P@K/R@K are only computed for **labeled queries** (those with gold `relevant_resume_ids` in [`dataset/eval_labels.json`](dataset/eval_labels.json)); unlabeled queries still get LLM quality scoring.
4. An LLM structured call scores **faithfulness**, **groundedness**, **answer completeness**, and **per-excerpt relevance** (mean -> `avg_relevance`) via [`prompts/evaluate.md`](prompts/evaluate.md).

### Evaluation dataset

The evaluation dataset lives in [`dataset/eval_labels.json`](dataset/eval_labels.json) with **15 labeled queries** covering all resume categories (IT, Healthcare, Finance, Sales, HR, Education, Creative, Legal, Fitness, Banking, Aviation, Construction). Each query includes:

- `relevant_resume_ids` (3-4 gold-standard IDs)
- `irrelevant_resume_ids` (negative examples)
- `difficulty` (easy / medium / hard)
- `category_scope` (expected categories)

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| **Precision@K** | Retrieval | Fraction of top-K results that are relevant (labeled queries only) |
| **Recall@K** | Retrieval | Fraction of relevant docs found in top-K (labeled queries only) |
| **Faithfulness** | LLM quality | Answer does not contradict retrieved context (0-1) |
| **Groundedness** | LLM quality | Answer claims are supported by context wording (0-1) |
| **Answer Completeness** | LLM quality | How completely the answer addresses all aspects of the query (0-1) |
| **Avg Relevance** | LLM quality | Mean per-excerpt relevance to the query (0-1) |

### Saved artifacts

| Output | Location |
|--------|----------|
| **Per-run JSON** (run metadata, summary + per-query metrics) | `RESULTS_DIR` / `evaluation_results` / `evaluation_YYYYMMDD_HHMMSS.json` |
| **Visualization charts** (bar charts as PNG) | `RESULTS_DIR` / `visualizations/` |
| **Access audit log** (JSONL) | `RESULTS_DIR` / `audit` / `access_audit.jsonl` |
| **Application log** | `RESULTS_DIR` / `logs` / `rag_system.log` |

Each evaluation JSON includes a **`run_metadata`** block capturing the LLM model, embedding model, chunking settings, hybrid search config, and evaluating user role -- enabling reproducible comparisons across runs.

### Visualization

Use `--visualize` to generate the charts from evaluation results:

- **Per-query metrics** -- grouped bar chart showing P@5, R@5, faithfulness, groundedness, and answer completeness for each query
- **Summary averages** -- horizontal bar chart of all average metrics
- **Query label distribution** -- bar chart showing labeled vs unlabeled query counts
- **Run comparison** -- side-by-side bar chart comparing two evaluation runs (use with `--compare-with`)

Charts are saved as PNG files under `results/visualizations/`.

---

## Architecture

### Data Indexing

| Stage | Description | Implementation |
|-------|-------------|----------------|
| Load documents | CSV rows (`Resume_str`) -> LangChain `Document`; PDFs via multi-modal extraction | [`src/resume_rag/ingestion/loader.py`](src/resume_rag/ingestion/loader.py) (`DocumentLoader.load_dataset`) |
| Multi-modal PDF | Layout-aware extraction: text blocks, tables (as markdown), images (via vision LLM) | [`src/resume_rag/ingestion/multimodal_pdf.py`](src/resume_rag/ingestion/multimodal_pdf.py) |
| Split into chunks | Text elements split with `RecursiveCharacterTextSplitter`; tables and image descriptions pass through unsplit | `DocumentLoader.chunk_documents` |
| Embed chunks | Azure OpenAI embedding deployment | Chroma calls [`src/resume_rag/retrieval/embeddings.py`](src/resume_rag/retrieval/embeddings.py) when adding vectors |
| Store vectors | Persisted Chroma index with idempotent upsert (deduplication by `chunk_uid`) | [`src/resume_rag/retrieval/vector_store.py`](src/resume_rag/retrieval/vector_store.py) |

### Data Retrieval and Generation

| Stage | Description | Implementation |
|-------|-------------|----------------|
| User query | The question | `--query` > [`src/resume_rag/main.py`](src/resume_rag/main.py) > `RAGSystem.search` / `generate_answer` |
| Query embedding | Query text is embedded with the **same** embedding model as indexing | `VectorStore.search` > Chroma `similarity_search` (uses [`AzureEmbeddings`](src/resume_rag/retrieval/embeddings.py)) |
| Retrieval | Nearest vectors in Chroma; optional metadata filter | [`src/resume_rag/retrieval/vector_store.py`](src/resume_rag/retrieval/vector_store.py); filters from [`src/resume_rag/security/access_control.py`](src/resume_rag/security/access_control.py) |
| Top candidates | Up to `k*2` candidates (dense; optional BM25 + RRF), dedupe by resume `id`, top `k` | [`src/resume_rag/retrieval/hybrid_retriever.py`](src/resume_rag/retrieval/hybrid_retriever.py) via `RAGSystem.search` |
| Generation | In-code system + user template (`RESUME_ANSWER_*`) and `output_spec` constants in [`src/resume_rag/app/answer_generator.py`](src/resume_rag/app/answer_generator.py) (`structured_output` on/off) | `AnswerGenerator` |

If the **query embedding** request fails (e.g. HTTP 403), dense similarity search cannot run, so you get no hits (BM25-only hybrid behavior depends on index state).

### Supported System Updates

The system supports **incremental corpus updates** without rebuilding the entire vector DB or BM25 index:

- **Idempotent ingestion**: `VectorStore.add_documents` checks existing `chunk_uid` values in Chroma and skips already-indexed chunks. Re-running `load_dataset` is safe.
- **Targeted BM25 operations**: `BM25ChunkIndex.remove_by_doc_id()` removes entries for a single document ID; `upsert_documents()` replaces old entries and adds new ones. No full rebuild needed.
- **`--update-doc`**: Removes all chunks for a document ID from both Chroma and BM25, re-chunks the new text, and indexes the new chunks.
- **`--delete-docs`**: Removes chunks from Chroma and BM25 for the specified document IDs.

### Access Control

| Aspect | Description |
|--------|-------------|
| **Role-based permissions** | ADMIN (all), HR_MANAGER (read + analyze), RECRUITER (read), ANALYST (read + analyze) |
| **Category filtering** | Chroma `$in` filter + post-retrieval filtering by department categories |
| **Default policy** | Non-admin users without department or explicit categories get **deny-all** (empty set) |
| **Document-level access** | `owner_id` and `access_list` fields on document metadata for fine-grained control |
| **Audit log** | Structured JSONL audit trail at `results/audit/access_audit.jsonl` with timestamp, user, action, resource, status |

### Multi-modal PDF Extraction

PDF loading uses a tiered backend strategy in [`src/resume_rag/ingestion/multimodal_pdf.py`](src/resume_rag/ingestion/multimodal_pdf.py):

| Backend | Install | Capabilities |
|---------|---------|-------------|
| **`unstructured`** (preferred) | `uv sync --extra multimodal` | Tables as structured markdown, image extraction, text block detection |
| **`pymupdf`** (fallback) | `uv sync --extra multimodal` | Text blocks with layout awareness, image extraction as base64 |
| **`pypdf`** (built-in fallback) | Included in core deps | Plain text extraction only |

Element-type-aware chunking ensures tables and image descriptions are **not split mid-element**. When a vision-capable LLM is available, extracted images can be described automatically.

### Prompts

Answer generation ([`src/resume_rag/app/answer_generator.py`](src/resume_rag/app/answer_generator.py)). Evaluation [`prompts/`](prompts/): define **`## System Message`** and **`## Template`**; [`PromptManager.get_messages`](src/resume_rag/prompts/prompt_manager.py) builds the message list for `llm.invoke`.

| Prompt | Role |
|--------|------|
| Answer | Resume answer: `RESUME_ANSWER_SYSTEM`, `RESUME_ANSWER_USER_TEMPLATE`, and `OUTPUT_SPEC_*` in `AnswerGenerator`. |
| Evaluate | Combined eval: `EvaluationQualityScores` (faithfulness, groundedness, answer_completeness, relevance_scores). |

---

## Key Design Decisions

### Vector Database: **Chroma** (persistent, local)

| Choice | **LangChain `Chroma`** over [`chromadb`](https://github.com/chroma-core/chroma) with `persist_directory`. |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | Simple embedded store for development and demos; fits the resume CSV/PDF scale; works with LangChain `Document` + custom embeddings without running a separate server. |
| **Alternatives** | **FAISS** (in-memory or file, no built-in metadata); **pgvector / LanceDB** (stronger ops, multi-user). |

---

### Chunking Algorithm: **`RecursiveCharacterTextSplitter`**

| Choice | LangChain **`RecursiveCharacterTextSplitter`** ([`src/resume_rag/ingestion/loader.py`](src/resume_rag/ingestion/loader.py)). |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | Recursive splitting on natural boundaries (paragraphs, lines, spaces) keeps chunks readable and under a max size; overlap preserves context across cuts -- good default for unstructured resumes. Tables and image descriptions from multi-modal extraction pass through without splitting. |
| **Alternatives** | **Token-based splitters** (closer to model context limits); **semantic / heading-based** splitters for structured HTML; **fixed-size** windows (simpler, worse boundaries). |

*(YAML defaults and behavior: see [Chunking](#chunking-overview) below.)*

---

### Retrieval Ranking

| Choice | Order is **Chroma dense similarity** (and optional **BM25 + RRF** fusion when `hybrid_search.enabled` is true). |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | Fewer dependencies, lower latency and simpler ops; rank is defined by embedding similarity and optional lexical fusion. |
| **Alternatives** | A **cross-encoder** or **LLM rerank**. |

---

### GenAI Framework: **LangChain**

| Choice | **LangChain** (`langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`) for documents, Chroma, chat models. |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | LangChain gives the abstractions for loaders, splitters, vector stores, and chat. |
| **Alternatives** | **LlamaIndex** (stronger data connectors and indexing patterns); **Haystack** (pipeline-oriented search). |

---

## Chunking Overview

- **Algorithm:** LangChain **`RecursiveCharacterTextSplitter`** from `langchain_text_splitters` ([`src/resume_rag/ingestion/loader.py`](src/resume_rag/ingestion/loader.py)).
- **Configuration:** [`config/app_config.yaml`](config/app_config.yaml) > `text_splitter` > `chunk_size`, `chunk_overlap` (validated in [`src/resume_rag/domain/models.py`](src/resume_rag/domain/models.py) so `chunk_overlap` is less than `chunk_size`).
- **Defaults:** `chunk_size: 1024`, `chunk_overlap: 200` characters.
- **Behavior:** Splits text **recursively** using a hierarchy of separators (e.g. paragraphs > newlines > spaces) so segments stay under `chunk_size` while preserving structure where possible. **`chunk_overlap`** duplicates a tail of each chunk into the next so context is not lost at boundaries.
- **Multi-modal awareness:** Table and image description elements from PDF extraction are kept whole and not passed through the text splitter.

---

## Vector Database Location

| Setting | Source | Default |
|---------|--------|---------|
| Base folder for DB | `.env` > `VECTOR_DB_DIR` | `./vector-db` |
| Subfolder name | `config/app_config.yaml` > `storage.chroma_db_path` | `chroma_db` |

---

## Project structure

```
ai-rag_task/
  config/
    app_config.yaml            # Application configuration (LLM, embeddings, chunking, ACL, eval)
  dataset/
    eval_labels.json           # Gold-standard evaluation labels (15 queries)
    Resume.csv                 # Resume dataset
  prompts/
    evaluate.md                # LLM evaluation prompt (faithfulness, groundedness, completeness)
  src/resume_rag/
    app/
      rag_system.py            # Orchestrator: load, search, generate, evaluate, update, delete
      answer_generator.py      # LLM answer generation with structured output
    config/
      settings.py              # YAML + .env config loader
    domain/
      models.py                # Pydantic models (User, DocumentMetadata, EvaluationMetrics, etc.)
    evaluation/
      evaluator.py             # Precision, Recall, LLM quality scores, result persistence
    ingestion/
      loader.py                # CSV/PDF loading, chunking, evaluation query definitions
      multimodal_pdf.py        # Layout-aware PDF extraction (unstructured / pymupdf / pypdf)
      resume_text.py           # Text normalization, headline/skills extraction, BM25 tokenization
    llm/
      json_utils.py            # JSON parsing helpers
    prompts/
      prompt_manager.py        # Loads prompts/*.md into LangChain messages
    retrieval/
      bm25_index.py            # BM25Okapi index with incremental add/remove/upsert
      embeddings.py            # Custom AzureEmbeddings via openai.AzureOpenAI
      hybrid_retriever.py      # Dense + BM25 fusion (RRF / weighted)
      vector_store.py          # Chroma wrapper
    security/
      access_control.py        # Role-based permissions, category/document-level filtering, audit log
    visualization/
      charts.py                # Chart generation (per-query, summary, comparison, distribution)
    main.py                    # CLI entry point
  pyproject.toml               # Package metadata and dependencies
  .env.example                 # Environment variable template
```
