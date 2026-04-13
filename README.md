# RAG Task

### Context

A **RAG** system for **searching and answering questions over a dataset of resumes** (HR / recruiting / etc.). It indexes resume text, retrieves the most relevant profiles for a user question and generates a short answer with the **role-based access** to job categories.

The system supports teams that need to **explore a structured resume approach without manually reading every file**. A recruiter or analyst asks questions; the system finds relevant resume chunks, ranks them and produces an **LLM-generated answer. Access control** limits which job categories each user role may see.

---

## Getting started

### Prerequisites

- **Python 3.11+**
- **UV package manager**
- **Azure OpenAI–compatible API**: base URL and API key (see `.env.example`). Need working deployments for **chat** and **embeddings** matching names in [`config/app_config.yaml`](config/app_config.yaml) (`llm.deployment_name`, `embeddings.deployment_name`).
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
- **Index once**, then use **`--skip-load`** for queries, evaluation, or doc maintenance so you do not re-append duplicate vectors.

### Common workflows

```bash
# Initial index from CSV (including: add PDFs per row under dataset/<Category>/<ID>.pdf)
uv run python -m resume_rag.main --csv-file Resume.csv
uv run python -m resume_rag.main --csv-file Resume.csv --load-pdfs

# Query against an existing index
uv run python -m resume_rag.main --skip-load --query "Who is Accountant?"

# Evaluation (uses built-in eval queries; writes JSON under results/)
uv run python -m resume_rag.main --skip-load --evaluate

# Same as above, with a role that affects category filtering (admin | hr | recruiter | analyst)
uv run python -m resume_rag.main --skip-load --query "Financial analyst" --user-type hr

# Alternate config file (path relative to repo root unless absolute)
uv run python -m resume_rag.main --config custom_config.yaml --csv-file Resume.csv
```

### Document Maintenance

```bash
# Delete logical documents by metadata id (comma-separated)
uv run python -m resume_rag.main --skip-load --delete-docs "id1,id2"

# Replace one document’s text (file or inline), then re-chunk and re-index
uv run python -m resume_rag.main --skip-load --update-doc RESUME_ID --update-file /path/to/new_text.txt
uv run python -m resume_rag.main --skip-load --update-doc RESUME_ID --update-text "New resume body..." \
  --update-category INFORMATION-TECHNOLOGY --update-source csv
```

### Flags (summary)

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML app config (default `config/app_config.yaml`) |
| `--csv-file` | CSV name inside `DATA_DIR` (default `Resume.csv`) |
| `--load-pdfs` | Also load `dataset/<Category>/<ID>.pdf` when present |
| `--skip-load` | Do not reload CSV/PDF into the vector index |
| `--query` | Run a single search + answer |
| `--evaluate` | Run the evaluation suite |
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
3. Computes **Precision@K** and **Recall@K** for K in `precision_k_values` / `recall_k_values` (default 1, 3, 5, 10), and one LLM structured call for **faithfulness**, **groundedness**, and **per-excerpt relevance** (mean → `avg_relevance`) via [`prompts/evaluate.md`](prompts/evaluate.md).

### Saved artifacts

| Output | Location |
|--------|----------|
| **Per-run JSON** (summary + per-query metrics) | `RESULTS_DIR` / `evaluation_results` / `evaluation_YYYYMMDD_HHMMSS.json` (defaults: [`results/evaluation_results/`](results/evaluation_results/)) |
| **Application log** | `RESULTS_DIR` / `logs` / `rag_system.log` (default: [`results/logs/`](results/logs/)) |

K values and evaluation limits are configured under **`evaluation:`** in [`config/app_config.yaml`](config/app_config.yaml).

---

### Data indexing

| Stage  |  Description   | Implementation |
|--------|----------------|----------------|
| Load documents | CSV rows (`Resume_str`) > LangChain `Document`; PDFs per row | [`src/resume_rag/ingestion/loader.py`](src/resume_rag/ingestion/loader.py) (`DocumentLoader.load_dataset`) |
| Split into chunks | Text split before embedding | `RecursiveCharacterTextSplitter.split_documents` |
| Embed chunks | Azure OpenAI embedding deployment | Chroma calls [`src/resume_rag/retrieval/embeddings.py`](src/resume_rag/retrieval/embeddings.py) when adding vectors |
| Store vectors | Persisted Chroma index | [`src/resume_rag/retrieval/vector_store.py`](src/resume_rag/retrieval/vector_store.py) (`VectorStore.add_documents` via `RAGSystem.load_dataset` in [`src/resume_rag/app/rag_system.py`](src/resume_rag/app/rag_system.py)) |

### Data retrieval and generation

|  Stage |  Description   | Implementation |
|--------|----------------|----------------|
| User query | The question | `--query` > [`src/resume_rag/main.py`](src/resume_rag/main.py) > `RAGSystem.search` / `generate_answer` |
| Query embedding | Query text is embedded with the **same** embedding model as indexing | `VectorStore.search` > Chroma `similarity_search` (uses [`AzureEmbeddings`](src/resume_rag/retrieval/embeddings.py)) |
| Retrieval | Nearest vectors in Chroma; optional metadata filter | [`src/resume_rag/retrieval/vector_store.py`](src/resume_rag/retrieval/vector_store.py); filters from [`src/resume_rag/security/access_control.py`](src/resume_rag/security/access_control.py) |
| Top candidates | Up to `k*2` candidates (dense; optional BM25 + RRF), dedupe by resume `id`, top `k` | [`src/resume_rag/retrieval/hybrid_retriever.py`](src/resume_rag/retrieval/hybrid_retriever.py) via `RAGSystem.search` |
| Generation | In-code system + user template (`RESUME_ANSWER_*`) and `output_spec` constants in [`src/resume_rag/app/answer_generator.py`](src/resume_rag/app/answer_generator.py) (`structured_output` on/off) | `AnswerGenerator` |

If the **query embedding** request fails (e.g. HTTP 403), dense similarity search cannot run, so you get no hits (BM25-only hybrid behavior depends on index state).

### Access Control / Evaluation / Retrieval

- **Access control:** category filters and result filtering by role/department ([`src/resume_rag/security/access_control.py`](src/resume_rag/security/access_control.py)).
- **Evaluation:** metrics pipeline ([`src/resume_rag/evaluation/evaluator.py`](src/resume_rag/evaluation/evaluator.py)).
- **Hybrid retrieval / structured answers:** see [`config/app_config.yaml`](config/app_config.yaml) (`hybrid_search`, `structured_output`).

### Prompts

Answer generation ([`src/resume_rag/app/answer_generator.py`](src/resume_rag/app/answer_generator.py)). Evaluation [`prompts/`](prompts/): define **`## System Message`** and **`## Template`**; [`PromptManager.get_messages`](src/resume_rag/prompts/prompt_manager.py) builds the message list for `llm.invoke`.

|    Prompt   | Role |
|-------------|------|
|  Answer  | Resume answer: `RESUME_ANSWER_SYSTEM`, `RESUME_ANSWER_USER_TEMPLATE`, and `OUTPUT_SPEC_*` in `AnswerGenerator`. |
| Evaluate | Combined eval: `EvaluationQualityScores` (faithfulness, groundedness, relevance_scores). |

---

## Key Design Decisions

### Vector database: **Chroma** (persistent, local)

| Choice  | **LangChain `Chroma`** over [`chromadb`](https://github.com/chroma-core/chroma) with `persist_directory`. |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | Simple embedded store for development and demos; fits the resume CSV/PDF scale; works with LangChain `Document` + custom embeddings without running a separate server. |
| **Alternatives** | **FAISS** (in-memory or file, no built-in metadata); **pgvector / LanceDB** (stronger ops, multi-user). |

---

### Chunking Algorithm: **`RecursiveCharacterTextSplitter`**

| Choice | LangChain **`RecursiveCharacterTextSplitter`** ([`src/resume_rag/ingestion/loader.py`](src/resume_rag/ingestion/loader.py)). |
|--------|-------------------------------------------------------------------------------------------------------|
| **Why** | Recursive splitting on natural boundaries (paragraphs, lines, spaces) keeps chunks readable and under a max size; overlap preserves context across cuts—good default for unstructured resumes. |
| **Alternatives** | **Token-based splitters** (closer to model context limits); **semantic / heading-based** splitters for structured HTML; **fixed-size** windows (simpler, worse boundaries). |

*(YAML defaults and behavior: see [Chunking](#chunking) below.)*

---

### Retrieval ranking

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

---

## Vector database location

| Setting |  Source |  Default |
|---------|---------|----------|
| Base folder for DB | `.env` > `VECTOR_DB_DIR` | `./vector-db` |
| Subfolder name | `config/app_config.yaml` > `storage.chroma_db_path` | `chroma_db` |

---