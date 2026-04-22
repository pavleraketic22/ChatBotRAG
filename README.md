# Car Insurance RAG Chatbot

Retrieval-augmented chatbot for car insurance Q&A over three PDF sources:

- MTPL Product Info
- User Regulations
- Terms & Conditions

The app uses hybrid retrieval (dense embeddings + BM25), provides citations, and handles out-of-scope questions gracefully.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy environment template:

```bash
cp .env.example .env
```

Set environment variables in `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
OPENAI_FALLBACK_MODELS=gpt-4o-mini,gpt-4o,gpt-3.5-turbo
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
```

Provider strategy:
- **Primary**: OpenAI (`OPENAI_API_KEY` required)
- **Fallback**: local Llama through Ollama
- **Last fallback**: extractive snippets (no generation)

If OpenAI model returns 404, the app automatically tries fallback models from `OPENAI_FALLBACK_MODELS`.

You can also choose provider mode directly in UI:
- `Auto`
- `OpenAI only`
- `Llama only`
- `Extractive only`

## 2) Add Knowledge Base PDFs

Create a `data/` directory and place the three PDFs there:

```text
data/
  MTPL Product Info.pdf
  User Regulations.pdf
  Terms & Conditions.pdf
```

## 3) Run App

```bash
streamlit run app.py
```

In the sidebar:
1. Click **Build Index** (runs PDF extraction/chunking/embedding)
2. Click **Reload Engine**
3. Select **Answer Provider** mode
4. Ask questions in chat

## 4) Architecture Overview

### Document processing
- PDF extraction via `pypdf`
- Text normalization (whitespace/artifact cleanup)
- Chunking with overlap to preserve context
- Metadata per chunk: `source`, `page`, `heading`, `chunk_id`

### Embedding + vector storage
- Default embedding provider: `fastembed` with `intfloat/multilingual-e5-large` (free, multilingual)
- Optional embedding provider: `openai` with `EMBEDDING_MODEL=text-embedding-3-small`
- Fallback provider: `hashing` (minimal dependency mode)
- You can configure via `.env`:
  - `EMBEDDING_PROVIDER=fastembed` (recommended free default)
  - `EMBEDDING_PROVIDER=openai` (high quality, requires API key)
  - `EMBEDDING_PROVIDER=hashing` (safe fallback)

E5 note:
- The app formats embeddings as `query: ...` and `passage: ...` for best E5 retrieval quality.
- Retrieval quality improvements included:
  - section-aware chunking (heading + body)
  - multilingual query variants for insurance intent
  - heading-aware ranking bonus
- Dense vectors saved to `index/kb_index.npz`
- Chunk metadata saved to `index/chunks.json`

### Retrieval mechanism
- **Hybrid retrieval**:
  - Dense cosine similarity over normalized embeddings
  - Sparse BM25 lexical scoring (`rank-bm25`)
- Final ranking: weighted fusion
  - `final = alpha*dense + (1-alpha)*sparse` with default `alpha=0.65`

### Response generation
- LLM synthesis via OpenAI (primary)
- Automatic fallback to local Llama via Ollama if OpenAI is unavailable
- Grounding prompt enforces: “answer only from snippets”
- Mandatory citation format `[n]` tied to retrieved snippets
- Structured answer protocol: model returns JSON claims with citation indexes, then app validates each claim against cited snippets before rendering final bullets

### Out-of-scope handling
- Confidence gate on top retrieval score
- Low-score queries return a graceful “not found in provided docs” response

## 5) Demonstration Queries (document-specific)

Run these after indexing to verify retrieval by source:

1. **MTPL Product Info**
   - “What risks are covered under MTPL and what are the key exclusions?”
2. **User Regulations**
   - “What are the policyholder’s obligations when reporting a claim?”
3. **Terms & Conditions**
   - “Under which conditions can the insurer deny a payout?”

Expected behavior:
- Answers include source citations like `[1] [2]`
- Evidence panel shows page-level chunk provenance
- Evidence panel also shows retrieval query variants for debugging
- Optional: enable **Translate evidence snippets** in sidebar to view evidence in question language

## 6) Evaluation Proposal

Use a small gold dataset (20–50 queries) with known expected answers and source locations.

### Retrieval metrics
- **Precision@k**: fraction of top-k chunks that are relevant
- **MRR**: rank quality of first relevant chunk
- **Recall@k**: whether at least one gold chunk appears in top-k

### Generation metrics
- **Answer faithfulness** (LLM-as-judge or human): does answer stay within retrieved evidence?
- **Answer relevance**: does answer address user intent?
- **Citation correctness**: do cited snippets support each claim?

### Performance metrics
- End-to-end latency (p50/p95)
- Retrieval latency vs generation latency breakdown

### Suggested acceptance thresholds
- Precision@5 >= 0.7
- MRR >= 0.75
- p95 latency < 3s (local embedding cache + fast model)

## 7) Notes / Next Improvements

- Add cross-encoder reranker for improved top-k ordering
- Add multilingual embeddings if documents/questions are multilingual
- Add automated evaluation script with CSV test set
- Add conversation memory with query rewriting for follow-up questions

## 8) Llama Fallback Setup (Ollama)

Install Ollama and pull a model (recommended: `gemma3:4b`):

```bash
ollama serve
ollama pull gemma3:4b
```

Then run the Streamlit app normally. If OpenAI fails/unavailable, the app automatically uses Ollama.

Alternative Ollama models for multilingual quality:
- `qwen2.5:7b` (better multilingual reasoning, needs more RAM)
- `mistral:7b` (solid general fallback)

## 9) Multilingual (Hungarian) Troubleshooting

If answers incorrectly return "not found":

1. Rebuild index after model change (important):
   - delete `index/` folder
   - click **Build Index** again
2. In UI lower **Out-of-scope threshold** (e.g. `0.05`)
3. Increase **Top-K passages** (e.g. `8`–`10`)
4. Try asking in the same language as the PDF (Hungarian)

Reason: multilingual text benefits from robust tokenization and less strict OOS filtering. If quality is still low, switch to `EMBEDDING_PROVIDER=openai` and rebuild index.

## 10) Fix for "meta tensor" error

If you saw an error like:

`Cannot copy out of meta tensor; no data!`

it came from the local `sentence-transformers/torch` stack in your environment. The app now defaults to `EMBEDDING_PROVIDER=fastembed` (ONNX-based), which avoids that issue.

After pulling latest changes:

1. Delete old index folder: `rm -rf index`
2. Rebuild index in app (**Build Index**)
3. Reload engine

If you want higher retrieval quality later, set `EMBEDDING_PROVIDER=openai` in `.env` and rebuild index.
