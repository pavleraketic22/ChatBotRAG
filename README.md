# ChatBotRAG (LangChain + ChromaDB)

Practical RAG chatbot for car-insurance documents.

This project uses:
- **LangChain** for orchestration
- **ChromaDB** for vector storage/retrieval
- **FastEmbed** with `intfloat/multilingual-e5-large` for embeddings
- **OpenAI** (primary) + **Ollama** (fallback) for answer generation

Retrieval strategy:
- Dense retrieval from Chroma (FastEmbed vectors)
- Lightweight reranking (dense score + lexical overlap blend)

---

## 1) Prerequisites

- Python **3.11+** (you are using 3.13, that is okay)
- `pip`
- (Optional) OpenAI API key
- (Optional, recommended fallback) Ollama installed locally

---

## 2) Project setup

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3) Environment variables

Create `.env` from template:

```bash
cp .env.example .env
```

Set values in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here

# Leave empty for official OpenAI endpoint.
OPENAI_BASE_URL=

# Primary OpenAI model.
OPENAI_MODEL=gpt-4o-mini

# If primary model fails (404/access), app tries these in order.
OPENAI_FALLBACK_MODELS=gpt-4o-mini,gpt-4o,gpt-3.5-turbo

# Embeddings (free multilingual, local ONNX).
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Ollama fallback model.
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
```

### Important notes

- If `OPENAI_BASE_URL` is set to Ollama URL (`http://localhost:11434/v1`), OpenAI models will fail with 404.
- Keep `.env` private. Do not commit secrets.

---

## 4) Add documents

Place your PDF files in `data/`.

Example:

```text
data/
  Copy of mtpl_coverage.pdf
  Copy of mtpl_regulations.pdf
  Copy of user_terms_conditions_filtered.pdf
```

> PDFs are ignored in git by `.gitignore` (`data/*.pdf`).

---

## 5) Run the app

```bash
python -m streamlit run app.py
```

Then in sidebar:
1. Click **Build Index**
2. (Optional) Click **Reload Engine**
3. Select provider mode
4. Ask questions

---

## 6) Provider modes

- **Auto**
  - OpenAI (with model fallback list)
  - if unavailable -> Ollama
  - if unavailable -> extractive fallback

- **OpenAI only**
  - uses only OpenAI models

- **Ollama only**
  - uses local Ollama model

- **Extractive only**
  - no LLM synthesis, only top snippets

The app shows active config in sidebar:
- `OPENAI_MODEL=...`
- `OPENAI_BASE_URL=...`
- `OLLAMA_MODEL=...`

---

## 7) Ollama setup (fallback)

Start Ollama and pull model:

```bash
ollama serve
ollama pull gemma3:4b
```

Alternative models:
- `qwen2.5:7b` (better multilingual reasoning, more RAM)
- `mistral:7b`

---

## 8) Typical usage flow

1. Put PDFs in `data/`
2. Build index
3. Ask questions like:
   - `Where is insurance covered?`
   - `What is the maximum coverage for property damage?`
   - `Is driving under the influence of alcohol covered?`

4. Open **Retrieved evidence** panel to inspect source/page and relevance.

For formal project deliverables, see:
- `demo.md` (document-specific Q&A demonstrations)
- `evaluation.md` (accuracy/relevance/latency evaluation proposal)

---

## 9) Troubleshooting

### A) OpenAI 404 model not found

- Verify available models:

```bash
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

- Set `OPENAI_MODEL` to one from your list (e.g. `gpt-4o-mini`)
- Ensure `OPENAI_BASE_URL` is empty or official OpenAI URL

### B) Guardrail triggers too often

- Lower **Min relevance score** in UI (e.g. `0.10`)
- Increase `Top-K`
- Rebuild index

### C) Slow responses on Ollama

- Use smaller/faster model (`gemma3:4b`)
- Reduce `Top-K`
- Use `OpenAI only` for faster generation

### D) Retrieval misses due to typos

`chat.py` includes lightweight query normalization (`insurence -> insurance`, `county -> country`).

---

## 10) Files and directories

- `chat.py` - main Streamlit app (LangChain + Chroma)
- `demo.md` - demonstration queries and expected evidence-backed outcomes
- `evaluation.md` - evaluation proposal and metrics
- `data/` - source PDFs
- `chroma_db/` - local vector DB persistence
- `.env` - local secrets/config (not committed)
- `.env.example` - safe template

---
