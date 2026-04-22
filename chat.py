from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Literal, cast

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-4o-mini,gpt-4o,gpt-3.5-turbo").split(",")
    if m.strip()
]
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

ProviderMode = Literal["Auto", "OpenAI only", "Ollama only", "Extractive only"]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"\b[\w]+\b", text.lower(), flags=re.UNICODE) if len(tok) > 1]


def lexical_overlap(query: str, doc_text: str) -> float:
    q = set(tokenize(query))
    d = set(tokenize(doc_text))
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-12:
        return [1.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def detect_language(question: str) -> str:
    q = question.lower()
    if any(ch in q for ch in "áéíóöőúüű"):
        return "Hungarian"
    hu_words = {"hol", "mikor", "hogyan", "biztosítás", "fedezet", "kár", "érvényes"}
    if hu_words.intersection(set(re.findall(r"\w+", q, flags=re.UNICODE))):
        return "Hungarian"
    return "English"


class InsuranceRAG:
    def __init__(self) -> None:
        self.embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            collection_name="insurance_docs",
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        self.openai_llm = None
        if os.getenv("OPENAI_API_KEY"):
            kwargs = {"model": OPENAI_MODEL, "temperature": 0.1}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self.openai_llm = ChatOpenAI(**kwargs)

        self.ollama_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    def has_index(self) -> bool:
        try:
            return int(self.vectorstore._collection.count()) > 0  # noqa: SLF001
        except Exception:
            return False

    def build_index(self, pdf_paths: list[Path]) -> int:
        docs = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            for d in loaded:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = pdf_path.name
                d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
            docs.extend(loaded)

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise RuntimeError("No chunks extracted from PDFs.")

        self.vectorstore.delete_collection()
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="insurance_docs",
            persist_directory=str(CHROMA_DIR),
        )
        return len(chunks)

    @staticmethod
    def _normalize_query(question: str) -> str:
        q = question.strip().lower()
        replacements = {
            "insurence": "insurance",
            "insurence?": "insurance?",
            "county": "country",
        }
        for src, dst in replacements.items():
            q = q.replace(src, dst)

        if "where is" in q and "insurance" in q and "covered" in q:
            q += " valid countries territorial scope europe switzerland green card"
        return q

    def retrieve(self, question: str, top_k: int = 6) -> list[tuple]:
        q = self._normalize_query(question)
        # Use raw score API and normalize ourselves.
        candidate_k = min(max(top_k * 4, top_k), 32)
        raw = self.vectorstore.similarity_search_with_score(q, k=candidate_k)
        if not raw:
            return []

        raw_scores = [float(score) for _, score in raw]
        dense_n = min_max(raw_scores)

        overlap_scores = [lexical_overlap(q, doc.page_content) for doc, _ in raw]
        overlap_n = min_max(overlap_scores)

        dense_w = 0.6
        lexical_w = 0.4

        merged = []
        for i, (doc, _) in enumerate(raw):
            final = dense_w * dense_n[i] + lexical_w * overlap_n[i]
            merged.append((doc, final))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

    @staticmethod
    def _extractive_answer(question: str, retrieved: list[tuple]) -> str:

        bullets = []
        for i, (doc, _) in enumerate(retrieved[:3], start=1):
            snippet = doc.page_content[:260].strip()
            if len(doc.page_content) > 260:
                snippet += "..."
            bullets.append(f"- {snippet} [{i}]")

        return (
            f"Extractive evidence for: '{question}'\n"
            + "\n".join(bullets)
            + "\n\n(Enable OpenAI or Ollama mode for synthesized answer.)"
        )

    def generate(self, question: str, retrieved: list[tuple], provider_mode: ProviderMode) -> tuple[str, str]:
        if not retrieved:
            return "No evidence retrieved.", "none"

        context_lines = []
        for i, (doc, score) in enumerate(retrieved, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            context_lines.append(f"[{i}] source={source}; page={page}; score={score:.3f}\n{doc.page_content}")
        context = "\n\n".join(context_lines)

        language = detect_language(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful insurance assistant. Answer ONLY from provided snippets. "
                    "If unsure, say so. Every claim must include citation [n]. "
                    "Do not mix languages. Respond only in {language}.",
                ),
                (
                    "user",
                    "Question:\n{question}\n\nContext:\n{context}\n\n"
                    "Return concise bullets with citations [n].",
                ),
            ]
        )

        if provider_mode == "Extractive only":
            return self._extractive_answer(question, retrieved), "extractive"

        def invoke_openai_with_fallback() -> tuple[str, str]:
            candidate_models = [OPENAI_MODEL] + [m for m in OPENAI_FALLBACK_MODELS if m != OPENAI_MODEL]
            last_error = "unknown"
            for model_name in candidate_models:
                try:
                    kwargs = {"model": model_name, "temperature": 0.1}
                    if OPENAI_BASE_URL:
                        kwargs["base_url"] = OPENAI_BASE_URL
                    llm = ChatOpenAI(**kwargs)
                    text = (prompt | llm | StrOutputParser()).invoke(
                        {"question": question, "context": context, "language": language}
                    )
                    provider = "openai" if model_name == OPENAI_MODEL else f"openai-fallback:{model_name}"
                    return text, provider
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)

            raise RuntimeError(last_error)

        if provider_mode == "OpenAI only":
            if self.openai_llm is None:
                return "OPENAI_API_KEY is missing.", "openai-unavailable"
            try:
                return invoke_openai_with_fallback()
            except Exception as exc:
                return (
                    "OpenAI request failed. The configured model may be unavailable for your account. "
                    f"Current OPENAI_MODEL='{OPENAI_MODEL}', OPENAI_BASE_URL='{OPENAI_BASE_URL or 'default'}'. "
                    f"Error: {exc}. "
                    "Set a valid OPENAI_MODEL in .env (for example a model available in your project), "
                    "or switch provider to Auto/Ollama.",
                    "openai-error",
                )

        if provider_mode == "Ollama only":
            return (prompt | self.ollama_llm | StrOutputParser()).invoke(
                {"question": question, "context": context, "language": language}
            ), "ollama"

        if self.openai_llm is not None:
            try:
                return invoke_openai_with_fallback()
            except Exception:
                pass

        try:
            return (prompt | self.ollama_llm | StrOutputParser()).invoke(
                {"question": question, "context": context, "language": language}
            ), "ollama-fallback"
        except Exception:
            return self._extractive_answer(question, retrieved), "extractive-fallback"


def list_pdfs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def init_engine() -> tuple[InsuranceRAG | None, str | None]:
    try:
        engine = InsuranceRAG()
        return engine, None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Insurance ChatBot", layout="wide")
    st.title("Insurance ChatBot")

    if "engine" not in st.session_state:
        st.session_state["engine"], st.session_state["engine_error"] = init_engine()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    engine: InsuranceRAG | None = st.session_state["engine"]

    with st.sidebar:
        st.header("Knowledge Base")
        pdfs = list_pdfs()
        if pdfs:
            for p in pdfs:
                st.write(f"- {p.name}")
        else:
            st.warning(f"No PDFs found in {DATA_DIR}")

        top_k = st.slider("Top-K", 3, 12, 6)
        min_score = st.slider("Min relevance score", 0.0, 1.0, 0.15, 0.01)
        provider_mode = cast(
            ProviderMode,
            st.radio(
                "Provider",
                options=["Auto", "OpenAI only", "Ollama only", "Extractive only"],
                index=0,
            ),
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build Index"):
                if engine is None:
                    st.error("Engine init failed.")
                elif not pdfs:
                    st.error("Add PDFs to data/ first.")
                else:
                    with st.spinner("Building index..."):
                        count = engine.build_index(pdfs)
                    st.success(f"Index built with {count} chunks.")

        with col2:
            if st.button("Reload Engine"):
                st.session_state["engine"], st.session_state["engine_error"] = init_engine()
                st.success("Engine reloaded.")

        st.markdown("---")
        st.caption(
            f"OPENAI_MODEL={OPENAI_MODEL} | OPENAI_BASE_URL={OPENAI_BASE_URL or 'default'} | OLLAMA_MODEL={OLLAMA_MODEL}"
        )

    if st.session_state.get("engine_error"):
        st.error(f"Engine init error: {st.session_state['engine_error']}")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about coverage, exclusions, claims...")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if engine is None:
        text = "Engine is unavailable. Please Reload Engine."
        with st.chat_message("assistant"):
            st.error(text)
        st.session_state["messages"].append({"role": "assistant", "content": text})
        return

    if not engine.has_index():
        text = "Index is empty. Add PDFs and click Build Index."
        with st.chat_message("assistant"):
            st.error(text)
        st.session_state["messages"].append({"role": "assistant", "content": text})
        return

    started = time.perf_counter()
    retrieved = engine.retrieve(question, top_k=top_k)

    if not retrieved or retrieved[0][1] < min_score:
        answer = (
            "I couldn't find reliable evidence for that question in the provided documents. "
            "Please rephrase or ask a more document-specific question."
        )
        provider = "guardrail"
    else:
        answer, provider = engine.generate(question, retrieved, provider_mode=provider_mode)

    latency_ms = (time.perf_counter() - started) * 1000

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"Latency: {latency_ms:.0f} ms • Provider: {provider}")
        with st.expander("Retrieved evidence"):
            for i, (doc, score) in enumerate(retrieved, start=1):
                source = str(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] {source} — page {page}**  \nscore={score:.3f}")
                st.write(doc.page_content)

    st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
