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


# ======================
# CONFIG
# ======================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

OPENAI = st.secrets.get("openai", {})
OPENAI_API_KEY = OPENAI.get("api_key") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = OPENAI.get("base_url") or os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = OPENAI.get("model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

EMBEDDING_MODEL = st.secrets.get(
    "EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
)

OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "gemma3:4b"))
OLLAMA_BASE_URL = st.secrets.get("OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

ProviderMode = Literal["Auto", "OpenAI only", "Ollama only", "Extractive only"]


# ======================
# UTILS
# ======================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[\w]+\b", text.lower()) if len(t) > 1]


def lexical_overlap(q: str, d: str) -> float:
    q_set, d_set = set(tokenize(q)), set(tokenize(d))
    if not q_set or not d_set:
        return 0.0
    return len(q_set & d_set) / len(q_set)


def min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def detect_language(q: str) -> str:
    return "Hungarian" if any(ch in q.lower() for ch in "áéíóöőúüű") else "English"


# ======================
# RAG CORE
# ======================

class InsuranceRAG:
    def __init__(self) -> None:
        self.embeddings = None
        self.vectorstore = None

        # OpenAI
        self.openai_llm = None
        if OPENAI_API_KEY:
            self.openai_llm = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=0.1,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
            )

        # Ollama
        self.ollama_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    # ---------- lazy init ----------
    def get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        return self.embeddings

    def get_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                collection_name="insurance_docs",
                embedding_function=self.get_embeddings(),
                persist_directory=str(CHROMA_DIR),
            )
        return self.vectorstore

    # ---------- index ----------
    def has_index(self) -> bool:
        try:
            return int(self.get_vectorstore()._collection.count()) > 0
        except Exception:
            return False

    def build_index(self, pdf_paths: list[Path]) -> int:
        docs = []

        for pdf in pdf_paths:
            loaded = PyPDFLoader(str(pdf)).load()
            for d in loaded:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = pdf.name
                d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
            docs.extend(loaded)

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise RuntimeError("No chunks extracted.")

        vs = self.get_vectorstore()
        vs.delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.get_embeddings(),
            collection_name="insurance_docs",
            persist_directory=str(CHROMA_DIR),
        )

        return len(chunks)

    # ---------- retrieval ----------
    def retrieve(self, question: str, top_k: int = 6):
        q = question.lower()
        raw = self.get_vectorstore().similarity_search_with_score(q, k=top_k * 4)

        if not raw:
            return []

        dense = min_max([float(s) for _, s in raw])
        lex = min_max([lexical_overlap(q, d.page_content) for d, _ in raw])

        merged = []
        for i, (doc, _) in enumerate(raw):
            score = 0.6 * dense[i] + 0.4 * lex[i]
            merged.append((doc, score))

        return sorted(merged, key=lambda x: x[1], reverse=True)[:top_k]

    # ---------- generation ----------
    def generate(self, question: str, retrieved, mode: ProviderMode):

        if not retrieved:
            return "No evidence found.", "none"

        context = "\n\n".join(
            f"[{i}] {d.metadata.get('source')} p{d.metadata.get('page')}\n{d.page_content}"
            for i, (d, _) in enumerate(retrieved, 1)
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer ONLY from context. Use citations [n]. Language: {language}."),
            ("user",
             "Q: {question}\n\nContext:\n{context}")
        ])

        language = detect_language(question)

        # Extractive
        if mode == "Extractive only":
            return context[:800], "extractive"

        # OpenAI
        def openai_call():
            llm = self.openai_llm
            return (prompt | llm | StrOutputParser()).invoke({
                "question": question,
                "context": context,
                "language": language,
            })

        if mode == "OpenAI only":
            return openai_call(), "openai"

        # Auto
        if self.openai_llm:
            try:
                return openai_call(), "openai"
            except Exception:
                pass

        return (prompt | self.ollama_llm | StrOutputParser()).invoke({
            "question": question,
            "context": context,
            "language": language,
        }), "ollama"


# ======================
# STREAMLIT APP
# ======================

def list_pdfs():
    if not DATA_DIR.exists():
        return []
    return list(DATA_DIR.glob("*.pdf"))


def init():
    try:
        return InsuranceRAG(), None
    except Exception as e:
        return None, str(e)


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Chatbot", layout="wide")

    if "engine" not in st.session_state:
        st.session_state.engine, st.session_state.err = init()
        st.session_state.messages = []

    engine = st.session_state.engine

    st.title("Insurance RAG Chatbot")

    with st.sidebar:
        pdfs = list_pdfs()
        st.write("PDFs:", [p.name for p in pdfs])

        top_k = st.slider("Top-K", 3, 12, 6)
        mode = cast(ProviderMode,
                    st.radio("Mode",
                             ["Auto", "OpenAI only", "Ollama only", "Extractive only"]))

        if st.button("Build Index") and engine:
            with st.spinner("Indexing..."):
                st.success(engine.build_index(pdfs))

    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    q = st.chat_input("Ask...")

    if not q:
        return

    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").markdown(q)

    if not engine:
        st.error("Engine failed")
        return

    t0 = time.time()
    retrieved = engine.retrieve(q, top_k)

    ans, provider = engine.generate(q, retrieved, mode)

    st.chat_message("assistant").markdown(ans)
    st.caption(f"{provider} • {round((time.time()-t0)*1000)}ms")

    st.session_state.messages.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    main()