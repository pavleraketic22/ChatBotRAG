# Evaluation Proposal (Accuracy, Relevance, Latency)

This document proposes a lightweight but practical framework to evaluate the RAG chatbot.

---

## 1) Evaluation dataset

Create a gold set of **30–50 queries** split by intent:

- 12–20 coverage/exclusion questions
- 10–15 regulatory/process questions
- 8–10 policy lifecycle/terms questions
- 5–10 out-of-scope or ambiguous questions

For each query, store:
- expected source doc(s)
- expected key fact(s)
- optional expected language of answer

---

## 2) Retrieval quality metrics

Measure on top-k retrieved chunks (e.g., k=5):

- **Precision@k**
  - relevant chunks / k
- **Recall@k**
  - whether at least one gold chunk is retrieved
- **MRR (Mean Reciprocal Rank)**
  - rewards earlier rank of first relevant chunk

Suggested starting targets:
- Precision@5 >= 0.70
- Recall@5 >= 0.85
- MRR >= 0.75

---

## 3) Answer quality metrics

### A) Faithfulness (groundedness)
- Does each claim appear supported by cited evidence?
- Scoring method:
  - manual (0/1 per answer), or
  - LLM-as-judge with strict rubric

### B) Relevance
- Does answer directly address user intent?
- Score 1–5 (or pass/fail)

### C) Citation correctness
- At least one citation present
- Citations actually support the attached claim

Suggested acceptance thresholds:
- Faithfulness >= 0.90
- Relevance >= 4.0/5 average
- Citation correctness >= 0.90

---

## 4) Latency metrics

Record end-to-end per query:
- retrieval latency
- generation latency
- total latency

Report:
- **p50**, **p95** total latency

Suggested targets (depends on provider):
- OpenAI mode: p95 < 4s
- Ollama local mode: p95 < 12s (hardware dependent)

---

## 5) Error analysis loop

For failed queries, tag error class:
- retrieval miss (wrong chunks)
- partial evidence (right section, incomplete span)
- generation error (wrong interpretation)
- citation mismatch
- language mismatch

Then tune in this order:
1. retrieval (query normalization/top-k/chunking)
2. guardrail threshold
3. generation prompt
4. provider/model selection

---

## 6) Minimal evaluation process

1. Run full gold set with fixed config.
2. Export outputs + evidence.
3. Score retrieval metrics automatically.
4. Score answer/citation quality manually or with rubric.
5. Track trend over iterations.

This keeps evaluation lightweight while still actionable for product-quality improvements.
