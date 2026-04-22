# Demonstrations: Document-Specific Queries

Use the following queries after indexing the 3 PDFs in `data/`.

## How to run demos

1. Start app:

```bash
python -m streamlit run app.py
```

2. In UI:
   - **Build Index**
   - Provider: `Auto` (or `OpenAI only`)
   - Top-K: `6` to `8`
   - Min relevance score: `0.10` to `0.15`

3. Ask each demo query and verify:
   - answer contains citations like `[1] [2]`
   - retrieved evidence includes expected document/section context

---

## Demo set A — Coverage scope and limits (MTPL coverage doc)

### Q1
**Query:** `Where is insurance covered?`

**Expected outcome:**
- Answer should mention territorial scope (EEA + Switzerland + Green Card system).
- Evidence should come from `Copy of mtpl_coverage.pdf`.

### Q2
**Query:** `What is the maximum coverage for property damage?`

**Expected outcome:**
- Answer should cite the property-damage cap from MTPL product summary.
- Evidence should come from `Copy of mtpl_coverage.pdf` lines around "Mire terjed ki a biztosítás?".

### Q3
**Query:** `What is not covered by insurance?`

**Expected outcome:**
- Answer should list exclusions (e.g., own vehicle damage of liable party, specific excluded scenarios).
- Evidence should point to exclusion section from `Copy of mtpl_coverage.pdf` and/or regulations.

---

## Demo set B — Regulatory obligations and claims process (MTPL regulations)

### Q4
**Query:** `What are my obligations when a claim occurs?`

**Expected outcome:**
- Answer mentions reporting obligations and timing constraints.
- Evidence should reference `Copy of mtpl_regulations.pdf` claim notification sections.

### Q5
**Query:** `When can the insurer seek recourse from the driver?`

**Expected outcome:**
- Answer should include recourse conditions (e.g., DUI/alcohol, no valid license, unauthorized use, intentional harm).
- Evidence should come from recourse paragraphs in `Copy of mtpl_regulations.pdf`.

### Q6
**Query:** `Is driving under the influence of alcohol covered?`

**Expected outcome:**
- Answer should clearly indicate unfavorable coverage outcome from policyholder perspective and mention insurer recourse context.
- Evidence should contain keywords like `alkoholos`, `visszkereset` or equivalent wording in regulations/coverage doc.

---

## Demo set C — Terms & Conditions and policy behavior

### Q7
**Query:** `When does insurance coverage terminate?`

**Expected outcome:**
- Answer should mention termination scenarios (sale, policy anniversary cancellation, fixed-term expiry, non-payment grace period).
- Evidence typically from `Copy of mtpl_coverage.pdf` lifecycle section.

### Q8
**Query:** `How can I cancel the policy?`

**Expected outcome:**
- Answer should mention cancellation window/rules (e.g., before anniversary).
- Evidence from cancellation section in coverage summary.

### Q9
**Query:** `What personal data is processed during claim handling?`

**Expected outcome:**
- Answer should reflect relevant data categories and legal basis references.
- Evidence should come from `Copy of user_terms_conditions_filtered.pdf`.

---

## Acceptance criteria for demo success

Each demo query is considered successful if:

1. The answer is directly relevant to the question.
2. At least one citation is present.
3. Retrieved evidence includes the appropriate source document.
4. No obvious contradiction with cited snippet text.
