# Slide 1 — Cover

- **Team Name:** IntelliRank
- **Problem Statement:** Build an AI system that ranks candidates for a Senior AI Engineer role by true role fit, not keyword count.
- **Team Leader Name:** Muskan Sundrani

---

# Slide 2 — Solution Overview

## What is your proposed solution?
IntelliRank is a hybrid candidate ranking system that combines structured profile scoring (title fit, trusted skill evidence, career quality, experience, location, education), semantic relevance (TF-IDF similarity to JD intent), behavioral readiness signals, and profile consistency checks.

## What differentiates your approach from traditional candidate matching systems?
- Traditional systems over-rely on keyword overlap.
- IntelliRank uses **role-fit reasoning**:
  - Hard penalties for title mismatch traps (e.g., non-ML titles with stuffed AI keywords)
  - Skill trust scoring from duration + endorsements + proficiency
  - Career narrative quality and production ML evidence
  - Behavioral availability (response rate, notice period, open-to-work)

---

# Slide 3 — JD Understanding & Candidate Evaluation

## Key requirements extracted from the JD
- Role: Senior AI Engineer (Founding Team)
- Experience sweet spot: ~5–9 years
- Core capability: retrieval, embeddings, vector search, ranking/evaluation
- Practical orientation: product-company ML production > pure research
- Location preference: India (Noida/Pune preferred)
- Recruitability: active candidates, better response behavior, shorter notice period

## Most important candidate signals
- Current title and title history relevance
- Trusted core skills: embeddings, FAISS/Pinecone/Qdrant/pgvector, semantic search, ranking
- Career evidence: shipped/deployed/production systems
- Platform behavior: response rate, activity recency, open-to-work, notice period
- Consistency checks: suspicious timelines, unrealistic expert claims

---

# Slide 4 — Ranking Methodology

## How does the system retrieve, score, and rank?
1. Load candidate profiles from JSONL.
2. Build profile text and compute semantic similarity against JD vector (precomputed TF-IDF).
3. Extract structured features per candidate.
4. Compute weighted base score.
5. Apply behavioral and honeypot multipliers.
6. Sort by score (rounded to submission precision), then candidate_id for tie-break.
7. Return top-100 with reasoning.

## Models, algorithms, and heuristics used
- **TF-IDF (scikit-learn)** for semantic relevance
- **Weighted scoring engine** for interpretable hybrid ranking
- **Heuristic gates/penalties** for title traps and suspicious profiles

## Signal combination formula
Final Score = BaseWeightedScore x BehavioralMultiplier x HoneypotMultiplier

---

# Slide 5 — Explainability & Data Validation

## How are ranking decisions explained?
Each shortlisted candidate gets a concise reasoning string with:
- title + years of experience
- top matching retrieval/embedding/ranking skills
- career quality context
- behavioral strengths/concerns

## How do you prevent hallucinations/unsupported justifications?
- Reasoning is generated from computed features only.
- No external LLM inference used in ranking-time.
- Every statement maps to profile fields or derived feature outputs.

## How do you handle low-quality or suspicious profiles?
- Honeypot multiplier penalizes:
  - expert skills with 0 months
  - timeline inconsistencies
  - non-ML titles with heavy AI keyword stuffing
- Title hard-cap prevents fake top ranking from keyword stuffing alone.

---

# Slide 6 — End-to-End Workflow

1. Input JD requirements (structured constants + JD narrative).
2. Offline precompute step (`precompute.py`) builds TF-IDF artifacts.
3. Online ranking step (`rank.py`) processes all candidates on CPU.
4. Features + semantic signals -> hybrid scoring.
5. Top 100 ranked candidates generated.
6. Submission file exported in required format:
   `candidate_id, rank, score, reasoning`
7. Validation via `validate_submission.py`.

---

# Slide 7 — System Architecture

- **Data Layer**
  - `data/candidates.jsonl`
  - `artifacts/` TF-IDF vectors + candidate IDs

- **Feature Layer**
  - `src/features.py`
  - title, skills, experience, location, education, behavioral, career signals

- **Intelligence Layer**
  - `src/scorer.py` (hybrid weighted score + multipliers)
  - `src/honeypot.py` (suspicious profile penalties)
  - `src/reasoning.py` (human-readable explanations)

- **Execution Layer**
  - `precompute.py` for offline vector prep
  - `rank.py` for final ranking + CSV output
  - `validate_submission.py` for spec compliance

---

# Slide 8 — Results & Performance

## Ranking quality highlights
- Top candidates are consistently ML/AI-focused titles.
- Strong alignment to JD-required retrieval/embedding/vector skills.
- Behavioral readiness and notice-period concerns are surfaced in reasoning.
- Trap profiles are effectively pushed down by title gate + trust penalties.

## Runtime and compute constraints
- Precompute: ~2 minutes (offline, CPU)
- Ranking: ~15 seconds for 100K candidates (CPU only)
- Network: off during ranking
- Resource profile: compatible with challenge constraints

---

# Slide 9 — Technologies Used

- **Python 3.9** — fast, reliable ML scripting ecosystem
- **NumPy / SciPy** — efficient numeric and sparse operations
- **scikit-learn** — TF-IDF vectorization and sparse similarity
- **Pandas** — result shaping and quick analysis
- **Streamlit / Gradio** — sandbox/demo interfaces
- **Git + Hugging Face Spaces** — reproducible hosting and submission demo

Why selected:
- Lightweight, interpretable, CPU-friendly stack
- Fast iteration and reproducibility under strict runtime constraints

---

# Slide 10 — Submission Assets

- **GitHub Repository:** `https://github.com/muskansundrani/redrob-hackathon`
- **Sandbox Link (HF Space):** `https://huggingface.co/spaces/muskansundrani17/retdrob-ranker`
- **Ranked Output File:** `submission.csv` (rename to participant ID at submission)
- **Approach Deck PDF:** (attach your final customized PDF)
- **Metadata:** `submission_metadata.yaml`

---

# Slide 11 — Closing Slide

## Final Note
IntelliRank improves recruiter trust by ranking candidates on real role fit, not surface-level keyword overlap.

## Next Improvements
- Learning-to-rank with labeled relevance tiers
- Cross-encoder re-ranking for top-N refinement
- Richer longitudinal career trajectory modeling

## Thank You
Muskan Sundrani  
IntelliRank — Redrob Data & AI Challenge
