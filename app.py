"""Streamlit sandbox for interactive candidate ranking demo."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rank import rank_candidates

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("Redrob Intelligent Candidate Ranker")
st.markdown(
    "Rank candidates for the **Senior AI Engineer — Founding Team** role. "
    "Uses title-fit gate, trusted skills, TF-IDF semantic match, career quality, "
    "behavioral signals, and honeypot detection."
)

SAMPLE_PATH = Path("data/sample_100.jsonl")
FALLBACK_PATH = Path("data/candidates.jsonl")
ARTIFACTS_SAMPLE = Path("artifacts_sample")
ARTIFACTS_FULL = Path("artifacts")


def resolve_paths():
    if SAMPLE_PATH.exists():
        return str(SAMPLE_PATH), str(ARTIFACTS_SAMPLE if ARTIFACTS_SAMPLE.exists() else ARTIFACTS_FULL)
    if FALLBACK_PATH.exists():
        return str(FALLBACK_PATH), str(ARTIFACTS_FULL)
    return None, str(ARTIFACTS_SAMPLE if ARTIFACTS_SAMPLE.exists() else ARTIFACTS_FULL)


default_candidates, default_artifacts = resolve_paths()

col1, col2 = st.columns(2)

with col1:
    use_sample = st.checkbox("Use built-in sample (100 candidates)", value=True)
    uploaded = st.file_uploader("Or upload candidates.jsonl", type=["jsonl", "json"])

with col2:
    top_k = st.slider("Top K results", min_value=10, max_value=100, value=20)
    artifacts_dir = st.text_input("Artifacts directory", value=default_artifacts)

if st.button("Rank Candidates", type="primary"):
    with st.spinner("Ranking..."):
        if use_sample:
            if not default_candidates:
                st.error("Sample data not found. Add data/sample_100.jsonl to the repo.")
                st.stop()
            results = rank_candidates(
                default_candidates,
                artifacts_dir,
                top_k=top_k,
                limit=100,
            )
        elif uploaded:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
                content = uploaded.read()
                if uploaded.name.endswith(".json"):
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            tmp.write((json.dumps(item) + "\n").encode())
                    else:
                        tmp.write((json.dumps(data) + "\n").encode())
                else:
                    tmp.write(content)
                tmp_path = tmp.name
            results = rank_candidates(tmp_path, artifacts_dir, top_k=top_k)
        else:
            st.warning("Select sample or upload a file.")
            st.stop()

        st.success(f"Ranked {len(results)} candidates")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="ranking_output.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    "**Reproduce full ranking:** "
    "`python precompute.py --candidates data/candidates.jsonl && "
    "python rank.py --candidates data/candidates.jsonl --out submission.csv`"
)
