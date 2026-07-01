"""Gradio sandbox for interactive candidate ranking demo."""

from pathlib import Path
import tempfile

import gradio as gr
import pandas as pd

from rank import rank_candidates


SAMPLE_PATH = Path("data/sample_100.jsonl")
FALLBACK_PATH = Path("data/candidates.jsonl")
ARTIFACTS_SAMPLE = Path("artifacts_sample")
ARTIFACTS_FULL = Path("artifacts")


def default_candidates_path() -> str:
    if SAMPLE_PATH.exists():
        return str(SAMPLE_PATH)
    if FALLBACK_PATH.exists():
        return str(FALLBACK_PATH)
    return ""


def default_artifacts_path() -> str:
    if ARTIFACTS_SAMPLE.exists():
        return str(ARTIFACTS_SAMPLE)
    if ARTIFACTS_FULL.exists():
        return str(ARTIFACTS_FULL)
    return "artifacts_sample"


def run_ranking(
    use_sample: bool,
    uploaded_file,
    candidates_path_text: str,
    artifacts_dir: str,
    top_k: int,
):
    candidates_path = candidates_path_text.strip()

    if use_sample:
        if not candidates_path:
            return pd.DataFrame(), "Sample path not found. Upload a JSONL file instead."
    else:
        if uploaded_file is None:
            return pd.DataFrame(), "Please upload a .jsonl file or enable sample mode."
        candidates_path = uploaded_file.name

    # Uploaded files can live in temporary locations; copy to stable path for read.
    if uploaded_file is not None and not use_sample:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            with open(uploaded_file.name, "rb") as src:
                tmp.write(src.read())
            candidates_path = tmp.name

    try:
        results = rank_candidates(
            candidates_path=candidates_path,
            artifacts_dir=artifacts_dir.strip(),
            top_k=int(top_k),
            limit=100 if use_sample else None,
        )
        df = pd.DataFrame(results)
        msg = f"Ranked {len(results)} candidates successfully."
        return df, msg
    except Exception as exc:
        return pd.DataFrame(), f"Ranking failed: {exc}"


with gr.Blocks(title="Redrob Candidate Ranker (Gradio)") as demo:
    gr.Markdown(
        """
# Redrob Intelligent Candidate Ranker
Rank candidates for **Senior AI Engineer — Founding Team**.
Uses title-fit gate, trusted skills, TF-IDF semantic match, career quality, behavioral signals, and honeypot detection.
"""
    )

    with gr.Row():
        use_sample = gr.Checkbox(label="Use built-in sample (100 candidates)", value=True)
        top_k = gr.Slider(minimum=10, maximum=100, value=20, step=1, label="Top K")

    uploaded = gr.File(label="Upload candidates JSONL (optional)", file_types=[".jsonl", ".json"])
    candidates_path_text = gr.Textbox(label="Candidates path", value=default_candidates_path())
    artifacts_dir = gr.Textbox(label="Artifacts directory", value=default_artifacts_path())

    run_btn = gr.Button("Rank Candidates", variant="primary")
    status = gr.Textbox(label="Status")
    output = gr.Dataframe(label="Ranked Results", wrap=True)

    run_btn.click(
        run_ranking,
        inputs=[use_sample, uploaded, candidates_path_text, artifacts_dir, top_k],
        outputs=[output, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
