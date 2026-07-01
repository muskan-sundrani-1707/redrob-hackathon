#!/usr/bin/env python3
"""
Pre-compute TF-IDF vectors for semantic similarity scoring.

Fast, CPU-only, no network required after sklearn is installed.
Completes in ~2-3 minutes for 100K candidates.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_candidates
from src.features import build_profile_text
from src.jd_requirements import JD


def main():
    parser = argparse.ArgumentParser(description="Pre-compute TF-IDF vectors")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates from {args.candidates}...")
    ids = []
    texts = []
    for i, c in enumerate(load_candidates(args.candidates)):
        ids.append(c["candidate_id"])
        texts.append(build_profile_text(c))
        if args.limit and i + 1 >= args.limit:
            break
        if (i + 1) % 20000 == 0:
            print(f"  Loaded {i + 1} candidates...")

    # Include JD text in vocabulary fitting for better alignment
    all_texts = texts + [JD.jd_text]

    print(f"Fitting TF-IDF on {len(all_texts)} documents...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    elapsed = time.time() - t0
    print(f"TF-IDF complete in {elapsed:.1f}s, shape={tfidf_matrix.shape}")

    # Split: last row is JD, rest are candidates
    candidate_matrix = tfidf_matrix[:-1]
    jd_vector = tfidf_matrix[-1]

    with open(out_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    from scipy import sparse
    sparse.save_npz(out_dir / "candidate_tfidf.npz", candidate_matrix)
    sparse.save_npz(out_dir / "jd_tfidf.npz", jd_vector)

    with open(out_dir / "candidate_ids.json", "w") as f:
        json.dump(ids, f)

    meta = {
        "method": "tfidf",
        "num_candidates": len(ids),
        "vocab_size": len(vectorizer.vocabulary_),
        "fit_time_seconds": round(elapsed, 2),
    }
    with open(out_dir / "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
