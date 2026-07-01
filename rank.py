#!/usr/bin/env python3
"""
Rank candidates for the Senior AI Engineer role and produce submission CSV.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Requires pre-computed embeddings in artifacts/ (run precompute.py first).
Ranking step: CPU-only, no network, ≤5 minutes on 100K candidates.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data_loader import load_candidates
from src.reasoning import generate_reasoning
from src.scorer import score_candidate


def load_semantic_similarities(artifacts_dir: Path) -> dict:
    """Load pre-computed semantic similarities (TF-IDF or embedding-based)."""
    id_to_sim = {}

    tfidf_candidates = artifacts_dir / "candidate_tfidf.npz"
    tfidf_jd = artifacts_dir / "jd_tfidf.npz"
    emb_candidates = artifacts_dir / "candidate_embeddings.npy"

    if tfidf_candidates.exists() and tfidf_jd.exists():
        from scipy import sparse
        import numpy as np

        candidate_matrix = sparse.load_npz(tfidf_candidates)
        jd_vector = sparse.load_npz(tfidf_jd)
        sims = (candidate_matrix @ jd_vector.T).toarray().flatten()

        with open(artifacts_dir / "candidate_ids.json") as f:
            ids = json.load(f)
        id_to_sim = dict(zip(ids, sims.astype(float)))
        print(f"Loaded TF-IDF similarities for {len(ids)} candidates")
    elif emb_candidates.exists():
        embeddings = np.load(artifacts_dir / "candidate_embeddings.npy")
        jd_emb = np.load(artifacts_dir / "jd_embedding.npy")
        sims = embeddings @ jd_emb
        with open(artifacts_dir / "candidate_ids.json") as f:
            ids = json.load(f)
        id_to_sim = dict(zip(ids, sims.astype(float)))
        print(f"Loaded embedding similarities for {len(ids)} candidates")

    return id_to_sim


def rank_candidates(
    candidates_path: str,
    artifacts_dir: str,
    top_k: int = 100,
    limit: Optional[int] = None,
) -> List[dict]:
    """Score all candidates and return top-k ranked results."""
    artifacts = Path(artifacts_dir)

    # Load semantic similarities if available
    id_to_sim = load_semantic_similarities(artifacts)
    if not id_to_sim:
        print("WARNING: No pre-computed similarities found. Running without semantic component.")
        print("Run: python precompute.py --candidates data/candidates.jsonl")

    print(f"Scoring candidates from {candidates_path}...")
    t0 = time.time()
    scored = []

    for i, candidate in enumerate(load_candidates(candidates_path)):
        cid = candidate["candidate_id"]
        sem_sim = float(id_to_sim.get(cid, 0.0))
        final_score, features = score_candidate(candidate, sem_sim)
        scored.append({
            "candidate_id": cid,
            "score": final_score,
            "features": features,
            "candidate": candidate,
        })

        if limit and i + 1 >= limit:
            break
        if (i + 1) % 20000 == 0:
            print(f"  Scored {i + 1} candidates...")

    elapsed = time.time() - t0
    print(f"Scored {len(scored)} candidates in {elapsed:.1f}s")

    # Sort by rounded score (submission precision) then candidate_id for tie-break
    scored.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))

    results = []
    for rank, item in enumerate(scored[:top_k], start=1):
        reasoning = generate_reasoning(item["candidate"], item["features"], rank)
        results.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": round(item["score"], 4),
            "reasoning": reasoning,
        })

    return results


def write_submission(results: List[dict], out_path: str):
    """Write ranked results to submission CSV."""
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote {len(results)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob AI Engineer role")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="Limit candidates (for testing)")
    args = parser.parse_args()

    t0 = time.time()
    results = rank_candidates(
        args.candidates,
        args.artifacts,
        top_k=args.top_k,
        limit=args.limit,
    )
    write_submission(results, args.out)
    total = time.time() - t0
    print(f"Total ranking time: {total:.1f}s")

    # Preview top 10
    print("\nTop 10 candidates:")
    for r in results[:10]:
        print(f"  #{r['rank']} {r['candidate_id']} score={r['score']:.4f}")
        print(f"     {r['reasoning']}")


if __name__ == "__main__":
    main()
