"""Hybrid scoring engine combining structured features and semantic similarity."""

from typing import Dict, Tuple

import numpy as np

from .features import extract_all_features


# Component weights — title and skills are decisive per JD guidance
WEIGHTS = {
    "title": 0.28,
    "core_skills": 0.24,
    "semantic": 0.14,
    "career": 0.14,
    "experience": 0.08,
    "location": 0.07,
    "education": 0.05,
}


def compute_base_score(features: dict, semantic_sim: float = 0.0) -> float:
    """Weighted combination of structured features plus semantic similarity."""
    structured = (
        WEIGHTS["title"] * features["title"]
        + WEIGHTS["core_skills"] * features["core_skills"]
        + WEIGHTS["semantic"] * semantic_sim
        + WEIGHTS["career"] * features["career"]
        + WEIGHTS["experience"] * features["experience"]
        + WEIGHTS["location"] * features["location"]
        + WEIGHTS["education"] * features["education"]
    )
    return structured


def compute_final_score(features: dict, semantic_sim: float = 0.0) -> float:
    """Apply behavioral and honeypot multipliers to base score."""
    base = compute_base_score(features, semantic_sim)
    score = base * features["behavioral"] * features["honeypot"]

    # Hard gate: trap titles with high skill scores still capped
    if features["title"] < 0.15:
        score = min(score, 0.25)

    return float(np.clip(score, 0.0, 1.0))


def score_candidate(candidate: dict, semantic_sim: float = 0.0) -> Tuple[float, Dict]:
    """Score a single candidate. Returns (final_score, features_dict)."""
    features = extract_all_features(candidate)
    features["semantic"] = semantic_sim
    final = compute_final_score(features, semantic_sim)
    return final, features
