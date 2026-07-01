"""Detect honeypot candidates with impossible or inconsistent profiles."""

from datetime import datetime
from typing import List


def detect_honeypot_issues(candidate: dict) -> List[str]:
    """Return list of honeypot red flags found in a candidate profile."""
    issues = []
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])

    expert_zero_duration = 0
    expert_no_trust = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0) or 0
        endorsements = skill.get("endorsements", 0) or 0

        if prof == "expert" and duration == 0:
            expert_zero_duration += 1
            issues.append(f"expert '{skill['name']}' with 0 months experience")

        if prof == "expert" and endorsements == 0 and duration < 6:
            expert_no_trust += 1

        if prof in ("expert", "advanced") and duration == 0 and endorsements == 0:
            issues.append(f"unverified {prof} '{skill['name']}'")

    if expert_zero_duration >= 2:
        issues.append(f"{expert_zero_duration} expert skills with zero duration")

    if expert_no_trust >= 3:
        issues.append(f"{expert_no_trust} expert skills lacking endorsements")

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10:
        issues.append(f"{expert_count} skills marked expert")

    yoe = profile.get("years_of_experience", 0)
    total_months = sum(h.get("duration_months", 0) for h in history)
    if yoe > 0 and total_months > yoe * 12 + 36:
        issues.append("career timeline exceeds stated experience")

    # Short stints pattern (title chaser)
    if len(history) >= 4:
        short_stints = sum(1 for h in history if h.get("duration_months", 0) < 12)
        if short_stints >= 3:
            issues.append(f"{short_stints} roles under 12 months")

    # Impossible: many advanced AI skills but non-ML title with no career evidence
    title = profile.get("current_title", "").lower()
    ai_skill_count = sum(
        1 for s in skills
        if any(k in s["name"].lower() for k in ("llm", "embedding", "vector", "rag", "pytorch", "nlp"))
    )
    non_ml = any(t in title for t in ("marketing", "hr ", "accountant", "sales", "graphic", "civil", "mechanical"))
    if non_ml and ai_skill_count >= 6 and expert_zero_duration >= 1:
        issues.append("non-ML title with stuffed AI skills")

    return issues


def honeypot_penalty(candidate: dict) -> float:
    """
    Return multiplier in [0.05, 1.0].
    Severe honeypots get near-zero scores.
    """
    issues = detect_honeypot_issues(candidate)
    if not issues:
        return 1.0
    if len(issues) >= 3:
        return 0.05
    if len(issues) == 2:
        return 0.15
    return 0.5
