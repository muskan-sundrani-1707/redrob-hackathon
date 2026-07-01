"""Build rich text representations and structured features from candidate profiles."""

from datetime import datetime, date
from typing import List, Tuple

from .honeypot import honeypot_penalty
from .jd_requirements import JD


PROFICIENCY_WEIGHT = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}


def build_profile_text(candidate: dict) -> str:
    """Concatenate the most informative profile fields for embedding."""
    p = candidate["profile"]
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", [])[:4]:
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}")
        parts.append(job.get("description", ""))
    for edu in candidate.get("education", [])[:2]:
        parts.append(f"{edu.get('degree', '')} {edu.get('field_of_study', '')}")
    skill_names = [s["name"] for s in candidate.get("skills", [])[:20]]
    parts.append(" ".join(skill_names))
    return " ".join(parts)


def title_fit_score(candidate: dict) -> float:
    """
    Decisive title component — catches keyword-stuffer traps.
    Returns score in [0, 1].
    """
    title = candidate["profile"]["current_title"].lower()

    for i, strong in enumerate(JD.strong_titles):
        if strong in title or title in strong:
            # Earlier entries in strong_titles are better matches
            return 0.75 + 0.25 * (1 - i / len(JD.strong_titles))

    for trap in JD.trap_titles:
        if trap in title:
            return 0.08

    # Adjacent technical roles
    adjacent = ("software engineer", "backend engineer", "data engineer", "devops", "cloud engineer")
    for adj in adjacent:
        if adj in title:
            return 0.35

    return 0.2


def core_skills_score(candidate: dict) -> Tuple[float, int, List[str]]:
    """
    Score core JD skills with trust multiplier based on endorsements and duration.
    Returns (score, matched_count, matched_skill_names).
    """
    matched = []
    weighted_sum = 0.0
    weight_total = 0.0

    for skill in candidate.get("skills", []):
        name_lower = skill["name"].lower()
        best_weight = 0.0
        for core_name, core_weight in JD.core_skills.items():
            if core_name in name_lower or name_lower in core_name:
                best_weight = max(best_weight, core_weight)

        if best_weight > 0:
            prof_w = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "intermediate"), 0.5)
            duration = skill.get("duration_months", 0) or 0
            endorsements = skill.get("endorsements", 0) or 0

            # Trust multiplier: penalize keyword stuffing
            duration_factor = min(1.0, duration / 18.0) if duration > 0 else 0.15
            endorsement_factor = min(1.0, endorsements / 15.0)
            trust = 0.4 * duration_factor + 0.3 * endorsement_factor + 0.3 * prof_w

            # Check platform assessment scores
            assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
            assess_score = assessments.get(skill["name"], assessments.get(name_lower))
            if assess_score is not None and assess_score > 0:
                trust = 0.6 * trust + 0.4 * (assess_score / 100.0)

            skill_score = best_weight * trust
            weighted_sum += skill_score
            weight_total += best_weight
            matched.append(skill["name"])

    if weight_total == 0:
        return 0.0, 0, []

    raw = min(1.0, weighted_sum / (weight_total * 0.65))
    return raw, len(matched), matched


def career_quality_score(candidate: dict) -> float:
    """Evaluate career trajectory: product vs consulting, job stability, ML evidence."""
    history = candidate.get("career_history", [])
    if not history:
        return 0.3

    score = 0.5
    companies = [h.get("company", "") for h in history]
    descriptions = " ".join(h.get("description", "") for h in history).lower()

    # Consulting-only penalty
    consulting_count = sum(
        1 for c in companies
        if any(cf in c.lower() for cf in JD.consulting_firms)
    )
    if consulting_count == len(companies):
        score -= 0.35
    elif consulting_count > 0 and consulting_count < len(companies):
        score -= 0.1

    # Product/ML evidence in career descriptions
    product_hits = sum(1 for sig in JD.product_signals if sig in descriptions)
    score += min(0.35, product_hits * 0.05)

    # Title chaser: many short stints
    short_stints = sum(1 for h in history if h.get("duration_months", 0) < 14)
    if short_stints >= 3:
        score -= 0.25
    elif short_stints >= 2:
        score -= 0.1

    # Wrong specialization penalty
    skill_names = " ".join(s["name"].lower() for s in candidate.get("skills", []))
    wrong_hits = sum(1 for w in JD.wrong_specialization if w in skill_names)
    ml_hits = sum(1 for w in ("nlp", "retrieval", "embedding", "llm", "rag") if w in skill_names)
    if wrong_hits >= 2 and ml_hits < 2:
        score -= 0.2

    # Research-only signal (no production)
    production_signals = ("production", "deployed", "shipped", "scale", "users", "a/b", "pipeline")
    if not any(sig in descriptions for sig in production_signals):
        score -= 0.1

    return max(0.0, min(1.0, score))


def experience_fit_score(candidate: dict) -> float:
    """Gaussian-like fit around ideal 5-9 years."""
    yoe = candidate["profile"].get("years_of_experience", 0)
    if yoe < 2:
        return 0.1
    if yoe < JD.min_experience:
        return 0.4 + 0.1 * (yoe / JD.min_experience)
    if yoe <= JD.ideal_experience + 2:
        # Peak around 6-8 years
        dist = abs(yoe - JD.ideal_experience)
        return max(0.6, 1.0 - dist * 0.08)
    if yoe <= JD.max_experience:
        return max(0.5, 0.85 - (yoe - JD.ideal_experience) * 0.05)
    return max(0.2, 0.6 - (yoe - JD.max_experience) * 0.08)


def location_fit_score(candidate: dict) -> float:
    """Location preference: India, Noida/Pune corridor."""
    p = candidate["profile"]
    country = p.get("country", "").lower()
    location = p.get("location", "").lower()
    signals = candidate.get("redrob_signals", {})

    if country != "india":
        return 0.25

    score = 0.6
    for loc in JD.preferred_locations:
        if loc in location:
            score = 0.95 if loc in ("noida", "pune") else 0.85
            break

    if signals.get("willing_to_relocate"):
        score = min(1.0, score + 0.1)

    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in JD.preferred_work_modes:
        score = min(1.0, score + 0.05)

    return score


def education_score(candidate: dict) -> float:
    """Light education tier bonus."""
    education = candidate.get("education", [])
    if not education:
        return 0.4

    tier_scores = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.65, "tier_4": 0.5, "unknown": 0.45}
    best = max(tier_scores.get(e.get("tier", "unknown"), 0.45) for e in education)

    cs_fields = ("computer", "software", "data", "machine learning", "ai", "information")
    has_cs = any(
        any(f in e.get("field_of_study", "").lower() for f in cs_fields)
        for e in education
    )
    if has_cs:
        best = min(1.0, best + 0.1)

    return best


def behavioral_multiplier(candidate: dict) -> float:
    """
    Behavioral signal modifier per redrob_signals doc.
    Perfect-on-paper but inactive candidates get down-weighted.
    """
    s = candidate.get("redrob_signals", {})
    mult = 1.0

    # Recruiter response rate — critical availability signal
    response = s.get("recruiter_response_rate", 0)
    if response < 0.15:
        mult *= 0.55
    elif response < 0.3:
        mult *= 0.75
    elif response >= 0.7:
        mult *= 1.08

    # Open to work
    if s.get("open_to_work_flag"):
        mult *= 1.06
    else:
        mult *= 0.92

    # Recency of activity
    last_active = s.get("last_active_date", "")
    if last_active:
        try:
            last = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (date(2026, 6, 1) - last).days
            if days_inactive > 180:
                mult *= 0.5
            elif days_inactive > 90:
                mult *= 0.75
            elif days_inactive <= 30:
                mult *= 1.05
        except ValueError:
            pass

    # Notice period — JD wants sub-30 ideal
    notice = s.get("notice_period_days", 90)
    if notice <= 30:
        mult *= 1.06
    elif notice > 60:
        mult *= 0.9

    # Engagement signals
    saved = s.get("saved_by_recruiters_30d", 0)
    if saved >= 3:
        mult *= 1.04

    interview_rate = s.get("interview_completion_rate", 0)
    if interview_rate < 0.3:
        mult *= 0.85

    # GitHub activity bonus for this technical role
    github = s.get("github_activity_score", -1)
    if github >= 40:
        mult *= 1.04
    elif github == -1:
        mult *= 0.97

    # Verification trust
    verified = sum([
        s.get("verified_email", False),
        s.get("verified_phone", False),
        s.get("linkedin_connected", False),
    ])
    mult *= 0.95 + verified * 0.02

    return max(0.3, min(1.2, mult))


def extract_all_features(candidate: dict) -> dict:
    """Extract all scoring features for a candidate."""
    skills_score, core_count, core_names = core_skills_score(candidate)
    return {
        "title": title_fit_score(candidate),
        "core_skills": skills_score,
        "core_skill_count": core_count,
        "core_skill_names": core_names,
        "career": career_quality_score(candidate),
        "experience": experience_fit_score(candidate),
        "location": location_fit_score(candidate),
        "education": education_score(candidate),
        "behavioral": behavioral_multiplier(candidate),
        "honeypot": honeypot_penalty(candidate),
    }
