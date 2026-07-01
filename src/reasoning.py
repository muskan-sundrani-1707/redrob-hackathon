"""Generate specific, honest reasoning strings for ranked candidates."""

from .jd_requirements import JD


def _format_yoe(yoe: float) -> str:
    return f"{yoe:.1f}"


def generate_reasoning(candidate: dict, features: dict, rank: int) -> str:
    """
    Produce a 1-2 sentence justification referencing real profile facts.
    Tone should match rank position.
    """
    p = candidate["profile"]
    s = candidate["redrob_signals"]
    title = p["current_title"]
    yoe = p["years_of_experience"]
    location = p["location"]
    country = p["country"]

    parts = []
    concerns = []

    # Title assessment
    if features["title"] >= 0.85:
        parts.append(f"{title} with { _format_yoe(yoe) } yrs")
    elif features["title"] >= 0.5:
        parts.append(f"{title} ({_format_yoe(yoe)} yrs) — adjacent ML role")
    else:
        parts.append(f"{title} ({_format_yoe(yoe)} yrs) — weak title fit for Senior AI Engineer")
        concerns.append("title mismatch")

    # Core skills
    core_count = features["core_skill_count"]
    if core_count >= 4:
        top_skills = ", ".join(features["core_skill_names"][:3])
        parts.append(f"{core_count} retrieval/ranking skills incl. {top_skills}")
    elif core_count >= 2:
        parts.append(f"{core_count} core IR/embedding skills")
    elif core_count == 0:
        concerns.append("no core retrieval/embedding skills")

    # Career signals
    if features["career"] >= 0.7:
        parts.append("product-company ML production background")
    elif features["career"] < 0.4:
        concerns.append("consulting-only or weak production signals")

    # Location
    if country == "India":
        loc_note = location.split(",")[0] if "," in location else location
        if "noida" in location.lower() or "pune" in location.lower():
            parts.append(f"based in {loc_note}")
        elif features["location"] >= 0.8:
            parts.append(f"India ({loc_note}), willing to relocate" if s.get("willing_to_relocate") else f"India ({loc_note})")
    else:
        concerns.append(f"located in {country}, not India")

    # Behavioral
    response = s.get("recruiter_response_rate", 0)
    if response >= 0.7:
        parts.append(f"high engagement (response rate {response:.2f})")
    elif response < 0.2:
        concerns.append(f"low recruiter response rate ({response:.2f})")

    if s.get("open_to_work_flag"):
        parts.append("open to work")
    elif rank <= 30:
        concerns.append("not flagged open to work")

    notice = s.get("notice_period_days", 90)
    if notice > 60 and rank <= 50:
        concerns.append(f"{notice}-day notice period")

    github = s.get("github_activity_score", -1)
    if github >= 50:
        parts.append(f"active GitHub (score {github:.0f})")

    # Honeypot concerns
    if features["honeypot"] < 0.5:
        concerns.append("profile consistency flags")

    # Compose final string
    main = "; ".join(parts[:4])
    if concerns and rank <= 50:
        main += f". Concerns: {', '.join(concerns[:2])}."
    elif concerns:
        main += f". Gaps: {concerns[0]}."
    else:
        main += "."

    # Keep within reasonable length
    if len(main) > 220:
        main = main[:217] + "..."

    return main
