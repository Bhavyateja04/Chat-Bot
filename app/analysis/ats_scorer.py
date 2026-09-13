"""Transparent, weighted ATS scoring engine.

The score is computed in Python, not asked of an LLM. Each of the six components
is scored 0-100 independently and then combined with documented weights, so the
final number can always be explained and reproduced.

Weighting rationale
-------------------
=========================  ======  ==========================================
Component                  Weight  Why
=========================  ======  ==========================================
Required skill match        25%    The single strongest predictor of fit;
                                   missing must-haves should dominate.
Keyword match               20%    Mirrors how real ATS software filters, and
                                   captures JD vocabulary beyond the skill list.
Experience match            20%    Seniority/years gate most shortlists.
Project / responsibility    15%    Evidence that skills were actually applied.
Education / certification   10%    Usually a gate, rarely a differentiator.
Resume structure            10%    Genuine ATS parseability signal.
=========================  ======  ==========================================

A role mismatch (e.g. a mechanical JD against a software resume) applies a hard
penalty *after* weighting, because a resume full of impressive but irrelevant
keywords must not score well.
"""

from __future__ import annotations

from app.analysis import skill_matcher, skill_taxonomy as tax
from app.models.models import (
    ATSBreakdown,
    JobDescription,
    MatchStatus,
    Priority,
    ResumeProfile,
    RoleMismatch,
    SkillImportance,
    SkillMatchResult,
)

WEIGHTS: dict[str, float] = {
    "required_skills": 0.25,
    "keyword_match": 0.20,
    "experience": 0.20,
    "projects": 0.15,
    "education": 0.10,
    "resume_structure": 0.10,
}

# Multiplier applied to the weighted total when a role mismatch is detected.
MISMATCH_PENALTY: dict[Priority, float] = {
    Priority.HIGH: 0.30,
    Priority.MEDIUM: 0.55,
    Priority.LOW: 0.85,
}

# Absolute ceiling on the ATS score for a hard role mismatch, so a keyword-dense
# but wrong-domain resume can never look like a shortlist candidate.
MISMATCH_CEILING: dict[Priority, float] = {
    Priority.HIGH: 30.0,
    Priority.MEDIUM: 55.0,
    Priority.LOW: 100.0,
}


# --------------------------------------------------------------------------- #
# Component scores
# --------------------------------------------------------------------------- #
def score_required_skills(matches: list[SkillMatchResult]) -> float:
    """Coverage of REQUIRED skills only (partial matches earn half credit)."""
    required = [m for m in matches if m.importance == SkillImportance.REQUIRED]
    if not required:
        # No explicit requirements - fall back to overall coverage rather than
        # rewarding the candidate with a free 100.
        return skill_matcher.skill_match_score(matches) if matches else 50.0
    earned = sum(m.credit for m in required)
    return round(100.0 * earned / len(required), 1)


def score_keywords(jd: JobDescription, profile: ResumeProfile) -> float:
    """How much of the JD vocabulary shows up in the resume."""
    keywords = list(jd.keywords)
    if not keywords:
        keywords = [r.skill for r in jd.all_requirements()]
    return skill_matcher.keyword_coverage(keywords, profile)


def score_experience(jd: JobDescription, profile: ResumeProfile,
                     matches: list[SkillMatchResult]) -> float:
    """Blend of years-of-experience fit and how relevant that experience is."""
    # --- years component ---
    if jd.min_years_experience is None:
        years_component = 65.0 if profile.years_of_experience is not None else 50.0
    elif profile.years_of_experience is None:
        years_component = 40.0  # requirement stated, resume does not evidence it
    else:
        required = max(jd.min_years_experience, 0.5)
        ratio = profile.years_of_experience / required
        if ratio >= 1.0:
            # Cap the bonus for being over-qualified.
            years_component = min(100.0, 85.0 + 15.0 * min(ratio - 1.0, 1.0))
        else:
            years_component = max(0.0, 100.0 * ratio * 0.85)

    # --- relevance component: are the JD skills evidenced inside job history? ---
    history_text = " ".join(
        f"{e.title} {e.company} {e.description}" for e in profile.experience
    )
    relevance_component = _evidence_coverage(history_text, matches)
    if not profile.experience:
        relevance_component = min(relevance_component, 35.0)

    return round(0.6 * years_component + 0.4 * relevance_component, 1)


def score_projects(jd: JobDescription, profile: ResumeProfile,
                   matches: list[SkillMatchResult]) -> float:
    """Evidence that required skills were actually applied in work/projects."""
    project_text = " ".join(
        f"{p.name} {p.description} {' '.join(p.technologies)}" for p in profile.projects
    )
    history_text = " ".join(e.description for e in profile.experience)
    achievements = " ".join(profile.achievements)
    combined = f"{project_text} {history_text} {achievements}"

    coverage = _evidence_coverage(combined, matches)

    # Responsibility overlap: do the JD's responsibilities appear in the resume?
    responsibility_score = _responsibility_overlap(jd, profile)

    base = 0.65 * coverage + 0.35 * responsibility_score
    if not profile.projects and not profile.experience:
        base = min(base, 30.0)
    return round(base, 1)


def score_education(jd: JobDescription, profile: ResumeProfile) -> float:
    """Degree / field / certification fit."""
    has_education = bool(profile.education)

    if not jd.education_requirements and not jd.certifications:
        return 75.0 if has_education else 55.0

    if not has_education and not profile.certifications:
        return 30.0

    resume_edu_text = tax.normalize(
        " ".join(
            f"{e.degree} {e.field_of_study} {e.institution}" for e in profile.education
        )
        + " "
        + " ".join(profile.certifications)
    )

    # --- degree/field requirement matching ---
    # Scored in two halves so that "has a bachelor's degree" alone cannot earn
    # full marks when the JD also names a field of study.
    edu_score = 60.0
    if jd.education_requirements:
        _DEGREE_WORDS = {
            "bachelor", "bachelors", "master", "masters", "phd", "doctorate",
            "diploma", "btech", "mtech", "bsc", "msc", "mba", "associate",
        }
        _IGNORE = {"degree", "field", "related", "equivalent", "practical", "experience", "or"}

        level_hits = 0.0
        field_hits = 0.0
        field_requirements = 0
        for requirement in jd.education_requirements:
            tokens = [t for t in tax.normalize(requirement).split() if len(t) > 2]
            degree_tokens = [t for t in tokens if t in _DEGREE_WORDS]
            field_tokens = [
                t for t in tokens
                if len(t) > 3 and t not in _DEGREE_WORDS and t not in _IGNORE
            ]

            if degree_tokens and any(t in resume_edu_text for t in degree_tokens):
                level_hits += 1
            elif has_education:
                # Some degree is present even if the level does not line up.
                level_hits += 0.5

            if field_tokens:
                field_requirements += 1
                if any(t in resume_edu_text for t in field_tokens):
                    field_hits += 1

        count = len(jd.education_requirements)
        level_score = 100.0 * level_hits / count
        if field_requirements:
            field_score = 100.0 * field_hits / field_requirements
            edu_score = 0.45 * level_score + 0.55 * field_score
        else:
            edu_score = level_score
        # An unrelated degree still counts for something - many employers accept
        # adjacent fields - but it must not look like a full match.
        edu_score = max(edu_score, 40.0 if has_education else 20.0)

    # --- certification matching ---
    if jd.certifications:
        resume_certs = tax.canonical_set(profile.certifications)
        cert_hits = sum(
            1
            for c in jd.certifications
            if tax.canonical(c) in resume_certs or tax.normalize(c) in resume_edu_text
        )
        cert_score = 100.0 * cert_hits / len(jd.certifications)
        edu_score = 0.7 * edu_score + 0.3 * cert_score

    return round(min(edu_score, 100.0), 1)


def score_structure(profile: ResumeProfile, parse_warnings: list[str] | None = None) -> float:
    """ATS parseability: contact details, recognisable sections, usable length."""
    score = 0.0
    if profile.email:
        score += 15.0
    if profile.phone:
        score += 10.0
    if profile.candidate_name and profile.candidate_name != "Unknown Candidate":
        score += 10.0
    if profile.experience:
        score += 20.0
    if profile.education:
        score += 15.0
    if profile.all_skills():
        score += 20.0
    if profile.projects:
        score += 10.0

    text_len = len(profile.raw_text.strip())
    if text_len < 400:
        score *= 0.5   # suspiciously thin - likely a bad extraction
    elif text_len < 800:
        score *= 0.8

    for _ in parse_warnings or []:
        score -= 8.0

    return round(max(0.0, min(score, 100.0)), 1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _evidence_coverage(text: str, matches: list[SkillMatchResult]) -> float:
    """Share of matched JD skills that are actually evidenced inside ``text``."""
    considered = [m for m in matches if m.status != MatchStatus.MISSING]
    if not considered or not text.strip():
        return 0.0
    norm = " " + tax.normalize(text) + " "
    hits = 0
    for match in considered:
        needle = tax.normalize(match.matched_via or match.skill)
        if needle and (f" {needle} " in norm or needle in norm):
            hits += 1
    # Denominator counts every JD skill, so evidencing 3 of 10 requirements in
    # real work stays a low score even if all 3 matched skills are evidenced.
    denominator = max(len(matches), 1)
    return round(100.0 * hits / denominator, 1)


def _responsibility_overlap(jd: JobDescription, profile: ResumeProfile) -> float:
    """Rough overlap between JD responsibilities and resume narrative text."""
    if not jd.responsibilities:
        return 60.0 if (profile.experience or profile.projects) else 30.0

    resume_text = tax.normalize(profile.raw_text)
    if not resume_text:
        return 0.0

    stopwords = {
        "with", "and", "the", "for", "our", "you", "will", "work", "team",
        "using", "across", "into", "from", "that", "this", "their", "them",
        "other", "also", "have", "been", "your", "within", "ensure", "help",
    }
    matched = 0
    for responsibility in jd.responsibilities:
        tokens = [
            t for t in tax.normalize(responsibility).split()
            if len(t) > 3 and t not in stopwords
        ]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in resume_text)
        # A responsibility counts as covered when a third of its content words
        # appear somewhere in the resume.
        if hits >= max(1, len(tokens) // 3):
            matched += 1
    return round(100.0 * matched / len(jd.responsibilities), 1)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_breakdown(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
    parse_warnings: list[str] | None = None,
) -> ATSBreakdown:
    """Score all six components."""
    return ATSBreakdown(
        keyword_match=score_keywords(jd, profile),
        required_skills=score_required_skills(matches),
        experience=score_experience(jd, profile, matches),
        projects=score_projects(jd, profile, matches),
        education=score_education(jd, profile),
        resume_structure=score_structure(profile, parse_warnings),
    )


def weighted_total(breakdown: ATSBreakdown) -> float:
    """Combine components using the documented weights."""
    total = (
        breakdown.required_skills * WEIGHTS["required_skills"]
        + breakdown.keyword_match * WEIGHTS["keyword_match"]
        + breakdown.experience * WEIGHTS["experience"]
        + breakdown.projects * WEIGHTS["projects"]
        + breakdown.education * WEIGHTS["education"]
        + breakdown.resume_structure * WEIGHTS["resume_structure"]
    )
    return round(total, 1)


def apply_mismatch_penalty(score: float, mismatch: RoleMismatch) -> float:
    """Penalise and cap the score when the resume is from the wrong domain."""
    if not mismatch.detected:
        return score
    penalised = score * MISMATCH_PENALTY.get(mismatch.severity, 0.6)
    ceiling = MISMATCH_CEILING.get(mismatch.severity, 100.0)
    return round(min(penalised, ceiling), 1)


def compute_ats_score(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
    mismatch: RoleMismatch,
    parse_warnings: list[str] | None = None,
) -> tuple[float, ATSBreakdown]:
    """Full ATS pipeline: components -> weighted total -> mismatch penalty."""
    breakdown = compute_breakdown(jd, profile, matches, parse_warnings)
    total = weighted_total(breakdown)
    total = apply_mismatch_penalty(total, mismatch)
    return round(max(0.0, min(total, 100.0)), 1), breakdown


def explain_score(breakdown: ATSBreakdown, total: float, mismatch: RoleMismatch) -> str:
    """One-paragraph, human-readable justification of the number."""
    items = sorted(breakdown.as_dict().items(), key=lambda kv: kv[1], reverse=True)
    strongest = ", ".join(f"{name} ({value:.0f}%)" for name, value in items[:2])
    weakest = ", ".join(f"{name} ({value:.0f}%)" for name, value in items[-2:])

    parts = [
        f"Weighted across six components the resume scores **{total:.0f}/100**.",
        f"Strongest areas: {strongest}.",
        f"Weakest areas: {weakest}.",
    ]
    if mismatch.detected:
        parts.append(
            f"A role-mismatch penalty was applied because the JD targets "
            f"{tax.domain_label(mismatch.jd_domain)} while the resume reads as "
            f"{tax.domain_label(mismatch.resume_domain)}."
        )
    return " ".join(parts)


# Public alias - the alignment scorer in analyzer.py reuses this component.
responsibility_overlap = _responsibility_overlap
evidence_coverage = _evidence_coverage
