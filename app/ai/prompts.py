"""Prompt construction for the three LLM passes.

Design rule: the LLM extracts and explains, it never scores. Every number in the
final report is computed in Python from the structures the LLM returns.
"""

from __future__ import annotations

from app.ai import schemas
from app.models.models import (
    JobDescription,
    MatchStatus,
    ResumeProfile,
    SkillMatchResult,
)

_JSON_RULES = (
    "Respond with a single valid JSON object and nothing else. "
    "No markdown code fences, no commentary before or after the JSON. "
    "Use empty strings, empty arrays or null for anything not present in the "
    "document. Never invent information."
)

EXTRACTION_SYSTEM = (
    "You are a precise document information extractor for a recruitment system. "
    "You extract only what is explicitly present in the document. "
    "You never infer, guess, embellish or add skills that are not written down. "
    + _JSON_RULES
)

INSIGHT_SYSTEM = (
    "You are a senior technical recruiter writing concise, evidence-based "
    "candidate assessments. You are given scores that have ALREADY been computed "
    "by a deterministic scoring engine - never recompute, dispute or restate "
    "different numbers. Your job is only to explain them in plain language. "
    "When a skill is missing, say 'No evidence of X was found in the resume', "
    "never 'the candidate cannot do X'. "
    + _JSON_RULES
)


def jd_extraction_prompt(text: str) -> str:
    return (
        "Extract the structured requirements from this JOB DESCRIPTION.\n\n"
        "Rules:\n"
        "- Split requirements into required_skills (must-have) and preferred_skills "
        "(nice-to-have / bonus / preferred). If the JD does not distinguish, treat "
        "them as required.\n"
        "- Each skill must be a short atomic name ('PostgreSQL', 'Docker'), not a "
        "sentence.\n"
        "- Only set min_years_experience when a number is actually stated.\n\n"
        f"Return JSON in exactly this shape:\n{schemas.JD_SCHEMA}\n\n"
        f"--- JOB DESCRIPTION ---\n{text}\n--- END ---"
    )


def resume_extraction_prompt(text: str) -> str:
    return (
        "Extract structured information from this RESUME.\n\n"
        "Rules:\n"
        "- Only record skills that literally appear in the resume text.\n"
        "- Do NOT infer skills from job titles (a 'Backend Engineer' does not "
        "automatically know Docker).\n"
        "- Set years_of_experience only if stated outright or clearly derivable "
        "from employment dates; otherwise null.\n\n"
        f"Return JSON in exactly this shape:\n{schemas.RESUME_SCHEMA}\n\n"
        f"--- RESUME ---\n{text}\n--- END ---"
    )


def _format_matches(matches: list[SkillMatchResult], limit: int = 18) -> str:
    lines: list[str] = []
    for match in matches[:limit]:
        tag = match.importance.value
        if match.status == MatchStatus.EXACT:
            lines.append(f"  MATCHED   [{tag}] {match.skill} (found: {match.matched_via})")
        elif match.status == MatchStatus.PARTIAL:
            lines.append(
                f"  PARTIAL   [{tag}] {match.skill} "
                f"(resume shows {match.matched_via}; {match.relation})"
            )
        else:
            lines.append(f"  MISSING   [{tag}] {match.skill}")
    return "\n".join(lines) or "  (none)"


def insight_prompt(
    jd: JobDescription,
    candidates: list[tuple[ResumeProfile, list[SkillMatchResult], dict[str, float]]],
) -> str:
    """One batched call covering every resume in the session.

    Batching keeps the demo fast and the token cost low: one narrative call for
    the whole comparison rather than one per candidate.
    """
    required = ", ".join(r.skill for r in jd.required_skills[:20]) or "not specified"
    preferred = ", ".join(r.skill for r in jd.preferred_skills[:15]) or "none listed"

    blocks: list[str] = []
    for profile, matches, scores in candidates:
        blocks.append(
            f"### CANDIDATE: {profile.candidate_name}\n"
            f"Detected profile: {profile.detected_domain or 'unclassified'}\n"
            f"Years of experience found: {profile.years_of_experience if profile.years_of_experience is not None else 'not stated'}\n"
            f"COMPUTED SCORES (authoritative, do not change):\n"
            f"  ATS score: {scores.get('ats', 0):.0f}/100\n"
            f"  Skill match: {scores.get('skill', 0):.0f}%\n"
            f"  Alignment: {scores.get('alignment', 0):.0f}%\n"
            f"Skill matching results:\n{_format_matches(matches)}\n"
        )

    return (
        "Write an evidence-based assessment for each candidate below against the "
        "job description.\n\n"
        f"JOB TITLE: {jd.job_title}\n"
        f"REQUIRED SKILLS: {required}\n"
        f"PREFERRED SKILLS: {preferred}\n"
        f"MINIMUM YEARS: {jd.min_years_experience if jd.min_years_experience is not None else 'not specified'}\n"
        f"KEY RESPONSIBILITIES:\n"
        + "\n".join(f"  - {r}" for r in jd.responsibilities[:8])
        + "\n\n"
        + "\n".join(blocks)
        + "\n\nRules:\n"
        "- Use the computed scores exactly as given; never state a different number.\n"
        "- Every strong point must cite something actually in that resume.\n"
        "- Phrase gaps as absence of evidence, not absence of ability.\n"
        "- extra_missing_areas is for gaps that simple skill matching would miss "
        "(e.g. no production deployment evidence, no team leadership evidence). "
        "Return an empty array if there are none.\n"
        "- Return one entry in 'analyses' per candidate, in the same order.\n\n"
        f"Return JSON in exactly this shape:\n{schemas.INSIGHT_SCHEMA}"
    )
