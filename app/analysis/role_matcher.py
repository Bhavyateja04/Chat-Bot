"""Role / domain mismatch detection.

The recruiter explicitly tests a Mechanical Engineering JD against a Software
Engineering resume. A naive keyword scorer rates that resume highly because it is
dense with impressive technical terms. This module classifies both documents into
a professional domain and flags the mismatch so the ATS scorer can penalise it.

Design notes
------------
* Detection requires *both* documents to classify confidently. When either is
  unclassifiable we stay silent rather than accuse the candidate of a mismatch.
* Domains inside the same family (software / data / devops) are treated as
  compatible: a full-stack resume against a backend JD is a specialisation
  difference, not a career mismatch. Those cases are handled by the ordinary
  score, which drops on its own when required skills are missing.
"""

from __future__ import annotations

from app.analysis import skill_taxonomy as tax
from app.models.models import (
    JobDescription,
    MatchStatus,
    Priority,
    ResumeProfile,
    RoleMismatch,
    SkillImportance,
    SkillMatchResult,
)

# Minimum share of vocabulary hits before we trust a domain classification.
_CONFIDENT_DOMAIN_SHARE = 0.30
# Required-skill coverage below this reinforces a mismatch verdict.
_LOW_COVERAGE = 0.35


def classify_jd(jd: JobDescription) -> tuple[str, dict[str, float]]:
    """Classify the JD's professional domain."""
    skills = [r.skill for r in jd.all_requirements()] + list(jd.domain_knowledge)
    text = " ".join(
        [jd.job_title, jd.raw_text, " ".join(jd.responsibilities), " ".join(jd.keywords)]
    )
    return tax.classify_domain(text, skills)


def classify_resume(profile: ResumeProfile) -> tuple[str, dict[str, float]]:
    """Classify the resume's professional domain."""
    titles = " ".join(f"{e.title} {e.description}" for e in profile.experience)
    education = " ".join(f"{e.degree} {e.field_of_study}" for e in profile.education)
    text = " ".join([profile.summary, titles, education, profile.raw_text])
    return tax.classify_domain(text, profile.all_skills())


def _resume_role_label(profile: ResumeProfile, domain: str) -> str:
    """Best available human label for what the resume looks like."""
    if profile.experience and profile.experience[0].title:
        return profile.experience[0].title
    return tax.domain_label(domain)


def detect(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult] | None = None,
) -> RoleMismatch:
    """Decide whether this resume is from the wrong professional domain."""
    jd_domain, jd_scores = classify_jd(jd)
    resume_domain, resume_scores = classify_resume(profile)

    result = RoleMismatch(
        detected=False,
        jd_role=jd.job_title or tax.domain_label(jd_domain),
        resume_profile=_resume_role_label(profile, resume_domain),
        jd_domain=jd_domain,
        resume_domain=resume_domain,
        severity=Priority.LOW,
    )

    # Not enough signal in one of the documents - do not make an accusation.
    if jd_domain == "unknown" or resume_domain == "unknown":
        return result

    if tax.domains_compatible(jd_domain, resume_domain):
        return result

    # Both classified confidently and into incompatible families.
    jd_confidence = jd_scores.get(jd_domain, 0.0)
    resume_confidence = resume_scores.get(resume_domain, 0.0)

    # How much of the JD's own domain does the resume actually speak?
    resume_share_of_jd_domain = resume_scores.get(jd_domain, 0.0)

    required_matches = [
        m for m in (matches or []) if m.importance == SkillImportance.REQUIRED
    ]
    coverage = (
        sum(m.credit for m in required_matches) / len(required_matches)
        if required_matches
        else 0.0
    )

    confident = (
        jd_confidence >= _CONFIDENT_DOMAIN_SHARE
        and resume_confidence >= _CONFIDENT_DOMAIN_SHARE
    )

    if confident and resume_share_of_jd_domain < 0.20 and coverage < _LOW_COVERAGE:
        severity = Priority.HIGH
    elif confident or coverage < _LOW_COVERAGE:
        severity = Priority.MEDIUM
    else:
        severity = Priority.LOW

    result.detected = True
    result.severity = severity
    result.reason = (
        f"The job description is centred on {tax.domain_label(jd_domain)} "
        f"(role: {result.jd_role}), while the resume's evidence is concentrated in "
        f"{tax.domain_label(resume_domain)}. "
        f"Only {coverage * 100:.0f}% of the required skills are demonstrated."
    )
    result.critical_missing = _critical_missing(jd, matches or [], jd_domain)
    return result


def _critical_missing(
    jd: JobDescription, matches: list[SkillMatchResult], jd_domain: str
) -> list[str]:
    """Required JD skills that are both missing and core to the JD's domain."""
    vocabulary = tax.DOMAIN_VOCABULARY.get(jd_domain, set())
    canonical_vocab = tax.canonical_set(vocabulary)

    core: list[str] = []
    other: list[str] = []
    for match in matches:
        if match.status != MatchStatus.MISSING:
            continue
        if match.importance != SkillImportance.REQUIRED:
            continue
        if tax.canonical(match.skill) in canonical_vocab:
            core.append(match.skill)
        else:
            other.append(match.skill)

    # Prefer domain-defining skills, then top up with any other missing must-haves.
    if not matches:
        return [r.skill for r in jd.required_skills][:6]
    return (core + other)[:6]
