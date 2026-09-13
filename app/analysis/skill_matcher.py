"""Semantic skill matching between JD requirements and a resume.

The matcher is deliberately conservative. It awards:

* ``EXACT``   - the resume names the skill or a true synonym ("JS" == JavaScript).
* ``PARTIAL`` - the resume shows a genuinely related skill from the curated
  related-skill graph (MySQL vs PostgreSQL). Worth half credit.
* ``MISSING`` - no evidence found. We never claim the candidate *lacks* the
  skill, only that no evidence appeared in the document.

Unrelated skills are never conflated: Python does not satisfy Java, and React
does not satisfy Angular beyond "same category" partial credit.
"""

from __future__ import annotations

import re

from app.analysis import skill_taxonomy as tax
from app.models.models import (
    JDRequirement,
    MatchStatus,
    ResumeProfile,
    SkillImportance,
    SkillMatchResult,
)

# Very short canonical skills ("go", "r", "c") would produce false positives via
# substring search, so they are only matched as whole words.
_MIN_SUBSTRING_LEN = 4


def _evidence_snippet(raw_text: str, term: str, max_len: int = 160) -> str:
    """Pull the line of the resume that mentions ``term``, for explainability."""
    if not raw_text or not term:
        return ""
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for line in raw_text.splitlines():
        if pattern.search(line):
            snippet = " ".join(line.split()).strip(" -*•\t")
            if len(snippet) > max_len:
                snippet = snippet[: max_len - 1].rstrip() + "…"
            if snippet:
                return snippet
    return ""


def _mentions(raw_text_norm: str, canonical_skill: str) -> bool:
    """Whole-word / phrase presence check against normalised resume text."""
    if not canonical_skill:
        return False
    needle = tax.normalize(canonical_skill)
    if not needle:
        return False
    if len(needle) < _MIN_SUBSTRING_LEN:
        # Short tokens must appear as standalone words.
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", raw_text_norm) is not None
    return f" {needle} " in raw_text_norm


def _resume_surface_forms(profile: ResumeProfile) -> dict[str, str]:
    """canonical skill -> the original wording used in the resume."""
    surface: dict[str, str] = {}
    for skill in profile.all_skills():
        canon = tax.canonical(skill)
        if canon and canon not in surface:
            surface[canon] = skill
    # Technologies named inside project entries count too.
    for project in profile.projects:
        for tech in project.technologies:
            canon = tax.canonical(tech)
            if canon and canon not in surface:
                surface[canon] = tech
    return surface


def match_skill(
    requirement: JDRequirement,
    profile: ResumeProfile,
    *,
    resume_surface: dict[str, str] | None = None,
    raw_text_norm: str | None = None,
) -> SkillMatchResult:
    """Judge a single JD requirement against the resume."""
    surface = resume_surface if resume_surface is not None else _resume_surface_forms(profile)
    text_norm = (
        raw_text_norm
        if raw_text_norm is not None
        else " " + tax.normalize(profile.raw_text) + " "
    )

    jd_canon = tax.canonical(requirement.skill)
    if not jd_canon:
        return SkillMatchResult(
            skill=requirement.skill,
            importance=requirement.importance,
            status=MatchStatus.MISSING,
        )

    # 1. Exact / synonym match against the extracted skill list.
    if jd_canon in surface:
        via = surface[jd_canon]
        return SkillMatchResult(
            skill=requirement.skill,
            importance=requirement.importance,
            status=MatchStatus.EXACT,
            matched_via=via,
            evidence=_evidence_snippet(profile.raw_text, via),
        )

    # 2. Exact match found in the resume body even if the extractor missed it.
    if _mentions(text_norm, jd_canon):
        return SkillMatchResult(
            skill=requirement.skill,
            importance=requirement.importance,
            status=MatchStatus.EXACT,
            matched_via=requirement.skill,
            evidence=_evidence_snippet(profile.raw_text, jd_canon)
            or _evidence_snippet(profile.raw_text, requirement.skill),
        )

    # 3. Related-skill partial credit, preferring the strongest available relation.
    for resume_canon, resume_surface_form in surface.items():
        relation = tax.relation_between(jd_canon, resume_canon)
        if relation:
            return SkillMatchResult(
                skill=requirement.skill,
                importance=requirement.importance,
                status=MatchStatus.PARTIAL,
                matched_via=resume_surface_form,
                relation=relation,
                evidence=_evidence_snippet(profile.raw_text, resume_surface_form),
            )

    # 4. Related skill mentioned in prose but not extracted as a skill.
    for related_canon, relation in tax.RELATED.get(jd_canon, {}).items():
        if _mentions(text_norm, related_canon):
            return SkillMatchResult(
                skill=requirement.skill,
                importance=requirement.importance,
                status=MatchStatus.PARTIAL,
                matched_via=related_canon,
                relation=relation,
                evidence=_evidence_snippet(profile.raw_text, related_canon),
            )

    return SkillMatchResult(
        skill=requirement.skill,
        importance=requirement.importance,
        status=MatchStatus.MISSING,
    )


def match_all(
    requirements: list[JDRequirement], profile: ResumeProfile
) -> list[SkillMatchResult]:
    """Match every JD requirement, de-duplicating by canonical skill name."""
    surface = _resume_surface_forms(profile)
    text_norm = " " + tax.normalize(profile.raw_text) + " "

    results: list[SkillMatchResult] = []
    seen: set[str] = set()
    for requirement in requirements:
        canon = tax.canonical(requirement.skill)
        if canon and canon in seen:
            continue
        if canon:
            seen.add(canon)
        results.append(
            match_skill(
                requirement,
                profile,
                resume_surface=surface,
                raw_text_norm=text_norm,
            )
        )
    return results


def skill_match_score(results: list[SkillMatchResult]) -> float:
    """Weighted skill-coverage percentage (0-100).

    Required skills carry 3x the weight of preferred skills so that a candidate
    cannot compensate for a missing must-have by collecting nice-to-haves.
    """
    if not results:
        return 0.0

    weight_for = {SkillImportance.REQUIRED: 3.0, SkillImportance.PREFERRED: 1.0}
    earned = sum(r.credit * weight_for[r.importance] for r in results)
    possible = sum(weight_for[r.importance] for r in results)
    if possible == 0:
        return 0.0
    return round(100.0 * earned / possible, 1)


def split_by_status(
    results: list[SkillMatchResult],
) -> tuple[list[SkillMatchResult], list[SkillMatchResult], list[SkillMatchResult]]:
    """Split into (exact, partial, missing), required-first within each bucket."""

    def sort_key(r: SkillMatchResult) -> tuple[int, str]:
        return (0 if r.importance == SkillImportance.REQUIRED else 1, r.skill.lower())

    exact = sorted([r for r in results if r.status == MatchStatus.EXACT], key=sort_key)
    partial = sorted([r for r in results if r.status == MatchStatus.PARTIAL], key=sort_key)
    missing = sorted([r for r in results if r.status == MatchStatus.MISSING], key=sort_key)
    return exact, partial, missing


def missing_keywords(jd_keywords: list[str], profile: ResumeProfile) -> list[str]:
    """JD keywords with no presence (or synonym) anywhere in the resume.

    Used to justify a keyword-coverage gap with concrete, checkable evidence
    rather than just a percentage.
    """
    if not jd_keywords:
        return []
    text_norm = " " + tax.normalize(profile.raw_text) + " "
    surface = _resume_surface_forms(profile)

    absent: list[str] = []
    seen: set[str] = set()
    for keyword in jd_keywords:
        canon = tax.canonical(keyword)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        if canon not in surface and not _mentions(text_norm, canon):
            absent.append(keyword)
    return absent


def keyword_coverage(jd_keywords: list[str], profile: ResumeProfile) -> float:
    """Percentage of JD keywords that appear (as synonyms too) in the resume."""
    if not jd_keywords:
        return 0.0
    text_norm = " " + tax.normalize(profile.raw_text) + " "
    surface = _resume_surface_forms(profile)

    hits = 0
    checked = 0
    seen: set[str] = set()
    for keyword in jd_keywords:
        canon = tax.canonical(keyword)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        checked += 1
        if canon in surface or _mentions(text_norm, canon):
            hits += 1
    if checked == 0:
        return 0.0
    return round(100.0 * hits / checked, 1)
