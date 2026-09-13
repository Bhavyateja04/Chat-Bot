"""Analysis orchestration.

Division of labour, deliberately:

* **LLM**   - semantic extraction (what the JD asks for, what the resume shows)
  and the human-readable narrative.
* **Python** - all matching, all scoring, all ranking, all validation, all
  formatting. Every number is reproducible without the model.

If the LLM is unavailable, misconfigured or returns junk, the deterministic
heuristic extractor takes over and the analysis still completes - the confidence
indicator drops to reflect that.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from app.ai import prompts
from app.ai.client import LLMClient, LLMError
from app.analysis import (
    ats_scorer,
    heuristic_extractor,
    recommender,
    role_matcher,
    skill_matcher,
    skill_taxonomy as tax,
)
from app.models.models import (
    AlignmentBreakdown,
    AnalysisResult,
    ComparisonReport,
    Confidence,
    JDRequirement,
    JobDescription,
    MatchStatus,
    MissingArea,
    ParsedDocument,
    Priority,
    ResumeProfile,
    SkillImportance,
    SkillMatchResult,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]

# Alignment weighting - skills dominate, but breadth of evidence matters.
ALIGNMENT_WEIGHTS: dict[str, float] = {
    "technical_skills": 0.30,
    "experience": 0.20,
    "projects": 0.15,
    "responsibilities": 0.15,
    "keywords": 0.10,
    "education": 0.10,
}


async def _report(callback: Optional[ProgressCallback], message: str) -> None:
    if callback is not None:
        try:
            await callback(message)
        except Exception:  # progress must never break the analysis
            logger.debug("Progress callback failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Extraction (LLM with deterministic fallback)
# --------------------------------------------------------------------------- #
async def extract_job_description(
    document: ParsedDocument, client: LLMClient | None = None
) -> tuple[JobDescription, bool]:
    """Return ``(job_description, llm_used)``."""
    client = client or LLMClient()
    fallback = heuristic_extractor.extract_jd(document)

    if not client.available:
        return fallback, False

    try:
        data = await client.complete_json(
            prompts.EXTRACTION_SYSTEM, prompts.jd_extraction_prompt(document.text)
        )
        jd = _jd_from_llm(data, document)
        # Guard against a model that returns a technically valid but empty result.
        if not jd.required_skills and not jd.preferred_skills:
            logger.info("LLM returned no JD skills; using heuristic extraction.")
            return fallback, False
        return jd, True
    except LLMError as exc:
        logger.warning("JD extraction via LLM failed, falling back: %s", exc)
        return fallback, False
    except Exception:
        logger.exception("Unexpected JD extraction failure, falling back.")
        return fallback, False


def _jd_from_llm(data: dict, document: ParsedDocument) -> JobDescription:
    """Validate raw LLM JSON into a JobDescription."""

    def requirements(key: str, importance: SkillImportance) -> list[JDRequirement]:
        items = data.get(key) or []
        out: list[JDRequirement] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, str):
                skill, context = item, ""
            elif isinstance(item, dict):
                skill = str(item.get("skill") or "").strip()
                context = str(item.get("context") or "").strip()
            else:
                continue
            if not skill or len(skill) > 80:
                continue
            canon = tax.canonical(skill)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(
                JDRequirement(skill=skill, importance=importance, context=context)
            )
        return out

    def strings(key: str, limit: int = 40) -> list[str]:
        items = data.get(key) or []
        return [
            str(i).strip()
            for i in items
            if isinstance(i, (str, int, float)) and str(i).strip()
        ][:limit]

    years = data.get("min_years_experience")
    try:
        min_years = float(years) if years is not None else None
    except (TypeError, ValueError):
        min_years = None

    jd = JobDescription(
        job_title=str(data.get("job_title") or "Unspecified Role").strip()[:120],
        company=str(data.get("company") or "").strip()[:120],
        seniority=str(data.get("seniority") or "").strip()[:40],
        required_skills=requirements("required_skills", SkillImportance.REQUIRED),
        preferred_skills=requirements("preferred_skills", SkillImportance.PREFERRED),
        min_years_experience=min_years,
        education_requirements=strings("education_requirements", 6),
        certifications=strings("certifications", 8),
        responsibilities=strings("responsibilities", 15),
        domain_knowledge=strings("domain_knowledge", 10),
        keywords=strings("keywords", 40),
        source_filename=document.filename,
        raw_text=document.text,
    )
    jd.detected_domain = tax.classify_domain(
        document.text, [r.skill for r in jd.all_requirements()]
    )[0]
    return jd


async def extract_resume_profile(
    document: ParsedDocument, client: LLMClient | None = None
) -> tuple[ResumeProfile, bool]:
    """Return ``(resume_profile, llm_used)``."""
    client = client or LLMClient()
    fallback = heuristic_extractor.extract_resume(document)

    if not client.available:
        return fallback, False

    try:
        data = await client.complete_json(
            prompts.EXTRACTION_SYSTEM, prompts.resume_extraction_prompt(document.text)
        )
        profile = _profile_from_llm(data, document)
        if not profile.all_skills() and fallback.all_skills():
            logger.info("LLM found no resume skills; using heuristic extraction.")
            return fallback, False
        return profile, True
    except LLMError as exc:
        logger.warning("Resume extraction via LLM failed, falling back: %s", exc)
        return fallback, False
    except Exception:
        logger.exception("Unexpected resume extraction failure, falling back.")
        return fallback, False


def _profile_from_llm(data: dict, document: ParsedDocument) -> ResumeProfile:
    """Validate raw LLM JSON into a ResumeProfile."""
    payload = dict(data)
    payload["source_filename"] = document.filename
    payload["raw_text"] = document.text

    years = payload.get("years_of_experience")
    try:
        payload["years_of_experience"] = float(years) if years is not None else None
    except (TypeError, ValueError):
        payload["years_of_experience"] = None

    try:
        profile = ResumeProfile.model_validate(payload)
    except Exception:
        logger.warning("LLM resume payload failed validation; using heuristics.")
        return heuristic_extractor.extract_resume(document)

    # raw_text is excluded from serialisation, so set it explicitly.
    profile.raw_text = document.text
    profile.source_filename = document.filename
    if not profile.candidate_name.strip():
        profile.candidate_name = "Unknown Candidate"
    profile.detected_domain = tax.classify_domain(
        document.text, profile.all_skills()
    )[0]
    return profile


# --------------------------------------------------------------------------- #
# Alignment + confidence
# --------------------------------------------------------------------------- #
def compute_alignment(
    jd: JobDescription,
    profile: ResumeProfile,
    matches: list[SkillMatchResult],
    ats_components,
) -> tuple[float, AlignmentBreakdown]:
    """Job/resume alignment, scored independently of the ATS number."""
    breakdown = AlignmentBreakdown(
        technical_skills=skill_matcher.skill_match_score(matches),
        experience=ats_components.experience,
        projects=ats_components.projects,
        responsibilities=ats_scorer._responsibility_overlap(jd, profile),
        keywords=ats_components.keyword_match,
        education=ats_components.education,
    )
    total = sum(
        getattr(breakdown, field) * weight
        for field, weight in ALIGNMENT_WEIGHTS.items()
    )
    return round(min(max(total, 0.0), 100.0), 1), breakdown


def assess_confidence(
    document: ParsedDocument,
    jd: JobDescription,
    profile: ResumeProfile,
    llm_used: bool,
) -> tuple[Confidence, str]:
    """How much should the reader trust this analysis?"""
    penalties: list[str] = []
    score = 100

    text_length = len(profile.raw_text.strip())
    if text_length < 600:
        score -= 35
        penalties.append("very little text was extracted from the resume")
    elif text_length < 1200:
        score -= 15
        penalties.append("the resume is short, limiting the available evidence")

    if document.warnings:
        score -= 15
        penalties.append("the document had extraction warnings")

    if not profile.has_structure():
        score -= 20
        penalties.append("no clear experience, education or project sections were found")

    if not profile.all_skills():
        score -= 25
        penalties.append("no explicit skills could be identified")

    if not jd.required_skills:
        score -= 25
        penalties.append("the job description does not clearly list required skills")
    elif len(jd.required_skills) < 3:
        score -= 10
        penalties.append("the job description lists very few explicit requirements")

    if len(jd.raw_text.strip()) < 600:
        score -= 15
        penalties.append("the job description is brief")

    if not llm_used:
        score -= 10
        penalties.append("analysis ran on the deterministic extractor without LLM assistance")

    if score >= 75:
        level = Confidence.HIGH
    elif score >= 50:
        level = Confidence.MEDIUM
    else:
        level = Confidence.LOW

    if not penalties:
        reason = "Both documents parsed cleanly with clear requirements and evidence."
    elif level == Confidence.HIGH:
        # Still high overall - report the caveats without claiming a downgrade.
        reason = "Both documents parsed well. Note: " + "; ".join(penalties[:2]) + "."
    else:
        reason = "Reduced because " + "; ".join(penalties[:3]) + "."
    return level, reason


# --------------------------------------------------------------------------- #
# Single-resume analysis
# --------------------------------------------------------------------------- #
def analyze_one(
    jd: JobDescription,
    profile: ResumeProfile,
    document: ParsedDocument,
    llm_used: bool = False,
) -> AnalysisResult:
    """Full deterministic analysis of one resume against the JD."""
    requirements = jd.all_requirements()
    matches = skill_matcher.match_all(requirements, profile)
    exact, partial, missing = skill_matcher.split_by_status(matches)

    mismatch = role_matcher.detect(jd, profile, matches)
    ats_score, breakdown = ats_scorer.compute_ats_score(
        jd, profile, matches, mismatch, document.warnings
    )
    alignment_score, alignment_breakdown = compute_alignment(
        jd, profile, matches, breakdown
    )
    if mismatch.detected:
        alignment_score = ats_scorer.apply_mismatch_penalty(alignment_score, mismatch)

    missing_areas = recommender.build_missing_areas(
        jd, profile, matches, alignment_breakdown
    )
    courses = recommender.recommend_courses(
        matches,
        missing_areas,
        recommender.keyword_learning_gaps(jd, profile, alignment_breakdown),
    )
    roadmap = recommender.build_roadmap(matches, courses)
    confidence, confidence_reason = assess_confidence(document, jd, profile, llm_used)

    result = AnalysisResult(
        candidate_name=profile.candidate_name,
        source_filename=profile.source_filename or document.filename,
        job_title=jd.job_title,
        ats_score=ats_score,
        ats_breakdown=breakdown,
        ats_explanation=ats_scorer.explain_score(breakdown, ats_score, mismatch),
        skill_match_score=skill_matcher.skill_match_score(matches),
        matched_skills=exact,
        partial_skills=partial,
        missing_skills=missing,
        missing_areas=missing_areas,
        alignment_score=alignment_score,
        alignment_breakdown=alignment_breakdown,
        alignment_explanation=_default_alignment_text(
            jd, profile, alignment_score, mismatch
        ),
        strong_points=_default_strong_points(exact, partial, profile),
        weak_points=_default_weak_points(missing, missing_areas, alignment_breakdown),
        recommended_courses=courses,
        learning_roadmap=roadmap,
        role_mismatch=mismatch,
        confidence=confidence,
        confidence_reason=confidence_reason,
        improvement_suggestions=recommender.improvement_suggestions(jd, profile, matches),
        warnings=list(document.warnings),
        llm_assisted=llm_used,
    )
    return result


def _default_alignment_text(
    jd: JobDescription, profile: ResumeProfile, score: float, mismatch
) -> str:
    if mismatch.detected:
        return (
            f"Overall alignment is {score:.0f}%. {mismatch.reason} "
            "The resume would need substantial retraining to fit this role."
        )
    if score >= 80:
        judgement = "a strong fit for this role"
    elif score >= 65:
        judgement = "a good fit with a few gaps"
    elif score >= 45:
        judgement = "a partial fit with meaningful gaps"
    else:
        judgement = "a weak fit for this role as written"
    return (
        f"{profile.candidate_name} is {judgement}, scoring {score:.0f}% overall "
        f"alignment against the {jd.job_title} requirements."
    )


def _default_strong_points(
    exact: list[SkillMatchResult],
    partial: list[SkillMatchResult],
    profile: ResumeProfile,
) -> list[str]:
    points: list[str] = []
    required_hits = [m for m in exact if m.importance == SkillImportance.REQUIRED]
    for match in required_hits[:3]:
        name = tax.display_name(match.skill)
        if match.evidence:
            points.append(f"{name} — evidenced by: \"{match.evidence}\"")
        else:
            points.append(f"{name} is named in the resume as a required match.")
    if profile.years_of_experience is not None:
        points.append(
            f"{profile.years_of_experience:.0f} years of experience are evidenced "
            "in the resume."
        )
    if not points and partial:
        points.append(
            f"Related experience in {tax.display_name(partial[0].matched_via)} "
            f"({partial[0].relation})."
        )
    return points[:4]


def _default_weak_points(
    missing: list[SkillMatchResult],
    missing_areas: list[MissingArea],
    alignment: AlignmentBreakdown | None = None,
) -> list[str]:
    points: list[str] = []
    for match in missing:
        if match.importance == SkillImportance.REQUIRED:
            points.append(
                f"No evidence of {tax.display_name(match.skill)} was found in the resume."
            )
        if len(points) >= 2:
            break

    # Weak alignment dimensions are gaps in their own right, even when every
    # named skill matched. Reported straight from the scored breakdown so the
    # text can never contradict the numbers shown above it.
    for label, score in weak_alignment_dimensions(alignment):
        if len(points) >= 4:
            break
        points.append(f"{label} alignment is low ({score:.0f}%).")

    for area in missing_areas:
        if len(points) >= 4:
            break
        if area.category in {"experience", "project_evidence", "responsibility"}:
            points.append(area.detail)

    if not points:
        preferred_missing = [
            m for m in missing if m.importance == SkillImportance.PREFERRED
        ]
        if preferred_missing:
            names = ", ".join(
                tax.display_name(m.skill) for m in preferred_missing[:4]
            )
            points.append(
                f"No evidence was found for these preferred skills: {names}."
            )
        else:
            points.append(
                "No material gaps were found against the stated requirements."
            )
    return points[:4]


def weak_alignment_dimensions(
    alignment: AlignmentBreakdown | None,
) -> list[tuple[str, float]]:
    """Alignment dimensions scoring below the acceptable threshold, worst first.

    Single source of truth for "which dimensions are weak", shared by the weak
    points text and the Discord renderer so they cannot disagree.
    """
    if alignment is None:
        return []
    weak = [
        (name, score)
        for name, score in alignment.as_dict().items()
        if score < recommender.ALIGNMENT_OK
    ]
    return sorted(weak, key=lambda item: item[1])


# --------------------------------------------------------------------------- #
# Narrative pass (batched across all candidates)
# --------------------------------------------------------------------------- #
async def enrich_with_narrative(
    jd: JobDescription,
    results: list[AnalysisResult],
    profiles: list[ResumeProfile],
    matches_per_resume: list[list[SkillMatchResult]],
    client: LLMClient | None = None,
) -> bool:
    """One batched LLM call that writes the human-readable explanations.

    Returns True when the narrative was applied. Failures are swallowed: the
    deterministic explanations written by ``analyze_one`` remain in place.
    """
    client = client or LLMClient()
    if not client.available or not results:
        return False

    candidates = [
        (
            profile,
            matches,
            {
                "ats": result.ats_score,
                "skill": result.skill_match_score,
                "alignment": result.alignment_score,
            },
        )
        for profile, matches, result in zip(profiles, matches_per_resume, results)
    ]

    try:
        data = await client.complete_json(
            prompts.INSIGHT_SYSTEM, prompts.insight_prompt(jd, candidates)
        )
    except LLMError as exc:
        logger.warning("Narrative pass failed, keeping deterministic text: %s", exc)
        return False
    except Exception:
        logger.exception("Unexpected narrative failure, keeping deterministic text.")
        return False

    analyses = data.get("analyses")
    if not isinstance(analyses, list):
        return False

    applied = False
    for result, analysis in zip(results, analyses):
        if not isinstance(analysis, dict):
            continue
        _apply_narrative(result, analysis)
        applied = True
    return applied


def _apply_narrative(result: AnalysisResult, analysis: dict) -> None:
    """Copy validated narrative fields onto a result. Numbers are never touched."""

    def clean_list(key: str, limit: int) -> list[str]:
        items = analysis.get(key) or []
        return [
            str(i).strip()
            for i in items
            if isinstance(i, str) and str(i).strip()
        ][:limit]

    explanation = str(analysis.get("alignment_explanation") or "").strip()
    if explanation:
        result.alignment_explanation = explanation[:800]

    ats_explanation = str(analysis.get("ats_explanation") or "").strip()
    if ats_explanation:
        # Keep the deterministic breakdown sentence, add the model's reading.
        result.ats_explanation = f"{result.ats_explanation} {ats_explanation[:400]}"

    strong = clean_list("strong_points", 4)
    if strong:
        result.strong_points = strong
    weak = clean_list("weak_points", 4)
    if weak:
        result.weak_points = weak

    extra = analysis.get("extra_missing_areas") or []
    known = {a.area.lower() for a in result.missing_areas}
    for item in extra:
        if not isinstance(item, dict):
            continue
        area = str(item.get("area") or "").strip()
        if not area or area.lower() in known:
            continue
        try:
            priority = Priority(str(item.get("priority", "medium")).lower())
        except ValueError:
            priority = Priority.MEDIUM
        result.missing_areas.append(
            MissingArea(
                area=area[:120],
                category=str(item.get("category") or "technical_skill")[:40],
                priority=priority,
                detail=str(item.get("detail") or "")[:300],
            )
        )
        known.add(area.lower())

    result.llm_assisted = True


# --------------------------------------------------------------------------- #
# Full session pipeline
# --------------------------------------------------------------------------- #
async def run_analysis(
    jd_document: ParsedDocument,
    resume_documents: list[ParsedDocument],
    client: LLMClient | None = None,
    progress: Optional[ProgressCallback] = None,
) -> ComparisonReport:
    """Analyse every resume against the JD and return a ranked report."""
    client = client or LLMClient()

    await _report(progress, "🔍 Extracting job description requirements…")
    jd, jd_llm = await extract_job_description(jd_document, client)

    count = len(resume_documents)
    await _report(
        progress,
        f"📄 Processing {count} resume{'s' if count != 1 else ''}…",
    )

    # Resume extraction is independent per file, so run them concurrently.
    extraction = await asyncio.gather(
        *(extract_resume_profile(doc, client) for doc in resume_documents),
        return_exceptions=True,
    )

    profiles: list[ResumeProfile] = []
    documents: list[ParsedDocument] = []
    llm_flags: list[bool] = []
    for document, outcome in zip(resume_documents, extraction):
        if isinstance(outcome, BaseException):
            logger.warning("Extraction failed for %s: %s", document.filename, outcome)
            profiles.append(heuristic_extractor.extract_resume(document))
            llm_flags.append(False)
        else:
            profile, used = outcome
            profiles.append(profile)
            llm_flags.append(used)
        documents.append(document)

    await _report(progress, "🧠 Comparing candidates against the requirements…")

    results: list[AnalysisResult] = []
    matches_per_resume: list[list[SkillMatchResult]] = []
    for profile, document, used_llm in zip(profiles, documents, llm_flags):
        result = analyze_one(jd, profile, document, llm_used=used_llm and jd_llm)
        results.append(result)
        matches_per_resume.append(
            skill_matcher.match_all(jd.all_requirements(), profile)
        )

    if client.available:
        await _report(progress, "✍️ Writing the assessment…")
        await enrich_with_narrative(jd, results, profiles, matches_per_resume, client)

    await _report(progress, "📊 Preparing results…")
    return ComparisonReport(job_title=jd.job_title, results=results)
