"""Pydantic domain models.

These are the contract between the parsers, the LLM, the scoring engine and the
Discord formatter. Every LLM response is validated into these models before any
of it reaches a user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DocumentKind(str, Enum):
    JOB_DESCRIPTION = "job_description"
    RESUME = "resume"
    UNKNOWN = "unknown"


class SkillImportance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class MatchStatus(str, Enum):
    """How well a single JD skill is covered by the resume."""

    EXACT = "exact"        # the resume names the skill (or a true synonym)
    PARTIAL = "partial"    # the resume shows a genuinely related skill
    MISSING = "missing"    # no evidence found


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def emoji(self) -> str:
        return {"high": "\U0001F7E2", "medium": "\U0001F7E1", "low": "\U0001F534"}[self.value]


class SessionState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_JD = "WAITING_FOR_JD"
    JD_RECEIVED = "JD_RECEIVED"
    WAITING_FOR_RESUMES = "WAITING_FOR_RESUMES"
    RESUMES_RECEIVED = "RESUMES_RECEIVED"
    ANALYZING = "ANALYZING"
    RESULTS = "RESULTS"
    AWAITING_DISAMBIGUATION = "AWAITING_DISAMBIGUATION"


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class ParsedDocument(BaseModel):
    """The raw text pulled out of an uploaded file, plus parse quality signals."""

    filename: str
    text: str
    char_count: int = 0
    page_count: int = 0
    file_type: str = ""
    extraction_ok: bool = True
    warnings: list[str] = Field(default_factory=list)

    @field_validator("char_count")
    @classmethod
    def _default_char_count(cls, v: int, info) -> int:
        if v:
            return v
        text = info.data.get("text") or ""
        return len(text)

    @property
    def is_usable(self) -> bool:
        """Enough readable text to bother analysing."""
        return self.extraction_ok and len(self.text.strip()) >= 100


# --------------------------------------------------------------------------- #
# Structured extraction
# --------------------------------------------------------------------------- #
class ExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""


class EducationEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    year: str = ""


class ProjectEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Everything we could extract from one resume. Nothing is invented."""

    model_config = ConfigDict(extra="ignore")

    candidate_name: str = "Unknown Candidate"
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""

    years_of_experience: Optional[float] = None
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

    technical_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    cloud_technologies: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    detected_domain: str = ""
    source_filename: str = ""
    raw_text: str = Field(default="", exclude=True)

    def all_skills(self) -> list[str]:
        """Every skill-ish token we found, de-duplicated, order preserved."""
        merged: list[str] = []
        seen: set[str] = set()
        for bucket in (
            self.technical_skills,
            self.programming_languages,
            self.frameworks,
            self.databases,
            self.cloud_technologies,
            self.tools,
            self.certifications,
        ):
            for item in bucket:
                key = item.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(item.strip())
        return merged

    def has_structure(self) -> bool:
        """Did the resume parse into recognisable sections?"""
        return bool(self.experience or self.education or self.projects)


class JDRequirement(BaseModel):
    """A single requirement lifted out of the job description."""

    model_config = ConfigDict(extra="ignore")

    skill: str
    importance: SkillImportance = SkillImportance.REQUIRED
    context: str = ""

    @field_validator("skill")
    @classmethod
    def _clean(cls, v: str) -> str:
        return v.strip()


class JobDescription(BaseModel):
    """Structured view of the JD."""

    model_config = ConfigDict(extra="ignore")

    job_title: str = "Unspecified Role"
    company: str = ""
    seniority: str = ""

    required_skills: list[JDRequirement] = Field(default_factory=list)
    preferred_skills: list[JDRequirement] = Field(default_factory=list)

    min_years_experience: Optional[float] = None
    education_requirements: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    domain_knowledge: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    detected_domain: str = ""
    source_filename: str = ""
    raw_text: str = Field(default="", exclude=True)

    def all_requirements(self) -> list[JDRequirement]:
        return list(self.required_skills) + list(self.preferred_skills)

    def is_complete(self) -> bool:
        """Enough substance to score against with confidence."""
        return bool(self.required_skills) and bool(self.job_title)


# --------------------------------------------------------------------------- #
# Matching + scoring
# --------------------------------------------------------------------------- #
class SkillMatchResult(BaseModel):
    """One JD requirement judged against the resume."""

    skill: str
    importance: SkillImportance
    status: MatchStatus
    matched_via: str = ""      # what in the resume satisfied it
    relation: str = ""         # why a partial match counts as related
    evidence: str = ""         # quote/snippet from the resume

    @property
    def credit(self) -> float:
        """Score credit awarded. Partial matches deliberately earn only half."""
        return {
            MatchStatus.EXACT: 1.0,
            MatchStatus.PARTIAL: 0.5,
            MatchStatus.MISSING: 0.0,
        }[self.status]


class ATSBreakdown(BaseModel):
    """Per-component ATS scores (0-100 each) plus the weights that produced them."""

    keyword_match: float = 0.0
    required_skills: float = 0.0
    experience: float = 0.0
    projects: float = 0.0
    education: float = 0.0
    resume_structure: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Keyword Match": self.keyword_match,
            "Required Skills": self.required_skills,
            "Experience": self.experience,
            "Projects / Responsibilities": self.projects,
            "Education / Certifications": self.education,
            "Resume Structure": self.resume_structure,
        }


class AlignmentBreakdown(BaseModel):
    technical_skills: float = 0.0
    experience: float = 0.0
    projects: float = 0.0
    responsibilities: float = 0.0
    keywords: float = 0.0
    education: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Technical Skills": self.technical_skills,
            "Experience": self.experience,
            "Projects": self.projects,
            "Responsibilities": self.responsibilities,
            "Keywords": self.keywords,
            "Education": self.education,
        }


class MissingArea(BaseModel):
    """A gap that is broader than a single missing keyword."""

    area: str
    category: str = "technical_skill"   # technical_skill | tool | experience |
                                        # domain_knowledge | certification |
                                        # responsibility | project_evidence
    priority: Priority = Priority.MEDIUM
    detail: str = ""


class CourseRecommendation(BaseModel):
    title: str
    provider: str = ""
    skill: str = ""
    reason: str = ""
    priority: Priority = Priority.MEDIUM
    # Either a curated official course/provider page, or a provider search URL.
    # Never a generated deep link - see recommender.course_url().
    url: str = ""


class RoadmapItem(BaseModel):
    priority: int = 1
    skill: str
    topics: list[str] = Field(default_factory=list)
    why: str = ""


class RoleMismatch(BaseModel):
    """Domain-level mismatch detection (e.g. mechanical JD vs software resume)."""

    detected: bool = False
    jd_role: str = ""
    resume_profile: str = ""
    jd_domain: str = ""
    resume_domain: str = ""
    severity: Priority = Priority.LOW
    reason: str = ""
    critical_missing: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """The complete analysis of one resume against one JD."""

    candidate_name: str
    source_filename: str = ""
    job_title: str = ""

    ats_score: float = 0.0
    ats_breakdown: ATSBreakdown = Field(default_factory=ATSBreakdown)
    ats_explanation: str = ""

    skill_match_score: float = 0.0
    matched_skills: list[SkillMatchResult] = Field(default_factory=list)
    partial_skills: list[SkillMatchResult] = Field(default_factory=list)
    missing_skills: list[SkillMatchResult] = Field(default_factory=list)

    missing_areas: list[MissingArea] = Field(default_factory=list)

    alignment_score: float = 0.0
    alignment_breakdown: AlignmentBreakdown = Field(default_factory=AlignmentBreakdown)
    alignment_explanation: str = ""
    strong_points: list[str] = Field(default_factory=list)
    weak_points: list[str] = Field(default_factory=list)

    recommended_courses: list[CourseRecommendation] = Field(default_factory=list)
    learning_roadmap: list[RoadmapItem] = Field(default_factory=list)

    role_mismatch: RoleMismatch = Field(default_factory=RoleMismatch)
    confidence: Confidence = Confidence.MEDIUM
    confidence_reason: str = ""

    improvement_suggestions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_assisted: bool = False

    @property
    def verdict(self) -> str:
        if self.role_mismatch.detected:
            return "Role mismatch"
        if self.ats_score >= 80:
            return "Strong fit"
        if self.ats_score >= 65:
            return "Good fit"
        if self.ats_score >= 45:
            return "Partial fit"
        return "Weak fit"


class ComparisonReport(BaseModel):
    """Ranking across several resumes analysed against the same JD."""

    job_title: str = ""
    results: list[AnalysisResult] = Field(default_factory=list)

    @property
    def ranked(self) -> list[AnalysisResult]:
        """Best first. Ties broken by alignment, then skill match."""
        return sorted(
            self.results,
            key=lambda r: (r.ats_score, r.alignment_score, r.skill_match_score),
            reverse=True,
        )

    @property
    def best(self) -> Optional[AnalysisResult]:
        ranked = self.ranked
        return ranked[0] if ranked else None
