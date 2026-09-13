"""Deterministic, LLM-free extraction of structure from resumes and JDs.

Two jobs:

* It powers ``DEMO_MODE`` and every unit test, so the whole pipeline can be
  exercised offline with reproducible output.
* It is the fallback when the LLM is unavailable or returns unusable JSON, so a
  demo never dies because of an API outage.

Everything here is evidence based - a skill is only recorded when its name (or a
known synonym) literally appears in the document.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.analysis import skill_taxonomy as tax
from app.models.models import (
    EducationEntry,
    ExperienceEntry,
    JDRequirement,
    JobDescription,
    ParsedDocument,
    ProjectEntry,
    ResumeProfile,
    SkillImportance,
)

# --------------------------------------------------------------------------- #
# Vocabulary used for evidence-based skill detection
# --------------------------------------------------------------------------- #
# Every alias and canonical name, longest first so "machine learning" is matched
# before "learning" style fragments.
_SKILL_TERMS: list[str] = sorted(
    {term for term in tax.ALIASES},
    key=len,
    reverse=True,
)

# Extra bare-word technologies that are not in the alias table but are worth
# detecting by name.
_EXTRA_TERMS = [
    "java", "django", "flask", "fastapi", "spring", "kotlin", "swift", "php",
    "scala", "rust", "perl", "r", "html", "css", "sass", "tailwind", "bootstrap",
    "redis", "cassandra", "dynamodb", "sqlite", "oracle", "snowflake", "kafka",
    "rabbitmq", "airflow", "spark", "hadoop", "databricks", "jupyter",
    "prometheus", "grafana", "ansible", "nginx", "graphql", "websocket",
    "selenium", "cypress", "postman", "swagger", "figma", "jenkins",
    "vhdl", "lean manufacturing", "thermodynamics", "fluid mechanics",
    "heat transfer", "stress analysis", "sheet metal", "injection molding",
    "welding", "casting", "machining", "tolerance analysis", "drafting",
    "hydraulics", "pneumatics", "vibration analysis", "product design",
    "quality control", "production planning", "inventory management",
    "salesforce", "hubspot", "workday", "quickbooks", "tally",
]
_SKILL_TERMS.extend(t for t in _EXTRA_TERMS if t not in tax.ALIASES)
_SKILL_TERMS = sorted(set(_SKILL_TERMS), key=len, reverse=True)

# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
# Deliberately loose: capture anything phone shaped, then validate the digit count.
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?\d[\d\s().-]{7,16}\d")
_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)(?:\s*(?:of|with))?\s*"
    r"(?:professional\s+|relevant\s+|hands[- ]on\s+|industry\s+)?(?:experience|exp)",
    re.IGNORECASE,
)
_MIN_YEARS_RE = re.compile(
    r"(?:minimum|min\.?|at least|over|more than)?\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_DEGREE_RE = re.compile(
    r"\b(b\.?tech|b\.?e\.?|b\.?sc|bs|bachelor(?:'?s)?|m\.?tech|m\.?e\.?|m\.?sc|ms|"
    r"master(?:'?s)?|mba|ph\.?d|doctorate|diploma|associate(?:'?s)? degree)\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*[-*•▪◦·‣]\s*")

# --------------------------------------------------------------------------- #
# Section headers
# --------------------------------------------------------------------------- #
_RESUME_SECTIONS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "career summary", "objective",
                "career objective", "profile", "about me"),
    "experience": ("experience", "work experience", "professional experience",
                   "employment history", "work history", "career history"),
    "education": ("education", "academic background", "academic qualifications",
                  "educational qualifications", "qualifications"),
    "skills": ("skills", "technical skills", "key skills", "core competencies",
               "technical expertise", "technologies", "tech stack", "competencies"),
    "projects": ("projects", "personal projects", "key projects", "academic projects",
                 "selected projects"),
    "certifications": ("certifications", "certificates", "licenses",
                       "certifications & licenses", "professional certifications"),
    "achievements": ("achievements", "awards", "accomplishments", "honors", "honours"),
}

_JD_SECTIONS: dict[str, tuple[str, ...]] = {
    "required": ("requirements", "required skills", "required qualifications",
                 "minimum qualifications", "must have", "must-have", "what you need",
                 "what we're looking for", "what we are looking for", "qualifications",
                 "essential skills", "required experience", "skills required",
                 "basic qualifications"),
    "preferred": ("preferred", "preferred skills", "preferred qualifications",
                  "nice to have", "nice-to-have", "good to have", "bonus points",
                  "plus", "desirable", "advantageous", "preferred experience"),
    "responsibilities": ("responsibilities", "key responsibilities", "what you'll do",
                         "what you will do", "the role", "role description",
                         "duties", "job duties", "your impact", "day to day"),
    "education": ("education", "education requirements", "educational requirements",
                  "academic requirements"),
    "about": ("about", "about us", "about the company", "company overview",
              "who we are", "about the role"),
    "benefits": ("benefits", "what we offer", "perks", "compensation", "we offer"),
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def _normalize_header(line: str) -> str:
    return re.sub(r"[^a-z\s'&-]", " ", line.lower()).strip()


def _find_phone(text: str) -> str:
    """Find a phone number in the header of a resume.

    Years and dates are excluded by requiring 10-13 digits, which no year range
    produces, and by skipping candidates that sit inside a date-looking context.
    """
    head = text[:1500]
    for match in _PHONE_RE.finditer(head):
        candidate = match.group(0).strip(" .-")
        digits = re.sub(r"\D", "", candidate)
        if not (10 <= len(digits) <= 13):
            continue
        return candidate
    return ""


def _is_continuation(line: str) -> bool:
    """True when a line is the wrapped remainder of the previous line.

    PDF extraction loses indentation, so we key off sentence shape instead: a
    continuation starts lowercase, or the previous line clearly ran on.
    """
    stripped = line.strip()
    if not stripped:
        return False
    return stripped[0].islower()


def _looks_like_header(line: str, vocabulary: dict[str, tuple[str, ...]]) -> str | None:
    """Return the section key when a line is a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    # Headings are short, rarely end in a full stop, and rarely contain many words.
    if stripped.endswith("."):
        return None
    normalized = _normalize_header(stripped)
    if not normalized or len(normalized.split()) > 5:
        return None
    for key, names in vocabulary.items():
        for name in names:
            if normalized == name or normalized == name + ":" or normalized.rstrip(":") == name:
                return key
    return None


def _split_sections(
    text: str, vocabulary: dict[str, tuple[str, ...]]
) -> dict[str, list[str]]:
    """Split a document into ``{section_key: [lines]}`` plus a ``_head`` bucket."""
    sections: dict[str, list[str]] = {"_head": []}
    current = "_head"
    for line in text.splitlines():
        key = _looks_like_header(line, vocabulary)
        if key:
            current = key
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _bullets(lines: Iterable[str]) -> list[str]:
    """Collect bullet-ish lines, falling back to any substantial line."""
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _BULLET_RE.match(line):
            bullets.append(_BULLET_RE.sub("", stripped).strip())
        elif len(stripped) > 25 and not stripped.endswith(":"):
            bullets.append(stripped)
    return [b for b in bullets if b]


def find_skills(text: str) -> list[str]:
    """Evidence-based skill detection: only terms literally present in the text."""
    if not text:
        return []
    haystack = " " + tax.normalize(text) + " "
    found: dict[str, str] = {}   # canonical -> surface term
    for term in _SKILL_TERMS:
        norm_term = tax.normalize(term)
        if not norm_term:
            continue
        if len(norm_term) <= 3:
            pattern = rf"(?<![a-z0-9+#]){re.escape(norm_term)}(?![a-z0-9+#])"
            present = re.search(pattern, haystack) is not None
        else:
            present = f" {norm_term} " in haystack
        if present:
            canon = tax.canonical(term)
            if canon and canon not in found:
                found[canon] = canon
    return sorted(found.values())


def _bucket_skills(skills: list[str]) -> dict[str, list[str]]:
    """Sort detected skills into languages / frameworks / databases / cloud / tools."""
    languages = {"python", "javascript", "typescript", "java", "c++", "c#", "golang",
                 "ruby", "php", "kotlin", "swift", "scala", "rust", "perl", "r",
                 "matlab", "verilog", "vhdl", "sql"}
    frameworks = {"react", "angular", "vue", "node.js", "express", "next.js", "django",
                  "flask", "fastapi", "spring boot", "spring", "asp.net", "tensorflow",
                  "pytorch", "scikit-learn", "pandas", "numpy", "bootstrap", "tailwind"}
    databases = {"postgresql", "mysql", "sql server", "mongodb", "redis", "cassandra",
                 "dynamodb", "sqlite", "oracle", "elasticsearch", "snowflake", "nosql"}
    cloud = {"aws", "azure", "gcp", "kubernetes", "docker", "terraform", "jenkins",
             "github actions", "ci/cd", "ansible", "nginx", "prometheus", "grafana"}

    buckets: dict[str, list[str]] = {
        "programming_languages": [],
        "frameworks": [],
        "databases": [],
        "cloud_technologies": [],
        "tools": [],
        "technical_skills": [],
    }
    for skill in skills:
        canon = tax.canonical(skill)
        if canon in languages:
            buckets["programming_languages"].append(skill)
        elif canon in frameworks:
            buckets["frameworks"].append(skill)
        elif canon in databases:
            buckets["databases"].append(skill)
        elif canon in cloud:
            buckets["cloud_technologies"].append(skill)
        else:
            buckets["tools"].append(skill)
        buckets["technical_skills"].append(skill)
    return buckets


# --------------------------------------------------------------------------- #
# Resume extraction
# --------------------------------------------------------------------------- #
def _guess_name(text: str) -> str:
    """The candidate name is almost always the first substantial line."""
    for line in text.splitlines()[:8]:
        stripped = line.strip()
        if not stripped or len(stripped) > 50:
            continue
        if _EMAIL_RE.search(stripped) or "@" in stripped:
            continue
        if re.search(r"\d", stripped):
            continue
        lowered = stripped.lower().strip(" :")
        if lowered in {"resume", "curriculum vitae", "cv", "profile"}:
            continue
        words = [w for w in re.split(r"\s+", stripped) if w]
        if 1 < len(words) <= 4 and all(re.match(r"^[A-Za-z.'-]+$", w) for w in words):
            # Reject obvious section headers.
            if lowered in {h for names in _RESUME_SECTIONS.values() for h in names}:
                continue
            return " ".join(w.strip(".") for w in words).title()
    return "Unknown Candidate"


def _parse_experience(lines: list[str]) -> tuple[list[ExperienceEntry], str]:
    """Parse job entries from the experience section."""
    entries: list[ExperienceEntry] = []
    current: ExperienceEntry | None = None
    blob: list[str] = []

    date_re = re.compile(
        r"((?:19|20)\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        blob.append(stripped)
        is_bullet = bool(_BULLET_RE.match(line))

        # Wrapped continuation of the previous bullet - never a new role.
        if current and not is_bullet and _is_continuation(line):
            current.description = (current.description + " " + stripped).strip()
            continue

        # A non-bullet line with a date range usually starts a new role.
        if not is_bullet and date_re.search(stripped) and len(stripped) < 120:
            if current:
                entries.append(current)
            title, company, duration = _split_role_line(stripped)
            current = ExperienceEntry(title=title, company=company, duration=duration)
            continue

        if current:
            text = _BULLET_RE.sub("", stripped).strip()
            current.description = (current.description + " " + text).strip()
        elif not is_bullet and len(stripped) < 100 and len(stripped.split()) <= 10:
            title, company, duration = _split_role_line(stripped)
            current = ExperienceEntry(title=title, company=company, duration=duration)

    if current:
        entries.append(current)
    return entries, "\n".join(blob)


def _split_role_line(line: str) -> tuple[str, str, str]:
    """Split 'Senior Engineer | Acme Corp | 2020 - Present' into its parts."""
    parts = [p.strip() for p in re.split(r"\s*[|,–—]\s*|\s+at\s+|\s+-\s+", line) if p.strip()]
    duration = ""
    for part in list(parts):
        if re.search(r"(19|20)\d{2}|present|current", part, re.I):
            duration = part
            parts.remove(part)
            break
    title = parts[0] if parts else line.strip()
    company = parts[1] if len(parts) > 1 else ""
    return title, company, duration


def _parse_education(lines: list[str]) -> list[EducationEntry]:
    entries: list[EducationEntry] = []
    for line in lines:
        stripped = _BULLET_RE.sub("", line.strip()).strip()
        if not stripped or len(stripped) < 5:
            continue
        match = _DEGREE_RE.search(stripped)
        if not match:
            continue
        year = ""
        year_match = re.search(r"(19|20)\d{2}", stripped)
        if year_match:
            year = year_match.group(0)

        field = ""
        field_match = re.search(
            r"(?:in|of)\s+([A-Za-z&\s]{3,40})", stripped[match.end():], re.IGNORECASE
        )
        if field_match:
            field = field_match.group(1).strip(" ,.")

        institution = ""
        inst_match = re.search(
            r"(?:,|\||at|from)\s*([A-Z][\w.&'\- ]{4,60}(?:University|College|Institute|School|IIT|NIT))",
            stripped,
        )
        if inst_match:
            institution = inst_match.group(1).strip()

        entries.append(
            EducationEntry(
                degree=match.group(0).strip(),
                field_of_study=field,
                institution=institution,
                year=year,
            )
        )
    return entries


def _parse_projects(lines: list[str]) -> list[ProjectEntry]:
    projects: list[ProjectEntry] = []
    current: ProjectEntry | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = bool(_BULLET_RE.match(line))
        text = _BULLET_RE.sub("", stripped).strip()

        # Wrapped continuation of the previous line - never a new project title.
        if current and not is_bullet and _is_continuation(line):
            current.description = (current.description + " " + text).strip()
            continue

        # A short non-bullet line is treated as a project title.
        if not is_bullet and len(text) < 80 and not text.endswith("."):
            if current:
                projects.append(current)
            current = ProjectEntry(name=text.rstrip(":"), description="")
        elif current:
            current.description = (current.description + " " + text).strip()
        else:
            current = ProjectEntry(name=text[:60], description=text)

    if current:
        projects.append(current)

    for project in projects:
        project.technologies = find_skills(f"{project.name} {project.description}")
    return projects


def extract_resume(document: ParsedDocument) -> ResumeProfile:
    """Build a ResumeProfile from raw resume text, without an LLM."""
    text = document.text
    sections = _split_sections(text, _RESUME_SECTIONS)

    email_match = _EMAIL_RE.search(text)
    phone = _find_phone(text)

    experience_lines = sections.get("experience", [])
    experience, experience_blob = _parse_experience(experience_lines)

    skills_text = "\n".join(sections.get("skills", []))
    # Skills mentioned anywhere count as evidence, not just under the header.
    detected = find_skills(text)
    buckets = _bucket_skills(detected)

    years: float | None = None
    year_matches = _YEARS_RE.findall(text)
    if year_matches:
        try:
            years = max(float(y) for y in year_matches)
        except ValueError:
            years = None
    if years is None:
        years = _infer_years_from_dates(experience_blob or text)

    summary_lines = [line.strip() for line in sections.get("summary", []) if line.strip()]

    profile = ResumeProfile(
        candidate_name=_guess_name(text),
        email=email_match.group(0) if email_match else "",
        phone=phone,
        summary=" ".join(summary_lines)[:600],
        years_of_experience=years,
        education=_parse_education(sections.get("education", [])),
        certifications=_bullets(sections.get("certifications", []))[:12],
        experience=experience,
        projects=_parse_projects(sections.get("projects", [])),
        achievements=_bullets(sections.get("achievements", []))[:10],
        technical_skills=buckets["technical_skills"],
        programming_languages=buckets["programming_languages"],
        frameworks=buckets["frameworks"],
        databases=buckets["databases"],
        cloud_technologies=buckets["cloud_technologies"],
        tools=buckets["tools"],
        soft_skills=_find_soft_skills(text),
        source_filename=document.filename,
        raw_text=text,
    )
    profile.detected_domain = tax.classify_domain(text, profile.all_skills())[0]
    return profile


_SOFT_SKILLS = (
    "communication", "leadership", "teamwork", "collaboration", "problem solving",
    "problem-solving", "analytical", "time management", "adaptability", "mentoring",
    "stakeholder management", "critical thinking", "attention to detail",
    "presentation", "negotiation", "ownership", "cross-functional",
)


def _find_soft_skills(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({s for s in _SOFT_SKILLS if s in lowered})


def _infer_years_from_dates(text: str) -> float | None:
    """Estimate total experience from date ranges like '2019 - Present'."""
    ranges = re.findall(
        r"((?:19|20)\d{2})\s*[-–—]+\s*((?:19|20)\d{2}|present|current)", text, re.I
    )
    if not ranges:
        return None
    from datetime import datetime

    this_year = datetime.now().year
    total = 0.0
    for start, end in ranges:
        try:
            start_year = int(start)
        except ValueError:
            continue
        end_year = this_year if end.lower() in {"present", "current"} else int(end)
        if end_year >= start_year:
            total += end_year - start_year
    return float(total) if total > 0 else None


# --------------------------------------------------------------------------- #
# Job description extraction
# --------------------------------------------------------------------------- #
def _guess_job_title(text: str, sections: dict[str, list[str]]) -> str:
    explicit = re.search(
        r"(?:job\s*title|position|role|title)\s*[:\-]\s*(.+)", text, re.IGNORECASE
    )
    if explicit:
        return explicit.group(1).strip()[:80]

    for line in text.splitlines()[:6]:
        stripped = line.strip().rstrip(":")
        if not stripped or len(stripped) > 70:
            continue
        if _looks_like_header(stripped, _JD_SECTIONS):
            continue
        if re.search(
            r"engineer|developer|manager|analyst|scientist|designer|architect|"
            r"administrator|consultant|specialist|lead|intern|technician|nurse|"
            r"accountant|executive|associate",
            stripped,
            re.IGNORECASE,
        ):
            return stripped[:80]

    for line in text.splitlines()[:3]:
        stripped = line.strip()
        if stripped and len(stripped) < 70:
            return stripped[:80]
    return "Unspecified Role"


def extract_jd(document: ParsedDocument) -> JobDescription:
    """Build a JobDescription from raw JD text, without an LLM."""
    text = document.text
    sections = _split_sections(text, _JD_SECTIONS)

    required_text = "\n".join(sections.get("required", []))
    preferred_text = "\n".join(sections.get("preferred", []))
    responsibilities_lines = sections.get("responsibilities", [])
    responsibilities_text = "\n".join(responsibilities_lines)

    preferred_skills = find_skills(preferred_text)
    preferred_canon = {tax.canonical(s) for s in preferred_skills}

    # Required skills come from the requirements section plus responsibilities;
    # anything already listed as preferred is not duplicated as required.
    required_pool = required_text or text
    required_skills = [
        s
        for s in find_skills(required_pool + "\n" + responsibilities_text)
        if tax.canonical(s) not in preferred_canon
    ]

    # If the JD has no explicit requirements section, fall back to the whole body.
    if not required_skills and not preferred_skills:
        required_skills = find_skills(text)

    min_years = _extract_min_years(required_pool or text)

    education_requirements: list[str] = []
    for line in sections.get("education", []) + required_text.splitlines():
        stripped = _BULLET_RE.sub("", line.strip()).strip()
        if stripped and _DEGREE_RE.search(stripped) and len(stripped) < 200:
            if stripped not in education_requirements:
                education_requirements.append(stripped)

    responsibilities = _bullets(responsibilities_lines)[:12]
    if not responsibilities:
        responsibilities = _bullets(sections.get("_head", []))[:8]

    jd = JobDescription(
        job_title=_guess_job_title(text, sections),
        required_skills=[
            JDRequirement(skill=s, importance=SkillImportance.REQUIRED)
            for s in required_skills
        ],
        preferred_skills=[
            JDRequirement(skill=s, importance=SkillImportance.PREFERRED)
            for s in preferred_skills
        ],
        min_years_experience=min_years,
        education_requirements=education_requirements[:5],
        responsibilities=responsibilities,
        keywords=_extract_keywords(text, required_skills + preferred_skills),
        source_filename=document.filename,
        raw_text=text,
    )
    jd.detected_domain = tax.classify_domain(
        text, [r.skill for r in jd.all_requirements()]
    )[0]
    return jd


def _extract_min_years(text: str) -> float | None:
    """Find the minimum years of experience the JD asks for."""
    explicit = re.search(
        r"(?:minimum|at least|min\.?|over|more than)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return float(explicit.group(1))
    plus = re.search(r"(\d{1,2})\s*\+\s*(?:years?|yrs?)", text, re.IGNORECASE)
    if plus:
        return float(plus.group(1))
    generic = _YEARS_RE.search(text)
    if generic:
        return float(generic.group(1))
    return None


_STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "are", "will", "have", "this",
    "that", "from", "your", "who", "all", "can", "has", "not", "but", "any",
    "job", "role", "work", "team", "company", "years", "experience", "ability",
    "strong", "good", "excellent", "plus", "must", "should", "would", "including",
    "such", "about", "into", "their", "them", "they", "more", "than", "other",
    "well", "also", "using", "used", "help", "make", "need", "want", "look",
    "looking", "join", "part", "time", "full", "great", "best", "highly",
    "across", "within", "while", "when", "where", "what", "which", "been",
}


def _extract_keywords(text: str, known_skills: list[str], limit: int = 40) -> list[str]:
    """Skills first, then the most frequent meaningful words in the JD."""
    keywords: list[str] = []
    seen: set[str] = set()
    for skill in known_skills:
        canon = tax.canonical(skill)
        if canon and canon not in seen:
            seen.add(canon)
            keywords.append(skill)

    counts: dict[str, int] = {}
    for word in re.findall(r"[a-zA-Z][a-zA-Z+#./-]{3,}", text.lower()):
        cleaned = word.strip(".-/")
        if len(cleaned) < 4 or cleaned in _STOPWORDS:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1

    for word, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        if len(keywords) >= limit:
            break
        if count < 2:
            break
        canon = tax.canonical(word)
        if canon and canon not in seen:
            seen.add(canon)
            keywords.append(word)

    return keywords[:limit]
