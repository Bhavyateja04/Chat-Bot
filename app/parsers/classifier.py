"""Heuristic classification of an uploaded document as a JD or a resume.

The bot is supposed to figure this out on its own so the user never has to say
"this is the JD". Filename hints and body-text signals are combined into a score;
when the margin between the two is thin the bot asks the user instead of guessing.
"""

from __future__ import annotations

import re

from app.models.models import DocumentKind

# --- filename hints ------------------------------------------------------- #
_JD_FILENAME_HINTS = (
    "jd", "job_description", "job-description", "jobdescription", "job_desc",
    "job", "role", "position", "vacancy", "opening", "requisition", "hiring",
)
_RESUME_FILENAME_HINTS = ("resume", "cv", "curriculum", "candidate", "profile", "bio_data", "biodata")

# --- body-text signals ---------------------------------------------------- #
_JD_PHRASES = (
    "we are looking for", "we are seeking", "you will", "you'll", "join our team",
    "about the role", "about the company", "about us", "responsibilities",
    "key responsibilities", "what you'll do", "requirements", "qualifications",
    "required qualifications", "preferred qualifications", "minimum qualifications",
    "nice to have", "what we offer", "benefits", "we offer", "apply now",
    "job description", "job title", "employment type", "the ideal candidate",
    "the successful candidate", "reports to", "salary", "compensation",
    "equal opportunity", "full-time", "job type", "location:",
)
_RESUME_PHRASES = (
    "work experience", "professional experience", "employment history",
    "career summary", "professional summary", "objective", "career objective",
    "education", "academic background", "certifications", "projects",
    "personal projects", "technical skills", "key skills", "achievements",
    "awards", "curriculum vitae", "resume", "references available",
    "linkedin.com/in", "github.com/", "declaration", "date of birth",
    "languages known", "hobbies", "extracurricular",
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
_YEAR_RANGE_RE = re.compile(r"\b(19|20)\d{2}\s*[-–—to]+\s*((19|20)\d{2}|present|current)\b", re.I)


def _filename_signal(filename: str) -> tuple[float, float]:
    """Return (jd_points, resume_points) from the filename alone."""
    name = filename.lower().replace(" ", "_")
    stem = re.sub(r"\.[a-z0-9]+$", "", name)

    jd_points = 0.0
    resume_points = 0.0

    for hint in _RESUME_FILENAME_HINTS:
        if hint in stem:
            resume_points += 3.0
            break
    for hint in _JD_FILENAME_HINTS:
        # Match as a token so "jd" does not fire inside "jdoe".
        if re.search(rf"(?<![a-z]){re.escape(hint)}(?![a-z])", stem):
            jd_points += 3.0
            break

    return jd_points, resume_points


# Real resumes and JDs score 12+ on this scale; unrelated documents (invoices,
# articles, certificates, notes) score 0-3. Anything below this has no business
# being treated as either.
MIN_DOCUMENT_SIGNAL = 5.0


def document_scores(text: str, filename: str = "") -> tuple[float, float]:
    """Raw ``(jd_points, resume_points)`` evidence totals for a document.

    Exposed separately from ``classify_document`` because the *absolute* amount
    of evidence matters as much as the ratio: a shopping list is not a
    50/50 toss-up between a JD and a resume, it is simply neither.
    """
    body = (text or "").lower()
    if not body.strip():
        return 0.0, 0.0

    jd_points, resume_points = _filename_signal(filename)

    for phrase in _JD_PHRASES:
        if phrase in body:
            jd_points += 1.0
    for phrase in _RESUME_PHRASES:
        if phrase in body:
            resume_points += 1.0

    # Contact details at the top are a strong resume signal.
    head = body[:600]
    if _EMAIL_RE.search(head):
        resume_points += 2.0
    if _PHONE_RE.search(head):
        resume_points += 2.0

    # Employment date ranges ("2019 - Present") are typical of resumes.
    if len(_YEAR_RANGE_RE.findall(body)) >= 2:
        resume_points += 2.0

    # First-person pronouns lean resume; second-person leans JD.
    resume_points += min(body.count(" i "), 3) * 0.5
    jd_points += min(len(re.findall(r"\byou(?:r|'ll| will)?\b", body)), 6) * 0.5

    return jd_points, resume_points


def has_enough_signal(text: str, filename: str = "") -> bool:
    """True when a document looks like a JD or a resume at all."""
    jd_points, resume_points = document_scores(text, filename)
    return (jd_points + resume_points) >= MIN_DOCUMENT_SIGNAL


def classify_document(text: str, filename: str = "") -> tuple[DocumentKind, float]:
    """Classify a document.

    Returns ``(kind, confidence)`` where confidence is 0-1. A confidence below
    ~0.6 means the caller should ask the user to disambiguate. A document with
    too little evidence to be either is UNKNOWN with zero confidence.
    """
    jd_points, resume_points = document_scores(text, filename)

    total = jd_points + resume_points
    if total < MIN_DOCUMENT_SIGNAL:
        return DocumentKind.UNKNOWN, 0.0

    if jd_points > resume_points:
        return DocumentKind.JOB_DESCRIPTION, round(jd_points / total, 2)
    if resume_points > jd_points:
        return DocumentKind.RESUME, round(resume_points / total, 2)
    return DocumentKind.UNKNOWN, 0.5
