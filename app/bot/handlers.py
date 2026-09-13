"""Conversation logic, deliberately free of any Discord types.

Handlers take a session plus plain ``(filename, bytes)`` uploads and return a
list of message strings. That keeps the whole conversation testable without a
Discord connection, and keeps ``discord_bot.py`` down to plumbing.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from app.analysis.analyzer import run_analysis
from app.ai.client import LLMClient
from app.config import get_settings
from app.formatting import discord_format
from app.models.models import DocumentKind, ParsedDocument, SessionState
from app.parsers import loader
from app.parsers.classifier import classify_document, has_enough_signal
from app.bot.session import Session

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]

# Below this confidence we ask the user rather than guessing the document type.
_DISAMBIGUATION_THRESHOLD = 0.60
# Once a JD is set, an upload must look at least this much like a resume to be
# accepted. Anything weaker is rejected rather than silently scored.
_RESUME_CONFIDENCE_FLOOR = 0.55

WELCOME = """# 👋 Welcome to ResumeMatch AI!

I compare resumes against a Job Description and give you a recruiter-grade
breakdown for each candidate.

**Please provide:**
1️⃣ A **Job Description**
2️⃣ **One or more Resumes**

Just upload the files here — I work out which is which automatically.
You can upload several resumes for the same JD and I'll rank them.

**Supported formats:** PDF, DOCX, TXT
**Commands:** `/help` `/status` `/analyze` `/reset`
"""

HELP = """# 📖 ResumeMatch AI — Help

**How it works**
1. Upload a **Job Description** (PDF / DOCX / TXT).
2. Upload **one or more resumes** — attach several files at once if you like.
3. I analyse each resume and rank the candidates.

**What you get for every resume**
🎯 **ATS Score** with a six-component weighted breakdown
🧩 **Skill Set Match** — strong / partial / missing, required vs preferred
❌ **Missing Areas** prioritised high / medium / low
📈 **Job Alignment** across six dimensions, with an explanation
📚 **Recommended Courses** based only on real gaps
🧠 **Learning Roadmap** — what to actually study
⚠️ **Role Mismatch Detection** — a mechanical JD won't score a software resume highly

**Commands**
`/start` — begin a new analysis
`/analyze` — force analysis with what you've uploaded
`/status` — show what I currently have
`/reset` — clear the session and start over
`/help` — this message

**Tips**
• Text-based PDFs work best — scanned/photo PDFs have no text layer to read.
• Name files clearly (`job_description.pdf`, `resume_priya.pdf`) and I'll classify them faster.
• You can also paste a JD as plain text with `/jd <paste the text>`.
"""


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def handle_start(session: Session) -> list[str]:
    session.start()
    return [WELCOME]


def handle_help() -> list[str]:
    return [HELP]


def handle_reset(session: Session) -> list[str]:
    session.reset()
    return [
        "🔄 **Session cleared.** The job description and all uploaded resumes have "
        "been discarded.\n\nUpload a new job description whenever you're ready."
    ]


def handle_status(session: Session) -> list[str]:
    if session.state == SessionState.IDLE and not session.has_jd:
        return [
            "📭 **Nothing uploaded yet.**\n\n"
            "Upload a job description to get started, or run `/help` to see how it works."
        ]
    lines = ["## 📊 Current Session", session.summary()]
    if session.is_ready:
        lines.append("\n✅ Ready to analyse — run `/analyze` or upload more resumes.")
    elif session.has_jd:
        lines.append("\n📄 Waiting for at least one resume.")
    else:
        lines.append("\n📋 Waiting for a job description.")
    return ["\n".join(lines)]


# --------------------------------------------------------------------------- #
# Text input (pasted JD)
# --------------------------------------------------------------------------- #
def handle_pasted_jd(session: Session, text: str) -> list[str]:
    """Accept a JD pasted directly into the chat."""
    from app.parsers.text_parser import from_string

    document = from_string(text, "pasted-job-description.txt")
    if len(document.text.strip()) < 100:
        return [
            "⚠️ That job description looks too short to analyse "
            f"({len(document.text.strip())} characters). "
            "Please paste the full JD or upload it as a file."
        ]
    session.set_jd(document)
    return [
        "✅ **Job description received** (pasted text).\n\n"
        "Now upload one or more resumes — you can attach several at once."
    ]


# --------------------------------------------------------------------------- #
# File uploads
# --------------------------------------------------------------------------- #
def handle_uploads(
    session: Session, uploads: list[tuple[str, bytes]]
) -> tuple[list[str], bool]:
    """Process a batch of uploaded files.

    Returns ``(messages, ready_to_analyze)``.
    """
    settings = get_settings()
    messages: list[str] = []

    if not uploads:
        return [], False

    if session.file_count + len(uploads) > settings.max_files_per_session:
        remaining = max(0, settings.max_files_per_session - session.file_count)
        return (
            [
                f"⚠️ **Too many files.** A session holds at most "
                f"{settings.max_files_per_session} documents and you have "
                f"{session.file_count} already"
                + (
                    f" — room for {remaining} more."
                    if remaining
                    else ". Run `/reset` to start a new session."
                )
            ],
            False,
        )

    accepted_resumes = 0
    accepted_jd = False
    deferred: list[str] = []

    for index, (filename, data) in enumerate(uploads):
        document = loader.parse_document(data, filename)

        if not document.extraction_ok:
            messages.append(_extraction_failure_message(document))
            continue

        kind, confidence = classify_document(document.text, document.filename)

        # A document that is neither a JD nor a resume is rejected outright -
        # asking "is this the JD or a resume?" about an invoice helps nobody.
        if not has_enough_signal(document.text, document.filename):
            messages.append(
                _wrong_document_message(
                    document, "resume" if session.has_jd else "jd"
                )
            )
            continue

        if not session.has_jd:
            if kind == DocumentKind.JOB_DESCRIPTION and confidence >= _DISAMBIGUATION_THRESHOLD:
                session.set_jd(document)
                accepted_jd = True
                messages.append(_jd_accepted_message(document, confidence))
            elif kind == DocumentKind.RESUME and confidence >= _DISAMBIGUATION_THRESHOLD:
                # A resume arriving first is held rather than discarded, but the
                # bot still needs the JD before it can do anything with it.
                session.add_resume(document)
                accepted_resumes += 1
                messages.append(
                    _wrong_document_message(document, "jd", detected="resume")
                    + f"\n_(I've saved `{document.filename}` as a resume for this "
                    "session, so you don't need to upload it again.)_"
                )
            else:
                session.await_disambiguation(document)
                messages.append(
                    f"🤔 I can't tell whether **`{document.filename}`** is the job "
                    "description or a resume.\n\n"
                    "Reply **`jd`** if it's the job description, or **`resume`** if "
                    "it's a resume."
                )
                # Stop processing this batch until the ambiguity is resolved, and
                # tell the user which files still need re-uploading.
                deferred = [name for name, _ in uploads[index + 1 :]]
                break
        else:
            # The JD is already set, so from here on only resumes are accepted.
            if kind == DocumentKind.JOB_DESCRIPTION and confidence >= 0.7:
                messages.append(
                    _wrong_document_message(document, "resume", detected="jd")
                    + "\n_I already have a job description for this session. To "
                    "analyse against a different one, run `/reset` first._"
                )
                continue
            if kind == DocumentKind.UNKNOWN or confidence < _RESUME_CONFIDENCE_FLOOR:
                messages.append(_wrong_document_message(document, "resume"))
                continue

            session.add_resume(document)
            accepted_resumes += 1
            messages.append(
                f"📄 **`{document.filename}`** received as a **resume** "
                f"({document.char_count:,} characters read)."
            )

        if document.warnings:
            messages.append(
                "⚠️ " + " ".join(f"_{w}_" for w in document.warnings[:2])
            )

    if deferred:
        names = ", ".join(f"`{n}`" for n in deferred)
        messages.append(
            f"⏸️ I haven't processed {names} yet. Answer the question above first, "
            "then upload them again."
        )

    # --- next-step guidance ---
    if accepted_jd and session.resume_count == 0:
        messages.append(
            "**Now upload one or more resumes.** You can attach several files at once."
        )
    elif accepted_resumes and session.has_jd:
        count = session.resume_count
        messages.append(
            f"✅ **{count} resume{'s' if count != 1 else ''} ready.**\n"
            "Upload more, or run **`/analyze`** to start the analysis."
        )

    ready = session.is_ready and accepted_resumes > 0
    return messages, ready


def handle_disambiguation(session: Session, answer: str) -> list[str]:
    """Resolve a pending ambiguous document from the user's reply."""
    document = session.pending_document
    if document is None:
        return []

    normalized = answer.strip().lower()
    if normalized in {"jd", "job", "job description", "jobdescription", "1"}:
        session.set_jd(document)
        return [
            f"✅ **`{document.filename}`** saved as the **job description**.\n\n"
            "Now upload one or more resumes."
        ]
    if normalized in {"resume", "cv", "2"}:
        session.add_resume(document)
        if session.has_jd:
            return [
                f"✅ **`{document.filename}`** saved as a **resume**.\n"
                "Run **`/analyze`** when you're ready."
            ]
        return [
            f"✅ **`{document.filename}`** saved as a **resume**.\n"
            "I still need the **job description** — please upload it."
        ]
    return [
        "Please reply **`jd`** if that file is the job description, or "
        "**`resume`** if it's a resume."
    ]


def _jd_accepted_message(document: ParsedDocument, confidence: float) -> str:
    return (
        f"✅ **Job Description received** — `{document.filename}` "
        f"({document.char_count:,} characters read).\n\n"
        "**Now upload one or more resumes.** You can attach several at once, and "
        "I'll rank them against this JD."
    )


def _wrong_document_message(
    document: ParsedDocument, expecting: str, detected: str = ""
) -> str:
    """Say plainly what the file is not, and what the bot needs instead.

    ``expecting`` is ``"jd"`` or ``"resume"`` - whichever the session is waiting
    for - so the guidance always points at the next useful action rather than
    generically asking for "a resume".
    """
    wanted_label = "job description" if expecting == "jd" else "resume"

    if detected == "resume":
        why = "It looks like a **resume**, not a job description."
    elif detected == "jd":
        why = "It looks like a **job description**, not a resume."
    else:
        missing = (
            "responsibilities, requirements or qualifications"
            if expecting == "jd"
            else "work experience, education, skills, projects or contact details"
        )
        why = (
            "I read the file successfully, but none of the sections a "
            f"{wanted_label} normally has were found ({missing})."
        )

    return (
        f"⚠️ **`{document.filename}` is not a {wanted_label}.**\n"
        f"{why}\n\n"
        f"**Please upload a {wanted_label}** in PDF, DOCX or TXT format."
    )


def _extraction_failure_message(document: ParsedDocument) -> str:
    reason = document.warnings[0] if document.warnings else "The file could not be read."
    return f"⚠️ **`{document.filename}` could not be processed.**\n{reason}"


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
async def handle_analyze(
    session: Session,
    progress: Optional[ProgressCallback] = None,
    client: LLMClient | None = None,
) -> list[str]:
    """Run the analysis and return Discord-ready messages."""
    if not session.has_jd:
        return [
            "⚠️ **No job description yet.**\n"
            "Upload the job description first, then the resumes."
        ]
    if not session.resume_documents:
        return [
            "⚠️ **No resumes uploaded yet.**\n"
            "Upload at least one resume (PDF, DOCX or TXT) and I'll analyse it."
        ]

    # Refuse a second concurrent run: a double /analyze, or an upload landing
    # mid-run, would otherwise duplicate the work and interleave the output.
    if session.is_analyzing:
        return [
            "⏳ **An analysis is already running for you.**\n"
            "I'll post the results here as soon as it finishes."
        ]

    session.is_analyzing = True
    session.state = SessionState.ANALYZING
    session.touch()

    # Snapshot the inputs so files uploaded while this runs cannot change the
    # set being analysed halfway through.
    jd_document = session.jd_document
    resume_documents = list(session.resume_documents)

    try:
        report = await run_analysis(
            jd_document,
            resume_documents,
            client=client,
            progress=progress,
        )
    except Exception:
        logger.exception("Analysis failed")
        session.state = SessionState.RESUMES_RECEIVED
        return [
            "❌ **The analysis failed unexpectedly.**\n"
            "Your uploads are still saved — run `/analyze` to try again, or "
            "`/reset` to start over."
        ]
    finally:
        session.is_analyzing = False

    session.state = SessionState.RESULTS
    session.touch()

    messages = discord_format.format_full_report(report)

    # If the LLM was configured but had to be switched off, say so plainly
    # rather than quietly serving a degraded analysis as if nothing happened.
    if client is not None and getattr(client, "degraded", False):
        messages.append(
            "ℹ️ **Note:** the AI provider was unavailable, so this analysis used "
            "the built-in deterministic engine only. All scores above are still "
            "fully computed — only the written commentary is less detailed.\n"
            f"_Reason: {client.disabled_reason}_\n"
            "_Set `DEMO_MODE=true` in `.env` to silence this and skip the "
            "provider entirely._"
        )

    messages.append(
        "─" * 40 + "\n"
        "✅ **Analysis complete.** Upload more resumes to add them to this "
        "comparison, or run `/reset` to start a new job description."
    )
    return messages


# --------------------------------------------------------------------------- #
# Small talk / unrecognised text
# --------------------------------------------------------------------------- #
_GREETINGS = {
    "hi", "hii", "hiii", "hey", "heyy", "hello", "helo", "hlo", "yo", "hiya",
    "hola", "namaste", "sup", "wassup", "whatsup", "greetings", "good morning",
    "good afternoon", "good evening", "gm", "ge", "hi there", "hey there",
    "hello there", "hi bot", "hello bot",
}
_THANKS = {
    "thanks", "thank you", "thankyou", "thx", "ty", "tysm", "cheers",
    "thanks a lot", "thank u", "great", "nice", "awesome", "perfect", "cool",
}
_FAREWELLS = {"bye", "goodbye", "good bye", "see ya", "cya", "later", "gn", "good night"}
_CAPABILITY = {
    "what can you do", "who are you", "what do you do", "what is this",
    "how do you work", "how does this work", "what are you",
}


def next_step_hint(session: Session) -> str:
    """One line telling the user exactly what to send next."""
    if not session.has_jd:
        return "📋 **Next step:** upload a **job description** (PDF, DOCX or TXT)."
    if session.resume_count == 0:
        return "📄 **Next step:** upload **one or more resumes**."
    return (
        f"✅ I have the JD and {session.resume_count} "
        f"resume{'s' if session.resume_count != 1 else ''} — "
        "run **`/analyze`**, or upload more resumes first."
    )


def handle_chat(session: Session, text: str) -> list[str] | None:
    """Reply to greetings, thanks and unrecognised text.

    Returns ``None`` when the message is not something to respond to, so the
    bot stays quiet rather than replying to every line in a busy channel.
    """
    normalized = " ".join(text.strip().lower().strip("!.?,").split())
    if not normalized:
        return None

    if normalized in _GREETINGS:
        return [
            "👋 **Hi there! I'm ResumeMatch AI.**\n"
            "I compare resumes against a job description and give you an ATS "
            "score, skill match, missing areas, alignment and course "
            "recommendations for each candidate.\n\n"
            f"{next_step_hint(session)}\n"
            "_Type_ `/help` _for everything I can do._"
        ]

    if normalized in _THANKS:
        return [f"You're welcome! 🙌\n{next_step_hint(session)}"]

    if normalized in _FAREWELLS:
        return [
            "👋 Goodbye! Run `/start` whenever you want to analyse more resumes.\n"
            "_Your uploads are cleared automatically after a period of inactivity._"
        ]

    if normalized in _CAPABILITY:
        return [HELP]

    return None


def handle_unrecognised(session: Session, text: str) -> list[str]:
    """Nudge for text the bot could not interpret during an active session."""
    return [
        "🤔 I didn't quite catch that.\n\n"
        f"{next_step_hint(session)}\n"
        "_Commands:_ `/start` `/analyze` `/status` `/reset` `/help`"
    ]
