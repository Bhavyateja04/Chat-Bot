"""Per-user conversation session state.

Sessions are keyed by (channel, user) so two people in the same Discord channel
never mix their files, and one person can run separate analyses in a DM and a
server channel. Sessions expire so a stale JD is never silently reused.

State machine:

    IDLE -> WAITING_FOR_JD -> JD_RECEIVED -> WAITING_FOR_RESUMES
         -> RESUMES_RECEIVED -> ANALYZING -> RESULTS

``AWAITING_DISAMBIGUATION`` is entered when a first upload cannot confidently be
classified as a JD or a resume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.models.models import ParsedDocument, SessionState


@dataclass
class Session:
    """One user's in-flight analysis."""

    key: tuple[int, int]
    state: SessionState = SessionState.IDLE
    jd_document: Optional[ParsedDocument] = None
    resume_documents: list[ParsedDocument] = field(default_factory=list)
    pending_document: Optional[ParsedDocument] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    # Guards against two analyses running for the same session at once (a
    # double /analyze, or an upload arriving mid-run). asyncio is cooperative,
    # so a plain flag set before the first await is sufficient.
    is_analyzing: bool = False

    def touch(self) -> None:
        self.last_activity = time.time()

    # --- transitions ---------------------------------------------------- #
    def start(self) -> None:
        self.reset()
        self.state = SessionState.WAITING_FOR_JD

    def set_jd(self, document: ParsedDocument) -> None:
        self.jd_document = document
        self.pending_document = None
        self.state = SessionState.WAITING_FOR_RESUMES
        self.touch()

    def add_resume(self, document: ParsedDocument) -> None:
        self.resume_documents.append(document)
        self.pending_document = None
        self.state = SessionState.RESUMES_RECEIVED
        self.touch()

    def await_disambiguation(self, document: ParsedDocument) -> None:
        self.pending_document = document
        self.state = SessionState.AWAITING_DISAMBIGUATION
        self.touch()

    def reset(self) -> None:
        self.state = SessionState.IDLE
        self.jd_document = None
        self.resume_documents = []
        self.pending_document = None
        self.is_analyzing = False
        self.created_at = time.time()
        self.touch()

    # --- queries -------------------------------------------------------- #
    @property
    def has_jd(self) -> bool:
        return self.jd_document is not None

    @property
    def resume_count(self) -> int:
        return len(self.resume_documents)

    @property
    def is_ready(self) -> bool:
        """Enough material to run an analysis."""
        return self.has_jd and bool(self.resume_documents)

    @property
    def file_count(self) -> int:
        return (1 if self.has_jd else 0) + len(self.resume_documents)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.last_activity) > ttl_seconds

    def summary(self) -> str:
        jd_name = self.jd_document.filename if self.jd_document else "not provided"
        resumes = (
            ", ".join(d.filename for d in self.resume_documents)
            if self.resume_documents
            else "none yet"
        )
        return (
            f"**State:** `{self.state.value}`\n"
            f"**Job description:** {jd_name}\n"
            f"**Resumes ({self.resume_count}):** {resumes}"
        )


class SessionStore:
    """In-memory session storage with TTL expiry.

    In-memory is the right call for this MVP: sessions are short lived and a
    restart should not resurrect someone's half-finished upload. Swapping in
    Redis later means replacing this one class.
    """

    # A purge sweep runs after this many get() calls.
    PURGE_EVERY = 50

    def __init__(self, ttl_minutes: int | None = None) -> None:
        settings = get_settings()
        self.ttl_seconds = (ttl_minutes or settings.session_ttl_minutes) * 60
        self._sessions: dict[tuple[int, int], Session] = {}
        self._gets_since_purge = 0

    def get(self, channel_id: int, user_id: int) -> Session:
        """Fetch or create a session, discarding it first if it has expired."""
        # Sweep abandoned sessions periodically. Without this the store grows
        # for every user who ever spoke to the bot and never came back.
        self._gets_since_purge += 1
        if self._gets_since_purge >= self.PURGE_EVERY:
            self._gets_since_purge = 0
            self.purge_expired()

        key = (channel_id, user_id)
        session = self._sessions.get(key)
        if session is not None and session.is_expired(self.ttl_seconds):
            session.reset()
        if session is None:
            session = Session(key=key)
            self._sessions[key] = session
        session.touch()
        return session

    def clear(self, channel_id: int, user_id: int) -> None:
        self._sessions.pop((channel_id, user_id), None)

    def purge_expired(self) -> int:
        """Drop expired sessions. Returns how many were removed."""
        expired = [
            key
            for key, session in self._sessions.items()
            if session.is_expired(self.ttl_seconds) and not session.is_analyzing
        ]
        for key in expired:
            del self._sessions[key]
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)
