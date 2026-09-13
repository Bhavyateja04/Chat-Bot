"""Session state machine and the Discord-free conversation handlers."""

from __future__ import annotations

import pytest

from app.bot import handlers
from app.bot.session import Session, SessionStore
from app.models.models import SessionState


@pytest.fixture
def session():
    return Session(key=(1, 2))


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
class TestSessionStateMachine:
    def test_a_new_session_is_idle(self, session):
        assert session.state == SessionState.IDLE
        assert not session.has_jd
        assert session.resume_count == 0
        assert not session.is_ready

    def test_start_moves_to_waiting_for_jd(self, session):
        session.start()
        assert session.state == SessionState.WAITING_FOR_JD

    def test_full_happy_path(self, session, parse_fixture):
        session.start()

        session.set_jd(parse_fixture("software_jd.txt"))
        assert session.state == SessionState.WAITING_FOR_RESUMES
        assert session.has_jd
        assert not session.is_ready

        session.add_resume(parse_fixture("strong_software_resume.txt"))
        assert session.state == SessionState.RESUMES_RECEIVED
        assert session.is_ready

    def test_multiple_resumes_accumulate(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        for name in (
            "strong_software_resume.txt",
            "weak_software_resume.txt",
            "mechanical_resume.txt",
        ):
            session.add_resume(parse_fixture(name))
        assert session.resume_count == 3
        assert session.file_count == 4

    def test_disambiguation_state(self, session, parse_fixture):
        session.await_disambiguation(parse_fixture("software_jd.txt"))
        assert session.state == SessionState.AWAITING_DISAMBIGUATION
        assert session.pending_document is not None

    def test_reset_clears_everything(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))

        session.reset()

        assert session.state == SessionState.IDLE
        assert session.jd_document is None
        assert session.resume_documents == []
        assert session.pending_document is None
        assert not session.is_ready

    def test_expiry(self, session):
        assert not session.is_expired(3600)
        session.last_activity -= 7200
        assert session.is_expired(3600)


# --------------------------------------------------------------------------- #
# Session store isolation
# --------------------------------------------------------------------------- #
class TestSessionStore:
    def test_different_users_get_different_sessions(self, session_store, parse_fixture):
        alice = session_store.get(channel_id=1, user_id=100)
        bob = session_store.get(channel_id=1, user_id=200)

        alice.set_jd(parse_fixture("software_jd.txt"))

        assert alice is not bob
        assert bob.jd_document is None, "Sessions must not leak between users"

    def test_same_user_in_different_channels_is_isolated(self, session_store,
                                                         parse_fixture):
        here = session_store.get(channel_id=1, user_id=100)
        there = session_store.get(channel_id=2, user_id=100)
        here.set_jd(parse_fixture("software_jd.txt"))
        assert there.jd_document is None

    def test_the_same_key_returns_the_same_session(self, session_store):
        assert session_store.get(1, 100) is session_store.get(1, 100)

    def test_clear_removes_a_session(self, session_store, parse_fixture):
        session = session_store.get(1, 100)
        session.set_jd(parse_fixture("software_jd.txt"))
        session_store.clear(1, 100)
        assert session_store.get(1, 100).jd_document is None

    def test_purge_expired(self, session_store):
        session = session_store.get(1, 100)
        session.last_activity -= 10 ** 6
        assert session_store.purge_expired() == 1
        assert len(session_store) == 0


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #
class TestCommandHandlers:
    def test_start_returns_the_welcome_message(self, session):
        messages = handlers.handle_start(session)
        assert "Welcome to ResumeMatch AI" in messages[0]
        assert session.state == SessionState.WAITING_FOR_JD

    def test_help_lists_the_five_outputs(self):
        text = handlers.handle_help()[0]
        for expected in ("ATS Score", "Skill Set Match", "Missing Areas",
                         "Alignment", "Courses"):
            assert expected in text

    def test_reset_confirms_and_clears(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        message = handlers.handle_reset(session)[0]
        assert "cleared" in message.lower()
        assert not session.has_jd

    def test_status_on_an_empty_session(self, session):
        assert "Nothing uploaded" in handlers.handle_status(session)[0]

    def test_status_reports_progress(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        text = handlers.handle_status(session)[0]
        assert "software_jd.txt" in text
        assert "resume" in text.lower()


# --------------------------------------------------------------------------- #
# Upload handling
# --------------------------------------------------------------------------- #
class TestUploadHandling:
    def test_first_upload_is_detected_as_the_jd(self, session, upload):
        messages, ready = handlers.handle_uploads(session, [upload("software_jd.txt")])

        assert session.has_jd
        assert not ready
        assert "Job Description received" in messages[0]

    def test_resume_upload_after_the_jd(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        messages, ready = handlers.handle_uploads(
            session, [upload("strong_software_resume.txt")]
        )

        assert session.resume_count == 1
        assert ready
        assert any("resume" in m.lower() for m in messages)

    def test_multiple_resumes_in_one_batch(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        _, ready = handlers.handle_uploads(
            session,
            [
                upload("strong_software_resume.txt"),
                upload("weak_software_resume.txt"),
                upload("mechanical_resume.txt"),
            ],
        )
        assert session.resume_count == 3
        assert ready

    def test_a_resume_uploaded_first_is_kept_and_the_jd_requested(self, session, upload):
        messages, ready = handlers.handle_uploads(
            session, [upload("strong_software_resume.txt")]
        )

        assert session.resume_count == 1
        assert not session.has_jd
        assert not ready
        assert "job description" in " ".join(messages).lower()

    def test_invalid_file_type_is_reported(self, session):
        messages, ready = handlers.handle_uploads(session, [("photo.png", b"\x89PNG")])
        assert not ready
        assert "unsupported" in messages[0].lower()
        assert session.file_count == 0

    def test_corrupt_pdf_is_reported_not_crashed(self, session):
        messages, _ = handlers.handle_uploads(session, [("broken.pdf", b"garbage")])
        assert "could not be processed" in messages[0].lower()

    def test_empty_upload_list(self, session):
        messages, ready = handlers.handle_uploads(session, [])
        assert messages == []
        assert not ready

    def test_too_many_files_is_refused(self, session, upload, monkeypatch):
        from app.config import get_settings

        monkeypatch.setattr(get_settings(), "max_files_per_session", 2)
        handlers.handle_uploads(session, [upload("software_jd.txt")])

        messages, ready = handlers.handle_uploads(
            session,
            [upload("strong_software_resume.txt"), upload("weak_software_resume.txt")],
        )
        assert not ready
        assert "too many files" in messages[0].lower()

    def test_files_do_not_leak_between_analyses(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        handlers.handle_uploads(session, [upload("strong_software_resume.txt")])

        handlers.handle_reset(session)
        handlers.handle_uploads(session, [upload("mechanical_jd.txt")])

        assert session.resume_count == 0
        assert session.jd_document.filename == "mechanical_jd.txt"


# --------------------------------------------------------------------------- #
# Disambiguation
# --------------------------------------------------------------------------- #
class TestDisambiguation:
    @pytest.fixture
    def ambiguous_session(self, session, parse_fixture):
        session.await_disambiguation(parse_fixture("software_jd.txt"))
        return session

    def test_answering_jd(self, ambiguous_session):
        message = handlers.handle_disambiguation(ambiguous_session, "jd")[0]
        assert ambiguous_session.has_jd
        assert "job description" in message.lower()

    def test_answering_resume(self, ambiguous_session):
        handlers.handle_disambiguation(ambiguous_session, "resume")
        assert ambiguous_session.resume_count == 1

    def test_an_unclear_answer_re_asks(self, ambiguous_session):
        message = handlers.handle_disambiguation(ambiguous_session, "maybe?")[0]
        assert "reply" in message.lower()
        assert ambiguous_session.state == SessionState.AWAITING_DISAMBIGUATION

    def test_nothing_pending_returns_nothing(self, session):
        assert handlers.handle_disambiguation(session, "jd") == []


# --------------------------------------------------------------------------- #
# Pasted JD
# --------------------------------------------------------------------------- #
class TestPastedJd:
    def test_a_pasted_jd_is_accepted(self, session, fixture_text):
        text = fixture_text("software_jd.txt").decode()
        message = handlers.handle_pasted_jd(session, text)[0]
        assert session.has_jd
        assert "received" in message.lower()

    def test_a_too_short_paste_is_rejected(self, session):
        message = handlers.handle_pasted_jd(session, "Need a dev")[0]
        assert not session.has_jd
        assert "too short" in message.lower()


# --------------------------------------------------------------------------- #
# Analysis handler guardrails
# --------------------------------------------------------------------------- #
class TestAnalyzeHandler:
    async def test_analyze_without_a_jd_is_refused(self, session, parse_fixture):
        session.add_resume(parse_fixture("strong_software_resume.txt"))
        messages = await handlers.handle_analyze(session)
        assert "no job description" in messages[0].lower()

    async def test_analyze_without_resumes_is_refused(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        messages = await handlers.handle_analyze(session)
        assert "no resumes" in messages[0].lower()

    async def test_analyze_produces_a_report(self, session, parse_fixture):
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))

        messages = await handlers.handle_analyze(session)

        assert session.state == SessionState.RESULTS
        blob = "\n".join(messages)
        assert "ATS SCORE" in blob
        assert "Analysis complete" in messages[-1]

    async def test_analysis_failure_is_handled_gracefully(
        self, session, parse_fixture, monkeypatch
    ):
        session.set_jd(parse_fixture("software_jd.txt"))
        session.add_resume(parse_fixture("strong_software_resume.txt"))

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(handlers, "run_analysis", boom)
        messages = await handlers.handle_analyze(session)

        assert "failed" in messages[0].lower()
        # Uploads survive so the user can retry.
        assert session.has_jd
        assert session.resume_count == 1


class TestDeferredUploads:
    """Files after an ambiguous one must not be silently dropped."""

    def test_remaining_files_are_reported_when_one_is_ambiguous(
        self, session, upload, monkeypatch
    ):
        from app.models.models import DocumentKind

        monkeypatch.setattr(
            handlers, "classify_document", lambda text, name: (DocumentKind.UNKNOWN, 0.5)
        )
        messages, ready = handlers.handle_uploads(
            session,
            [
                upload("software_jd.txt"),
                upload("strong_software_resume.txt"),
                upload("weak_software_resume.txt"),
            ],
        )

        blob = "\n".join(messages)
        assert "can't tell" in blob.lower()
        assert "strong_software_resume.txt" in blob
        assert "weak_software_resume.txt" in blob
        assert not ready


class TestOnlyResumesAfterJd:
    """Once the JD is set, non-resume uploads must be refused."""

    def test_a_second_job_description_is_refused(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        messages, ready = handlers.handle_uploads(session, [upload("mechanical_jd.txt")])

        assert session.resume_count == 0
        assert not ready
        assert "is not a resume" in messages[0]
        assert "job description" in messages[0]

    def test_an_unrecognisable_document_is_refused(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        junk = ("notes.txt", b"Milk, eggs, bread. Remember to call the plumber. " * 20)
        messages, ready = handlers.handle_uploads(session, [junk])

        assert session.resume_count == 0
        assert not ready
        assert "is not a resume" in messages[0]
        assert "please upload a resume" in messages[0].lower()

    def test_genuine_resumes_are_still_accepted(self, session, upload):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        _, ready = handlers.handle_uploads(
            session,
            [upload("strong_software_resume.txt"), upload("mechanical_resume.txt")],
        )
        assert session.resume_count == 2
        assert ready


class TestNonResumePdfRejected:
    """A readable PDF that simply isn't a resume must be refused."""

    JUNK = {
        "invoice.pdf": "INVOICE #4471 Bill To: Acme Corp Amount Due: 4500.00 "
                       "Thank you for your business. ",
        "article.pdf": "The history of the printing press begins in the fifteenth "
                       "century when Gutenberg developed movable type. ",
        "certificate.pdf": "This is to certify that the bearer attended the "
                           "workshop on modern art appreciation. ",
    }

    def _pdf(self, make_pdf, name):
        return (name, make_pdf([self.JUNK[name]] * 12))

    @pytest.mark.parametrize(
        "name", ["invoice.pdf", "article.pdf", "certificate.pdf"]
    )
    def test_rejected_before_any_jd(self, session, make_pdf, name):
        """With no JD yet, the bot asks for a job description, not a resume."""
        messages, ready = handlers.handle_uploads(session, [self._pdf(make_pdf, name)])

        assert not ready
        assert session.file_count == 0
        assert "is not a job description" in messages[0]
        assert "please upload a job description" in messages[0].lower()
        # It must NOT ask "is this the JD or a resume?"
        assert "can't tell" not in messages[0].lower()

    @pytest.mark.parametrize("name", ["invoice.pdf", "article.pdf"])
    def test_rejected_after_the_jd_is_set(self, session, upload, make_pdf, name):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        messages, ready = handlers.handle_uploads(session, [self._pdf(make_pdf, name)])

        assert not ready
        assert session.resume_count == 0
        assert "is not a resume" in messages[0]

    def test_a_real_resume_pdf_is_still_accepted(
        self, session, upload, make_pdf, resume_lines
    ):
        handlers.handle_uploads(session, [upload("software_jd.txt")])
        messages, ready = handlers.handle_uploads(
            session, [("resume_priya.pdf", make_pdf(resume_lines))]
        )
        assert ready
        assert session.resume_count == 1
        assert "received as a **resume**" in messages[0]


class TestAsksForJdWhenNoneIsSet:
    """With no JD yet, anything that isn't a JD must ask for a JD."""

    def test_a_resume_first_asks_for_the_jd(self, session, upload):
        messages, ready = handlers.handle_uploads(
            session, [upload("strong_software_resume.txt")]
        )

        assert not ready
        assert "is not a job description" in messages[0]
        assert "please upload a job description" in messages[0].lower()
        # The resume is kept so the user need not re-upload it.
        assert session.resume_count == 1
        assert "saved" in messages[0].lower()

    def test_a_resume_pdf_first_asks_for_the_jd(self, session, make_pdf, resume_lines):
        messages, _ = handlers.handle_uploads(
            session, [("resume_priya.pdf", make_pdf(resume_lines))]
        )
        assert "is not a job description" in messages[0]
        assert "It looks like a **resume**" in messages[0]

    def test_junk_pdf_first_asks_for_the_jd(self, session, make_pdf):
        messages, _ = handlers.handle_uploads(
            session, [("invoice.pdf", make_pdf(["INVOICE #4471 Amount Due 4500.00"] * 12))]
        )
        assert "is not a job description" in messages[0]
        assert session.file_count == 0

    def test_the_jd_is_still_accepted_normally(self, session, upload):
        messages, _ = handlers.handle_uploads(session, [upload("software_jd.txt")])
        assert session.has_jd
        assert "Job Description received" in messages[0]

    def test_resume_then_jd_completes_the_session(self, session, upload):
        handlers.handle_uploads(session, [upload("strong_software_resume.txt")])
        handlers.handle_uploads(session, [upload("software_jd.txt")])

        assert session.has_jd
        assert session.resume_count == 1
        assert session.is_ready
