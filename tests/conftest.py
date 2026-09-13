"""Shared pytest fixtures.

Every test runs with ``DEMO_MODE=true`` so the whole suite is deterministic and
never touches a real LLM API. Tests that exercise the LLM path mock the HTTP
layer explicitly.
"""

from __future__ import annotations

import io
import os
import pathlib
import sys

import pytest

# Make the project importable when pytest is run from anywhere.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before app.config is first imported.
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("DISCORD_BOT_TOKEN", "test-token-not-real")
os.environ.setdefault("LLM_API_KEY", "")

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _demo_settings():
    """Force demo mode for the whole session."""
    from app.config import reload_settings

    os.environ["DEMO_MODE"] = "true"
    os.environ["LLM_API_KEY"] = ""
    settings = reload_settings()
    assert not settings.llm_configured, "Tests must never call a real LLM."
    return settings


@pytest.fixture
def fixture_text():
    """Read a fixture file's raw bytes by name."""

    def _read(name: str) -> bytes:
        return (FIXTURES / name).read_bytes()

    return _read


@pytest.fixture
def parse_fixture(fixture_text):
    """Parse a text fixture into a ParsedDocument."""
    from app.parsers.text_parser import extract_text

    def _parse(name: str):
        return extract_text(fixture_text(name), name)

    return _parse


@pytest.fixture
def software_jd(parse_fixture):
    from app.analysis.heuristic_extractor import extract_jd

    return extract_jd(parse_fixture("software_jd.txt"))


@pytest.fixture
def mechanical_jd(parse_fixture):
    from app.analysis.heuristic_extractor import extract_jd

    return extract_jd(parse_fixture("mechanical_jd.txt"))


@pytest.fixture
def ml_jd(parse_fixture):
    from app.analysis.heuristic_extractor import extract_jd

    return extract_jd(parse_fixture("ml_engineer_jd.txt"))


@pytest.fixture
def backend_jd(parse_fixture):
    from app.analysis.heuristic_extractor import extract_jd

    return extract_jd(parse_fixture("backend_jd.txt"))


@pytest.fixture
def strong_resume(parse_fixture):
    from app.analysis.heuristic_extractor import extract_resume

    return extract_resume(parse_fixture("strong_software_resume.txt"))


@pytest.fixture
def weak_resume(parse_fixture):
    from app.analysis.heuristic_extractor import extract_resume

    return extract_resume(parse_fixture("weak_software_resume.txt"))


@pytest.fixture
def mechanical_resume(parse_fixture):
    from app.analysis.heuristic_extractor import extract_resume

    return extract_resume(parse_fixture("mechanical_resume.txt"))


@pytest.fixture
def fullstack_resume(parse_fixture):
    from app.analysis.heuristic_extractor import extract_resume

    return extract_resume(parse_fixture("fullstack_resume.txt"))


# --------------------------------------------------------------------------- #
# Binary document fixtures, generated at test time rather than committed
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_pdf():
    """Build a real, text-based PDF from lines of text."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    def _make(lines: list[str]) -> bytes:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=LETTER)
        y = 750
        for line in lines:
            pdf.drawString(60, y, line[:95])
            y -= 16
            if y < 60:
                pdf.showPage()
                y = 750
        pdf.save()
        return buffer.getvalue()

    return _make


@pytest.fixture
def image_only_pdf():
    """A PDF with a real page but no text layer, mimicking a scanned resume."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    def _make() -> bytes:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=LETTER)
        # Vector shapes only - nothing a text extractor can read.
        pdf.rect(100, 400, 300, 200, fill=1)
        pdf.circle(300, 300, 50, fill=1)
        pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    return _make


@pytest.fixture
def make_docx():
    """Build a real DOCX from lines of text."""
    import docx

    def _make(lines: list[str]) -> bytes:
        document = docx.Document()
        for line in lines:
            document.add_paragraph(line)
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()

    return _make


@pytest.fixture
def resume_lines(fixture_text):
    """The strong software resume as a list of lines, for PDF/DOCX generation."""
    return fixture_text("strong_software_resume.txt").decode("utf-8").splitlines()


@pytest.fixture
def session_store():
    from app.bot.session import SessionStore

    return SessionStore()


@pytest.fixture
def upload(fixture_text):
    """Build an ``(filename, bytes)`` upload tuple from a fixture name."""

    def _upload(name: str) -> tuple[str, bytes]:
        return (name, fixture_text(name))

    return _upload
