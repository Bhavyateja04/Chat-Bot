"""Document parsing: PDF, DOCX, TXT, and every malformed-input path."""

from __future__ import annotations

import pytest

from app.models.models import DocumentKind
from app.parsers import loader
from app.parsers.classifier import classify_document
from app.parsers.docx_parser import extract_docx
from app.parsers.pdf_parser import extract_pdf
from app.parsers.text_parser import extract_text, from_string


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
class TestPdfExtraction:
    def test_extracts_text_from_a_real_pdf(self, make_pdf, resume_lines):
        data = make_pdf(resume_lines)
        document = extract_pdf(data, "resume.pdf")

        assert document.extraction_ok
        assert document.page_count >= 1
        assert "Priya Raghavan" in document.text
        assert "Python" in document.text
        assert document.char_count == len(document.text)

    def test_pdf_survives_the_full_loader(self, make_pdf, resume_lines):
        document = loader.parse_document(make_pdf(resume_lines), "resume.pdf")
        assert document.extraction_ok
        assert document.is_usable
        assert document.file_type == "pdf"

    def test_corrupt_pdf_reports_an_error_instead_of_raising(self):
        document = extract_pdf(b"this is definitely not a pdf", "broken.pdf")
        assert not document.extraction_ok
        assert document.text == ""
        assert document.warnings
        assert "corrupt" in document.warnings[0].lower()

    def test_empty_pdf_bytes(self):
        document = extract_pdf(b"", "empty.pdf")
        assert not document.extraction_ok
        assert "empty" in document.warnings[0].lower()

    def test_image_only_pdf_gives_a_helpful_message(self, image_only_pdf):
        """A scanned PDF has pages but no text layer - say so, don't score it."""
        document = extract_pdf(image_only_pdf(), "scanned.pdf")

        assert not document.extraction_ok
        assert document.page_count == 1
        assert any(
            "scanned" in w.lower() or "no readable text" in w.lower()
            for w in document.warnings
        )
        # The message must tell the user what to do instead.
        assert any("docx" in w.lower() for w in document.warnings)

    def test_pdf_with_zero_pages(self, make_pdf):
        document = extract_pdf(make_pdf([]), "nopages.pdf")
        assert not document.extraction_ok
        assert document.page_count == 0


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
class TestDocxExtraction:
    def test_extracts_text_from_a_real_docx(self, make_docx, resume_lines):
        document = extract_docx(make_docx(resume_lines), "resume.docx")

        assert document.extraction_ok
        assert "Priya Raghavan" in document.text
        assert "PostgreSQL" in document.text

    def test_docx_survives_the_full_loader(self, make_docx, resume_lines):
        document = loader.parse_document(make_docx(resume_lines), "resume.docx")
        assert document.extraction_ok
        assert document.file_type == "docx"

    def test_corrupt_docx_reports_an_error(self):
        document = extract_docx(b"not a docx at all", "broken.docx")
        assert not document.extraction_ok
        assert document.warnings

    def test_empty_docx_bytes(self):
        document = extract_docx(b"", "empty.docx")
        assert not document.extraction_ok

    def test_docx_with_no_paragraph_text(self, make_docx):
        document = extract_docx(make_docx([]), "blank.docx")
        assert not document.extraction_ok
        assert "no readable text" in document.warnings[0].lower()


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
class TestTextExtraction:
    def test_utf8_text(self):
        document = extract_text("Résumé of Ann Lee\nPython, SQL".encode("utf-8"), "cv.txt")
        assert document.extraction_ok
        assert "Résumé" in document.text

    def test_cp1252_fallback(self):
        document = extract_text("Café Engineer".encode("cp1252"), "cv.txt")
        assert document.extraction_ok
        assert "Caf" in document.text

    def test_empty_text_file(self):
        document = extract_text(b"", "empty.txt")
        assert not document.extraction_ok

    def test_whitespace_only_file(self):
        document = extract_text(b"   \n\n  \t ", "blank.txt")
        assert not document.extraction_ok

    def test_from_string_for_pasted_content(self):
        document = from_string("Job description text here", "pasted")
        assert document.extraction_ok
        assert document.file_type == "text"


# --------------------------------------------------------------------------- #
# Loader-level validation
# --------------------------------------------------------------------------- #
class TestLoaderValidation:
    @pytest.mark.parametrize("filename", ["resume.pdf", "cv.docx", "jd.txt", "notes.md"])
    def test_supported_extensions(self, filename):
        assert loader.is_supported(filename)

    @pytest.mark.parametrize(
        "filename", ["photo.png", "archive.zip", "resume.doc", "script.exe", "noext"]
    )
    def test_unsupported_extensions(self, filename):
        assert not loader.is_supported(filename)

    def test_invalid_file_type_is_rejected_with_guidance(self):
        document = loader.parse_document(b"\x89PNG\r\n", "resume.png")
        assert not document.extraction_ok
        assert "unsupported" in document.warnings[0].lower()

    def test_legacy_doc_gets_a_specific_hint(self):
        message = loader.unsupported_message("old_resume.doc")
        assert ".docx" in message

    def test_oversized_file_is_rejected(self, monkeypatch):
        from app.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "max_file_size_mb", 0.001)

        document = loader.parse_document(b"x" * 50_000, "huge.txt")
        assert not document.extraction_ok
        assert "limit" in document.warnings[0].lower()

    def test_very_long_document_is_truncated(self, monkeypatch):
        from app.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "max_document_chars", 500)

        document = loader.parse_document(b"word " * 5000, "long.txt")
        assert document.extraction_ok
        assert len(document.text) <= 500
        assert any("truncated" in w.lower() for w in document.warnings)

    def test_loader_never_raises_on_garbage(self):
        for payload in (b"", b"\x00\x01\x02", b"%PDF-broken"):
            document = loader.parse_document(payload, "thing.pdf")
            assert document is not None  # returns a result rather than raising


# --------------------------------------------------------------------------- #
# JD vs resume classification
# --------------------------------------------------------------------------- #
class TestDocumentClassification:
    def test_identifies_a_job_description(self, fixture_text):
        text = fixture_text("software_jd.txt").decode()
        kind, confidence = classify_document(text, "software_jd.txt")
        assert kind == DocumentKind.JOB_DESCRIPTION
        assert confidence >= 0.6

    def test_identifies_a_resume(self, fixture_text):
        text = fixture_text("strong_software_resume.txt").decode()
        kind, confidence = classify_document(text, "strong_software_resume.txt")
        assert kind == DocumentKind.RESUME
        assert confidence >= 0.6

    def test_identifies_a_mechanical_resume(self, fixture_text):
        text = fixture_text("mechanical_resume.txt").decode()
        kind, _ = classify_document(text, "mechanical_resume.txt")
        assert kind == DocumentKind.RESUME

    def test_classifies_jd_by_content_even_with_a_neutral_filename(self, fixture_text):
        text = fixture_text("mechanical_jd.txt").decode()
        kind, _ = classify_document(text, "document1.pdf")
        assert kind == DocumentKind.JOB_DESCRIPTION

    def test_empty_document_is_unknown(self):
        kind, confidence = classify_document("", "mystery.pdf")
        assert kind == DocumentKind.UNKNOWN
        assert confidence == 0.0

    def test_filename_hint_does_not_fire_inside_a_word(self):
        """'jd' must not match inside a name like 'jdoe'."""
        text = "Experience\nEducation\nSkills\nProjects\njohn@example.com"
        kind, _ = classify_document(text, "jdoe_cv.pdf")
        assert kind == DocumentKind.RESUME
