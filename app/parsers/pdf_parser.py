"""PDF text extraction using PyMuPDF (fitz).

Handles the failure modes that actually occur with real resumes: corrupt files,
password-protected files, empty documents and image-only (scanned) PDFs.
"""

from __future__ import annotations

import logging

from app.models.models import ParsedDocument

logger = logging.getLogger(__name__)

# Below this many characters per page we assume the PDF is a scan with no text
# layer rather than a genuinely sparse document.
_SCANNED_CHARS_PER_PAGE = 40


class DocumentParseError(Exception):
    """Raised when a document cannot be read at all."""


def extract_pdf(data: bytes, filename: str = "document.pdf") -> ParsedDocument:
    """Extract text from PDF bytes.

    Never raises for content problems - returns a ParsedDocument with
    ``extraction_ok=False`` and an explanatory warning instead.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise DocumentParseError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if not data:
        return ParsedDocument(
            filename=filename,
            text="",
            file_type="pdf",
            extraction_ok=False,
            warnings=["The uploaded PDF is empty (0 bytes)."],
        )

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", filename, exc)
        return ParsedDocument(
            filename=filename,
            text="",
            file_type="pdf",
            extraction_ok=False,
            warnings=[f"The PDF could not be opened - it may be corrupt. ({exc})"],
        )

    warnings: list[str] = []
    try:
        if getattr(document, "needs_pass", False):
            return ParsedDocument(
                filename=filename,
                text="",
                file_type="pdf",
                extraction_ok=False,
                warnings=["The PDF is password protected, so no text could be read."],
            )

        page_count = document.page_count
        if page_count == 0:
            return ParsedDocument(
                filename=filename,
                text="",
                file_type="pdf",
                page_count=0,
                extraction_ok=False,
                warnings=["The PDF contains no pages."],
            )

        chunks: list[str] = []
        for index in range(page_count):
            try:
                page = document.load_page(index)
                chunks.append(page.get_text("text") or "")
            except Exception as exc:  # a single bad page should not kill the file
                logger.warning("Page %s of %s failed: %s", index, filename, exc)
                warnings.append(f"Page {index + 1} could not be read and was skipped.")

        text = _clean("\n".join(chunks))
    finally:
        try:
            document.close()
        except Exception:  # pragma: no cover
            pass

    stripped = text.strip()
    if not stripped:
        return ParsedDocument(
            filename=filename,
            text="",
            file_type="pdf",
            page_count=page_count,
            extraction_ok=False,
            warnings=warnings
            + [
                "No readable text was found. This is usually a scanned or "
                "image-only PDF. Please upload a text-based PDF or a DOCX file."
            ],
        )

    if page_count and len(stripped) / page_count < _SCANNED_CHARS_PER_PAGE:
        warnings.append(
            "Very little text was extracted - the PDF may be partly image based, "
            "so the analysis confidence is reduced."
        )

    return ParsedDocument(
        filename=filename,
        text=stripped,
        char_count=len(stripped),
        page_count=page_count,
        file_type="pdf",
        extraction_ok=True,
        warnings=warnings,
    )


def _clean(text: str) -> str:
    """Normalise whitespace while preserving line structure."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            out.append(line)
        else:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
    return "\n".join(out).strip()
