"""Single entry point for turning uploaded bytes into text.

Routes on file extension, validates size, and produces a ParsedDocument with
useful warnings rather than raising on bad input.
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings
from app.models.models import ParsedDocument
from app.parsers.docx_parser import extract_docx
from app.parsers.pdf_parser import extract_pdf
from app.parsers.text_parser import extract_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".text"}


def is_supported(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTENSIONS


def unsupported_message(filename: str) -> str:
    extension = os.path.splitext(filename)[1] or "(no extension)"
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    hint = ""
    if extension.lower() == ".doc":
        hint = " Legacy `.doc` files are not supported - please save as `.docx` or PDF."
    elif extension.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        hint = " Images have no text layer - please upload a PDF, DOCX or TXT file."
    return (
        f"⚠️ `{filename}` has an unsupported file type ({extension}).\n"
        f"Supported formats: {supported}.{hint}"
    )


def parse_document(data: bytes, filename: str) -> ParsedDocument:
    """Extract text from an uploaded file.

    Always returns a ParsedDocument. Content problems are reported through
    ``extraction_ok`` and ``warnings`` so the bot can give a helpful reply.
    """
    settings = get_settings()
    extension = os.path.splitext(filename.lower())[1]

    if len(data) > settings.max_file_size_bytes:
        size_mb = len(data) / (1024 * 1024)
        return ParsedDocument(
            filename=filename,
            text="",
            file_type=extension.lstrip("."),
            extraction_ok=False,
            warnings=[
                f"The file is {size_mb:.1f} MB, over the "
                f"{settings.max_file_size_mb:.0f} MB limit. Please upload a smaller file."
            ],
        )

    if extension not in SUPPORTED_EXTENSIONS:
        return ParsedDocument(
            filename=filename,
            text="",
            file_type=extension.lstrip("."),
            extraction_ok=False,
            warnings=[unsupported_message(filename)],
        )

    try:
        if extension == ".pdf":
            document = extract_pdf(data, filename)
        elif extension == ".docx":
            document = extract_docx(data, filename)
        else:
            document = extract_text(data, filename)
    except Exception as exc:  # last-resort guard - the bot must never crash
        logger.exception("Unexpected parse failure for %s", filename)
        return ParsedDocument(
            filename=filename,
            text="",
            file_type=extension.lstrip("."),
            extraction_ok=False,
            warnings=[f"An unexpected error occurred while reading the file: {exc}"],
        )

    # Truncate very long documents so prompts stay small and fast.
    if len(document.text) > settings.max_document_chars:
        document.warnings.append(
            f"The document was truncated to {settings.max_document_chars:,} characters "
            "for analysis."
        )
        document.text = document.text[: settings.max_document_chars]
        document.char_count = len(document.text)

    if document.extraction_ok and not document.is_usable:
        document.warnings.append(
            "Only a very small amount of text was extracted, so the analysis "
            "confidence will be low."
        )

    return document
