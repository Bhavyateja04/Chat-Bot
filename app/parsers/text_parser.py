"""Plain-text extraction (.txt, .md, and pasted chat text)."""

from __future__ import annotations

from app.models.models import ParsedDocument

_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")


def extract_text(data: bytes, filename: str = "document.txt") -> ParsedDocument:
    """Decode text bytes, trying the encodings that actually show up in practice."""
    if not data:
        return ParsedDocument(
            filename=filename,
            text="",
            file_type="txt",
            extraction_ok=False,
            warnings=["The uploaded text file is empty."],
        )

    text = ""
    warnings: list[str] = []
    for encoding in _ENCODINGS:
        try:
            text = data.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = data.decode("utf-8", errors="replace")
        warnings.append("Some characters could not be decoded and were replaced.")

    text = text.strip()
    if not text:
        return ParsedDocument(
            filename=filename,
            text="",
            file_type="txt",
            extraction_ok=False,
            warnings=["The text file contains no readable content."],
        )

    return ParsedDocument(
        filename=filename,
        text=text,
        char_count=len(text),
        file_type="txt",
        extraction_ok=True,
        warnings=warnings,
    )


def from_string(text: str, filename: str = "pasted-text") -> ParsedDocument:
    """Wrap text pasted directly into the chat as a ParsedDocument."""
    cleaned = (text or "").strip()
    return ParsedDocument(
        filename=filename,
        text=cleaned,
        char_count=len(cleaned),
        file_type="text",
        extraction_ok=bool(cleaned),
        warnings=[] if cleaned else ["No text was provided."],
    )
