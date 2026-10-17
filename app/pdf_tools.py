"""Local PDF creation, reading, and copy-on-edit operations."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
LIBRARY_DIR = ROOT / "data" / "library"
MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_PDF_PAGES = 200


def _safe_stem(value: object, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "", str(value or "")).strip(" .")
    text = re.sub(r"\s+", "-", text)[:70]
    return text or fallback


def _new_path(stem: str) -> Path:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    return LIBRARY_DIR / f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.pdf"


def _styles() -> dict:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("CCPDFTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=28, textColor=colors.HexColor("#20352f"), alignment=TA_LEFT, spaceAfter=18),
        "body": ParagraphStyle("CCPDFBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=16, textColor=colors.HexColor("#34423c"), spaceAfter=9),
        "note": ParagraphStyle("CCPDFNote", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=15, textColor=colors.HexColor("#34423c"), spaceAfter=8),
    }


def _paragraphs(text: str, style: ParagraphStyle) -> list:
    result = []
    for block in re.split(r"\n\s*\n", text.strip()):
        if block.strip():
            result.append(Paragraph("<br/>".join(escape(line) for line in block.splitlines()), style))
            result.append(Spacer(1, 4))
    return result


def create_pdf(title: object, text: object) -> dict:
    heading = str(title or "Untitled document").strip()[:180] or "Untitled document"
    body = str(text or "").strip()
    if not body or len(body) > 100_000:
        raise ValueError("Add document text up to 100,000 characters.")
    path = _new_path(_safe_stem(heading, "document"))
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, rightMargin=.8 * inch, leftMargin=.8 * inch, topMargin=.75 * inch, bottomMargin=.75 * inch, title=heading, author="AI Command Center")
    doc.build([Paragraph(escape(heading), _styles()["title"]), *_paragraphs(body, _styles()["body"])])
    return {"filename": path.name, "url": f"/api/library/view/{path.name}", "size": path.stat().st_size, "pages": len(PdfReader(str(path)).pages)}


def _source_pdf(filename: object) -> Path:
    name = Path(str(filename or "")).name
    path = (LIBRARY_DIR / name).resolve()
    if path.parent != LIBRARY_DIR.resolve() or path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("Choose a PDF already saved in the Library.")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError("This PDF is over the 30 MB size limit.")
    return path


def read_pdf(filename: object) -> dict:
    path = _source_pdf(filename)
    reader = PdfReader(str(path), strict=False)
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError("This PDF has more than 200 pages; split it before reading.")
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(f"--- Page {index + 1} ---\n{content}" for index, content in enumerate(pages))
    return {"filename": path.name, "pages": len(reader.pages), "text": text[:60_000], "truncated": len(text) > 60_000}


def edit_pdf(filename: object, note: object = "", title: object = "") -> dict:
    """Create a revised copy by appending a note page and/or updating metadata."""
    source = _source_pdf(filename)
    note_text = str(note or "").strip()
    new_title = str(title or "").strip()[:180]
    if not note_text and not new_title:
        raise ValueError("Add a note page or a new document title to make an edit.")
    if len(note_text) > 20_000:
        raise ValueError("Keep the note under 20,000 characters.")
    reader = PdfReader(str(source), strict=False)
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError("This PDF has more than 200 pages; split it before editing.")
    if note_text and len(reader.pages) >= MAX_PDF_PAGES:
        raise ValueError("This PDF already has 200 pages; remove a page before adding a note page.")
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    if note_text:
        note_path = LIBRARY_DIR / f".note-{uuid.uuid4().hex}.pdf"
        try:
            doc = SimpleDocTemplate(str(note_path), pagesize=LETTER, rightMargin=.8 * inch, leftMargin=.8 * inch, topMargin=.75 * inch, bottomMargin=.75 * inch)
            doc.build([Paragraph("Added note", _styles()["title"]), *_paragraphs(note_text, _styles()["note"])])
            for page in PdfReader(str(note_path)).pages:
                writer.add_page(page)
        finally:
            note_path.unlink(missing_ok=True)
    metadata = dict(reader.metadata or {})
    if new_title:
        metadata["/Title"] = new_title
    writer.add_metadata(metadata)
    path = _new_path(f"{_safe_stem(source.stem, 'document')}-edited")
    with path.open("wb") as output:
        writer.write(output)
    return {"filename": path.name, "url": f"/api/library/view/{path.name}", "size": path.stat().st_size, "pages": len(writer.pages), "source": source.name}
