"""Text extraction helpers for Smart-Sort.

Kept in a separate, dependency-light module so worker processes used for
parallel PDF/DOCX extraction do not pay the numpy/scikit-learn import cost.
"""

from __future__ import annotations

import codecs
import os
from pathlib import Path

# Files parsed by a real parser rather than a plain read; these are CPU-bound
# and worth dispatching to worker processes.
HEAVY_SUFFIXES = frozenset({".pdf", ".docx", ".docm"})

# Files with these suffixes are always attempted as text. Unknown suffixes are
# still attempted when the bytes look printable, so extensionless files
# (README, Makefile, Dockerfile) are picked up too.
TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".text", ".md", ".markdown", ".rst", ".org", ".log", ".csv",
        ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".conf", ".xml", ".html", ".htm", ".css", ".scss", ".js", ".jsx",
        ".ts", ".tsx", ".py", ".pyi", ".rb", ".go", ".rs", ".java", ".kt",
        ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".sh", ".ps1", ".bat",
        ".sql", ".tex", ".rtf", ".srt", ".vtt",
    }
)


class ExtractionError(Exception):
    """Raised when a file's text cannot be extracted."""


def is_heavy(path: Path) -> bool:
    return path.suffix.lower() in HEAVY_SUFFIXES


def _decode_bytes(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF32_LE) or data.startswith(codecs.BOM_UTF32_BE):
        return data.decode("utf-32")
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:8192]
    if sample.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        encoding = "utf-32"
    elif sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    elif b"\x00" in sample:
        return False
    else:
        encoding = "utf-8-sig"
    try:
        text = codecs.getincrementaldecoder(encoding)(errors="strict").decode(sample, final=False)
    except UnicodeDecodeError:
        if encoding != "utf-8-sig":
            return False
        try:
            text = sample.decode("cp1252")
        except UnicodeDecodeError:
            text = sample.decode("latin-1")
    if not text:
        return True
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    return printable / len(text) >= 0.9


def _read_text_file(path: Path) -> str:
    with path.open("rb") as handle:
        sample = handle.read(8192)
        if not looks_like_text(sample):
            raise ExtractionError("binary or unsupported format")
        data = sample + handle.read()
    return _decode_bytes(data)


def _extract_pdf_pdfium(path: Path, max_pages: int) -> str:
    """Fast path: PDFium (pypdfium2) is a C++ engine."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        page_count = min(len(document), max_pages)
        parts: list[str] = []
        for index in range(page_count):
            try:
                page = document[index]
            except Exception:
                continue
            try:
                textpage = page.get_textpage()
            except Exception:
                page.close()
                continue
            try:
                parts.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        return "\n".join(parts)
    finally:
        document.close()


def _extract_pdf_legacy(path: Path, max_pages: int) -> str:
    """Fallback: pure-Python pypdf/PyPDF2."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - depends on environment
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except ImportError as exc:
            raise ExtractionError("install pypdfium2 or pypdf to read PDFs") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"read error ({type(exc).__name__})") from exc

    pages: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def _extract_pdf(path: Path, max_pages: int) -> str:
    if os.environ.get("SMART_SORT_PDF_ENGINE", "").strip().lower() == "legacy":
        return _extract_pdf_legacy(path, max_pages)

    try:
        import pypdfium2  # noqa: F401
    except ImportError:  # pragma: no cover - depends on environment
        return _extract_pdf_legacy(path, max_pages)

    try:
        return _extract_pdf_pdfium(path, max_pages)
    except Exception as exc:
        raise ExtractionError(f"read error ({type(exc).__name__})") from exc


def _extract_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ExtractionError("install python-docx to read .docx files") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"read error ({type(exc).__name__})") from exc

    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text(path: Path, max_pdf_pages: int = 20) -> str:
    """Extract plain text from a file, raising ExtractionError when impossible."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path, max_pdf_pages)
    if suffix in {".docx", ".docm"}:
        return _extract_docx(path)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return _read_text_file(path)

    # Unknown extension: accept it only when the content is clearly text.
    with path.open("rb") as handle:
        data = handle.read(8192)
    if looks_like_text(data):
        return _read_text_file(path)
    raise ExtractionError("binary or unsupported format")


def extract_worker(job: tuple[str, int]) -> tuple[str | None, str | None]:
    """Process-pool entry point: return (text, error) instead of raising."""
    path_str, max_pdf_pages = job
    try:
        return extract_text(Path(path_str), max_pdf_pages), None
    except ExtractionError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"read error ({exc.strerror or type(exc).__name__})"
    except Exception as exc:  # defensive: one bad file must not kill the pool
        return None, f"read error ({type(exc).__name__})"
