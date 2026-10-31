"""Load knowledge-base files (.md and .pdf) into Document objects.

Used by the validator now, and by the chunker and indexer next.
"""
from dataclasses import dataclass
from pathlib import Path

import frontmatter
import yaml
from pypdf import PdfReader


@dataclass
class Document:
    id: str          # file name without extension, e.g. "stocks-basics"
    text: str        # the article body (front matter / sidecar removed)
    metadata: dict   # title, category, difficulty, source_name, source_url, last_updated


def read_document(path: Path) -> tuple[dict, str, list[str]]:
    """Return (metadata, text, problems found while reading) for a .md or .pdf file.

    .md  : metadata is the front matter at the top of the file.
    .pdf : PDFs have built-in metadata (Title, Author, Subject, Keywords, dates) but not
           our custom fields (category, difficulty, source_url), and in practice those
           built-ins are often blank or wrong. So ours live in a sidecar file with the
           same name: guide.pdf + guide.meta.yaml. The text is extracted with pypdf.
    """
    if path.suffix == ".md":
        post = frontmatter.load(path)
        return post.metadata, post.content, []

    problems = []
    sidecar = path.with_suffix(".meta.yaml")
    metadata = {}
    if sidecar.exists():
        metadata = yaml.safe_load(sidecar.read_text()) or {}
    else:
        problems.append(f"missing sidecar metadata file: {sidecar.name}")
    try:
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    except Exception as error:  # corrupt or encrypted PDF
        return metadata, "", problems + [f"could not read PDF: {type(error).__name__}: {error}"]
    if not text:
        problems.append("no extractable text (scanned/image-only PDF? needs OCR)")
    return metadata, text, problems


def load_documents(directory: str | Path = "knowledge_base") -> list[Document]:
    """Load every .md/.pdf file that has text. Problem files are skipped (the validator reports them)."""
    docs = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix not in (".md", ".pdf"):
            continue
        metadata, text, problems = read_document(path)
        if text and not problems:
            # YAML parses dates into date objects; store plain strings so metadata is JSON/Chroma friendly
            docs.append(Document(path.stem, text, {key: str(value) for key, value in metadata.items()}))
    return docs
