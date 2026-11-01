"""Knowledge-base loader: markdown front matter, PDFs with sidecar metadata, and files that must be skipped instead of crashing indexing."""
from pathlib import Path

from src.rag.loader import load_documents, read_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_a_markdown_article_is_split_into_metadata_and_body(tmp_path):
    (tmp_path / "etf-basics.md").write_text("---\ntitle: ETF Basics\ncategory: etfs\nlast_updated: 2026-10-30\n---\n# ETF Basics\nBody text.\n")
    [doc] = load_documents(tmp_path)
    assert doc.id == "etf-basics" and doc.text.startswith("# ETF Basics") and "title:" not in doc.text
    assert doc.metadata == {"title": "ETF Basics", "category": "etfs", "last_updated": "2026-10-30"}   # the YAML date became a plain string
    assert all(isinstance(v, str) for v in doc.metadata.values())


def test_malformed_front_matter_is_reported_not_raised(tmp_path):
    path = tmp_path / "oops.md"
    path.write_text("---\ntitle: Risk: what it is\n---\nbody\n")   # an unquoted colon makes the YAML invalid
    metadata, text, problems = read_document(path)
    assert text == "" and metadata == {} and "front matter" in problems[0]
    assert load_documents(tmp_path) == []


def test_a_pdf_with_a_sidecar_gives_text_and_our_metadata():
    metadata, text, problems = read_document(FIXTURES / "kb_pdf" / "sample-text.pdf")
    assert text and problems == [] and metadata.get("category")


def test_a_pdf_without_a_sidecar_and_a_scanned_pdf_are_flagged():
    _, _, no_sidecar = read_document(FIXTURES / "kb_pdf" / "sample-nometa.pdf")
    _, text, scanned = read_document(FIXTURES / "kb_pdf" / "sample-scanned.pdf")
    assert any("sidecar" in p for p in no_sidecar) and text == "" and any("no extractable text" in p for p in scanned)


def test_load_documents_skips_problem_files_and_other_extensions(tmp_path):
    (tmp_path / "good.md").write_text("---\ntitle: Good\ncategory: basics\n---\nText.\n")
    (tmp_path / "empty.md").write_text("---\ntitle: Empty\n---\n")
    (tmp_path / "notes.txt").write_text("not an article")
    assert [d.id for d in load_documents(tmp_path)] == ["good"]
    assert [d.id for d in load_documents(FIXTURES / "kb_pdf")] == ["sample-text"]   # the scanned and no-sidecar PDFs are left out


def test_documents_come_back_in_a_stable_order(tmp_path):
    for name in ("b", "a", "c"):
        (tmp_path / f"{name}.md").write_text(f"---\ntitle: {name}\n---\ntext {name}\n")
    assert [d.id for d in load_documents(tmp_path)] == ["a", "b", "c"]
