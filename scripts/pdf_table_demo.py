"""Demo: why tables are hard to extract from PDFs.

Builds the SAME table (made-up funds, illustrative numbers) as two PDFs that differ only
in the order the cells are drawn, then extracts text from each. A PDF is a list of
"draw this text at (x, y)" instructions, so the extractor has to guess the structure.

Run:  .venv/bin/python -m scripts.pdf_table_demo
"""
from pathlib import Path

from pypdf import PdfReader

ROWS = [
    ("Fund", "Expense ratio", "5-yr return"),   # made-up data, not real funds
    ("Alpha Index Fund", "0.05%", "9.1%"),
    ("Beta Growth Fund", "0.85%", "10.4%"),
    ("Gamma Bond Fund", "0.20%", "3.2%"),
]
COL_X = [72, 260, 400]
ROW_Y = [700, 680, 660, 640]


def build_pdf(path: Path, order: str) -> None:
    cells = [(r, c) for r in range(len(ROWS)) for c in range(3)]
    if order == "column-by-column":
        cells = sorted(cells, key=lambda rc: (rc[1], rc[0]))
    stream = "BT /F1 11 Tf\n"
    for r, c in cells:
        stream += f"1 0 0 1 {COL_X[c]} {ROW_Y[r]} Tm ({ROWS[r][c]}) Tj\n"
    stream += "ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


for order in ("row-by-row", "column-by-column"):
    path = Path(f"tests/fixtures/pdf_tables/table-{order}.pdf")
    build_pdf(path, order)
    page = PdfReader(path).pages[0]
    for mode in ("plain", "layout"):
        print(f"\n===== drawn {order} | extraction_mode={mode} =====")
        print(page.extract_text(extraction_mode=mode))
