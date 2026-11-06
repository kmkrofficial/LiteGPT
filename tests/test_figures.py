"""The committed charts: every chart is listed in the index and used in the docs. `python -m scripts.build_figures --check` compares them with the records."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"


def charts() -> set[str]:
    return {p.name for p in FIGURES.glob("*.svg")}


def test_every_chart_is_in_the_index_and_the_index_has_no_extras():
    listed = set(re.findall(r"\]\(([^)]+\.svg)\)", (FIGURES / "README.md").read_text(encoding="utf-8")))
    assert charts() and listed == charts()


def test_every_chart_is_shown_in_a_document():
    used = set()
    for doc in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        if doc == FIGURES / "README.md":
            continue
        used |= set(re.findall(r"figures/([^)/]+\.svg)\)", doc.read_text(encoding="utf-8")))
    assert charts() - used == set()


def test_charts_carry_no_timestamp():
    for p in FIGURES.glob("*.svg"):
        assert "<dc:date>" not in p.read_text(encoding="utf-8")
