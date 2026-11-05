"""Documentation checks: tagged Python snippets run, and relative links resolve.

A fenced ```python block whose first line is `# doc-test` is executed (it must be self-contained and fast, e.g. on
`litegpt-tiny`). Other blocks are illustrative and not executed.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
FENCE = re.compile(r"```python\n(.*?)```", re.S)
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def tagged_snippets():
    for path in DOCS:
        for m in FENCE.finditer(path.read_text(encoding="utf-8")):
            code = m.group(1)
            if code.startswith("# doc-test"):
                yield pytest.param(code, id=f"{path.relative_to(ROOT).as_posix()}:{m.start()}")


SNIPPETS = list(tagged_snippets())


def test_at_least_one_tagged_snippet_exists():
    assert SNIPPETS, "no `# doc-test` snippets found in docs/"


@pytest.mark.parametrize("code", SNIPPETS)
def test_snippet_runs(code):
    exec(compile(code, "<doc-snippet>", "exec"), {"__name__": "__doc_snippet__"})


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_relative_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    missing = []
    for target in LINK.findall(text):
        if re.match(r"^[a-z]+:", target) or target.startswith("/"):
            continue                                         # URLs and absolute paths are not checked
        if not (path.parent / target).exists():
            missing.append(target)
    assert not missing, f"broken relative links in {path.name}: {missing}"
