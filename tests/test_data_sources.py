"""Guard tests for docs/spec/data_sources.yaml against docs/spec/v2.yaml (ADR-008)."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt.spec import load_spec  # noqa: E402

SOURCES = Path(__file__).resolve().parents[1] / "docs" / "spec" / "data_sources.yaml"


@pytest.fixture(scope="module")
def ds():
    return yaml.safe_load(SOURCES.read_text(encoding="utf-8"))


def test_versions_agree(ds):
    assert ds["spec_version"] == load_spec()["spec_version"]
    assert load_spec()["data"]["sources_file"] == "docs/spec/data_sources.yaml"


def test_mixture_weights_sum_to_one(ds):
    for phase in ("stable", "decay"):
        assert abs(sum(ds["mixture"][phase].values()) - 1.0) < 2e-6, phase


def test_sources_mixture_matches_spec(ds):
    spec = load_spec()["data"]["mixture"]
    for phase in ("stable", "decay"):
        flat = {}
        for k, v in ds["mixture"][phase].items():
            key = "code" if k.startswith("code_") else k
            flat[key] = flat.get(key, 0.0) + v
        for key, w in spec[phase].items():
            assert flat[key] == pytest.approx(w, abs=2e-6), (phase, key)
        assert set(flat) == set(spec[phase])


def test_code_language_weights_sum_to_one_and_match_spec(ds):
    langs = ds["sources"]["code"]["languages"]
    assert sum(v["weight"] for v in langs.values()) == pytest.approx(1.0)
    assert {k: v["weight"] for k, v in langs.items()} == load_spec()["data"]["mixture"]["code_languages"]
    code = {k: v for k, v in ds["mixture"]["stable"].items() if k.startswith("code_")}
    total = sum(code.values())
    for lang, v in langs.items():
        assert code[f"code_{lang.lower()}"] / total == pytest.approx(v["weight"], abs=1e-4)


def test_wikipedia_is_excluded(ds):
    assert "wikipedia" not in ds["sources"]
    assert "wikipedia" not in load_spec()["data"]["mixture"]["stable"]
    assert "wiki" not in load_spec()["data"]["validation"]["val_domains"]


def test_revisions_are_full_commit_shas(ds):
    for name, src in ds["sources"].items():
        assert len(src["revision"]) == 40 and all(c in "0123456789abcdef" for c in src["revision"]), name
    for name, ref in ds["decontamination"]["reference"].items():
        assert len(ref["revision"]) == 40, name


def test_decontamination_reference_covers_planned_suite(ds):
    needed = {"hellaswag", "arc_easy", "arc_challenge", "piqa", "openbookqa", "commonsenseqa", "mmlu", "gsm8k", "winogrande",
              "humaneval", "mbpp", "ifeval", "truthfulqa", "triviaqa", "xstest", "multipl_e_humaneval_js",
              "multipl_e_humaneval_ts", "text_to_sql_spider"}
    assert needed <= set(ds["decontamination"]["reference"])
    assert load_spec()["data"]["decontamination"]["pretraining"] == "filter"


def test_partition_fractions_are_sane(ds):
    for name, src in ds["sources"].items():
        p = src.get("partition") or {}
        if "val_fraction" in p:
            assert 0 < p["val_fraction"] + p["decay_fraction"] < 1, name
