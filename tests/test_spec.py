import pytest

from litegpt.config import get_config
from litegpt.spec import load_spec, spec_sha256, spec_version, validate_model_config


def test_spec_loads_and_hash_is_stable():
    spec = load_spec()
    assert spec["spec_version"].startswith("2.0.0-spec.")
    h = spec_sha256()
    assert len(h) == 64 and h == spec_sha256()
    assert spec_version().startswith("2.0.0-spec.")


def test_extension_gate_is_preregistered_and_consistent():
    g = load_spec()["training"]["extension_gate"]
    assert g["threshold_bpb"] == 0.010 and g["frozen_before"] == "M3" and g["metric"] == "val_bpb"
    assert g["bpb_window_a"] == [6.5e9, 6.75e9, 7.0e9] and g["bpb_window_b"] == [8.5e9, 8.75e9, 9.0e9]
    p = load_spec()["training"]["pretraining"]
    assert p["stable_phase_end_tokens"] == 9.0e9 and p["budget_initial_tokens"] == 10.0e9 and p["budget_max_tokens"] == 15.0e9
    assert max(g["bpb_window_b"]) <= p["stable_phase_end_tokens"]


def test_v2_28l_config_conforms_to_spec():
    assert validate_model_config(get_config("litegpt-v2-28l")) == []


def test_deviation_is_reported():
    problems = validate_model_config(get_config("litegpt-v2-24l"))
    assert any(p.startswith("n_layer") for p in problems)


def test_rope_theta_stays_at_the_spec_value():
    assert validate_model_config(get_config("litegpt-v2-28l", rope_theta=10000.0)) == []
    assert validate_model_config(get_config("litegpt-v2-28l", rope_theta=100000.0)) != []      # ADR-009: no theta switch
    assert validate_model_config(get_config("litegpt-v2-28l", rope_theta=12345.0)) != []


def test_param_counts_match_spec():
    import torch
    from litegpt.model import LiteGPT
    spec = load_spec()["model"]
    with torch.device("meta"):
        m = LiteGPT(get_config("litegpt-v2-28l"))
    assert m.num_params() == pytest.approx(spec["params_total_expected"], rel=1e-3)
    assert m.num_params(non_embedding=True) == pytest.approx(spec["params_non_embedding_expected"], rel=1e-3)


def test_no_numeric_value_silently_loaded_as_string():
    """PyYAML parses `9.0e9` as a string; the spec must use `9.0e+9`. Guard every scalar in the spec."""
    import re
    pat = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")
    bad = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, str) and pat.match(x.strip()):
            bad.append(path)

    walk(load_spec())
    assert not bad, f"numeric-looking strings in spec (use signed exponent): {bad}"
