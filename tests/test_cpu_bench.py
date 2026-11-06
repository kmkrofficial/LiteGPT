"""CPU benchmark (EXP-M5-CPU): measurement shape on a tiny model, and the Pi estimate arithmetic."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import cpu_bench as C  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402


def test_measure_reports_medians_with_spread(tmp_path, monkeypatch):
    torch.manual_seed(0)
    ck = tmp_path / "m.pt"
    cfg = get_config("litegpt-tiny", vocab_size=32768, block_size=256, attention_backend="sdpa")
    torch.save({"format": 1, "model": LiteGPT(cfg).state_dict(), "model_config": cfg.to_dict(), "step": 1, "tokens": 1, "spec_sha256": None}, ck)
    monkeypatch.setitem(C.MODELS, "tiny", {"kind": "litegpt", "path": str(ck), "label": "tiny"})
    prefill, generate, encode, weight_bytes, params = C.load("tiny", torch.float32)
    assert weight_bytes == params * 4
    rows = C.measure(prefill, generate, encode, threads=(1,), prompts=(16,), new_tokens=3, repeats=2)
    assert len(rows) == 1
    r = rows[0]
    for k in ("prefill_tok_s", "decode_tok_s", "ttft_s"):
        assert r[k + "_min"] <= r[k] <= r[k + "_max"] and r[k] > 0


def test_pi_estimate_scales_by_bandwidth_and_is_capped_by_the_roofline():
    model = {"weight_bytes": 800_000_000, "peak_rss_gib": 1.5,
             "rows": [{"threads": 4, "prompt_tokens": 128, "prefill_tok_s": 4000.0, "decode_tok_s": 40.0}]}
    est = C.estimate_pi(model, bandwidth_4t_gbs=27.6)
    pi5 = est["Raspberry Pi 5"]["prompts"][128]
    assert pi5["decode_ceiling_tok_s"] == pytest.approx(13.8e9 / 8e8)       # 17.25
    assert pi5["decode_tok_s"] == pytest.approx(17.25)                     # scaled 40 x 13.8 / 27.6 = 20, capped at the ceiling
    assert pi5["prefill_tok_s"] == pytest.approx(1000.0)
    assert pi5["prefill_tok_s_range"] == pytest.approx([500.0, 1500.0])
    assert est["Raspberry Pi 5"]["fits_in_ram_gb"] == {4: True, 8: True}
    assert est["Raspberry Pi 3 Model B"]["fits_in_ram_gb"] == {1: False}


def test_pi_decode_scales_by_bandwidth_below_the_ceiling():
    model = {"weight_bytes": 800_000_000, "peak_rss_gib": 1.0, "rows": [{"threads": 4, "prompt_tokens": 128, "prefill_tok_s": 100.0, "decode_tok_s": 20.0}]}
    pi5 = C.estimate_pi(model, bandwidth_4t_gbs=27.6)["Raspberry Pi 5"]["prompts"][128]
    assert pi5["decode_tok_s"] == pytest.approx(10.0)                      # 20 x 13.8 / 27.6, under the 17.25 ceiling
