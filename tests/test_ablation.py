"""Ablation model parity: V1 switches == scripts/legacy_v1/model.py, V2 switches == litegpt.model.LiteGPT; ladder parameter matching."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.ablation import LADDER, LITEGPT_PROXY, AblationConfig, AblationGPT, from_legacy_v1, ladder_config  # noqa: E402
from litegpt.config import ModelConfig  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402


def legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_v1_model", ROOT / "scripts" / "legacy_v1" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v1_switches_reproduce_legacy_v1_logits():
    L = legacy_module()
    torch.manual_seed(0)
    legacy = L.GPT(L.GPTConfig(block_size=64, vocab_size=320, n_layer=3, n_head=4, n_embd=64, dropout=0.0, bias=False)).eval()
    abl = from_legacy_v1(legacy).eval()
    idx = torch.randint(0, 320, (2, 40))
    with torch.no_grad():
        ref, _ = legacy(idx, targets=idx)
        got, _ = abl(idx, targets=idx)
    assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def test_v2_switches_load_into_litegpt_and_match():
    torch.manual_seed(0)
    cfg = AblationConfig(n_layer=2, n_embd=64, n_head=4, n_kv_head=2, ffn_hidden=96, vocab_size=300, block_size=64,
                         positional="rope", norm="rmsnorm", qk_norm=True, mlp="swiglu")
    abl = AblationGPT(cfg).eval()
    lite = LiteGPT(ModelConfig(block_size=64, vocab_size=300, n_layer=2, n_embd=64, n_head=4, n_kv_head=2, ffn_hidden=96,
                               attention_backend="sdpa")).eval()
    lite.load_state_dict(abl.state_dict())                    # identical parameter names and shapes
    idx = torch.randint(0, 300, (2, 33))
    with torch.no_grad():
        a, _ = abl(idx, targets=idx)
        b, _ = lite(idx, targets=idx)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


@pytest.mark.parametrize("step", sorted(LADDER))
def test_ladder_steps_are_parameter_matched(step):
    base = AblationGPT(ladder_config("s1")).num_params(non_embedding=True)
    with torch.device("meta"):
        n = AblationGPT(ladder_config(step)).num_params(non_embedding=True)
    assert abs(n / base - 1) <= 0.03, (step, n, base)


def test_deep_thin_litegpt_proxy_is_parameter_matched():
    base = AblationGPT(ladder_config("s1")).num_params(non_embedding=True)
    with torch.device("meta"):
        lite = LiteGPT(ModelConfig(vocab_size=50304, **LITEGPT_PROXY))
    assert abs(lite.num_params(non_embedding=True) / base - 1) <= 0.03


def test_each_ladder_step_changes_one_thing():
    order = ["s1", "s2", "s3", "s4", "s5"]
    changed = []
    for a, b in zip(order, order[1:]):
        ca, cb = ladder_config(a).to_dict(), ladder_config(b).to_dict()
        diff = {k for k in ca if ca[k] != cb[k]} - {"ffn_hidden"}     # MLP width only re-balances parameters
        changed.append(diff)
    assert changed == [{"positional"}, {"norm", "qk_norm"}, {"mlp"}, {"n_kv_head"}]
