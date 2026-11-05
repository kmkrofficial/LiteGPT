"""Export to the Hugging Face Qwen3 layout, on a tiny model: the same checks EXP-M5-EXPORT runs on the real weights."""
import json
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import api  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.export import to_hf  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402

LENGTHS = (1, 16, 200)


@pytest.fixture(scope="module", params=["base", "instruct"])
def exported(tmp_path_factory, request):
    d = tmp_path_factory.mktemp(request.param)
    torch.manual_seed(0)
    cfg = get_config("litegpt-tiny", vocab_size=32768, block_size=256, attention_backend="sdpa")
    ck = d / "m.pt"
    torch.save({"format": 1, "model": LiteGPT(cfg).state_dict(), "model_config": cfg.to_dict(), "step": 7, "tokens": 70, "spec_sha256": None}, ck)
    out = d / "out"
    return ck, out, to_hf.export(ck, out, request.param), request.param


def test_the_checks_pass_on_a_tiny_model(exported):
    ck, out, _, _ = exported
    res = to_hf.check(ck, out, lengths=LENGTHS, greedy_lengths=(16, 200), new_tokens=12, strings=20)
    assert res["passed"], res
    assert res["logits_max"] <= 1e-4     # far inside the registered 1e-3: only floating-point noise should remain
    assert res["at_bfloat16"]["16"]["greedy_tokens"] == 12


def test_tied_embeddings_are_stored_once_at_bf16(exported):
    from safetensors import safe_open
    _, out, meta, _ = exported
    with safe_open(out / "model.safetensors", "pt") as f:
        keys = set(f.keys())
        assert "lm_head.weight" not in keys and "model.embed_tokens.weight" in keys
        assert f.get_tensor("model.embed_tokens.weight").dtype == torch.bfloat16
    assert meta["weights_sha256"] == to_hf.sha256(out / "model.safetensors")


def test_the_package_is_complete_and_its_card_has_no_draft_or_relative_links(exported):
    _, out, meta, kind = exported
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "README.md", "NOTICE", "LICENSE", "litegpt.json", "licenses/QWEN.txt"):
        assert (out / name).is_file(), name
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: apache-2.0")
    assert "Draft for owner" not in card and "](../" not in card and "draft model card" not in card
    notice = (out / "NOTICE").read_text(encoding="utf-8")
    assert "draft" not in notice and "Built with Qwen" in notice and "Built with Qwen" in card
    if kind == "instruct":
        assert meta["name"] == "LiteGPT-Llama-V2-Instruct"
        assert "Built with Llama" in notice and "Built with Llama" in card and "Llama 3.1 is licensed under" in notice
        assert (out / "licenses" / "LLAMA-3.1.txt").is_file()
        assert json.loads((out / "config.json").read_text(encoding="utf-8"))["eos_token_id"][0] == to_hf.IM_END
    else:
        assert meta["name"] == "LiteGPT-V2-Base"
        assert not (out / "licenses" / "LLAMA-3.1.txt").exists()


def test_litegpt_loads_the_exported_folder(exported):
    _, out, meta, _ = exported
    m = api.load(out, device="cpu", dtype=torch.float32)
    assert m.step == 7 and m.tokens == 70 and m.layout is None
    assert api.generate(m, "Hello", max_new_tokens=3) is not None
    assert api.info(m)["parameters"] == meta["parameters"]
