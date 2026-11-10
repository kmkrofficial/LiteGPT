"""GGUF export and checks (EXP-M5-GGUF): the G0 expectation table, output parsing, the perplexity chunk rule and the verdict arithmetic."""
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import gguf_check as C  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.export import to_gguf as G  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402

V2_HASH = "eaaa8956eacaed1edd1abc73640b574e5fa5cd3b3d8b98320978a51d8c10ad6c"


def test_patch_maps_the_v2_hash_to_qwen2_only():
    patch = (ROOT / "benchmarks" / "gguf" / "llama_cpp_litegpt.patch").read_text(encoding="utf-8")
    added = [x[1:] for x in patch.splitlines() if x.startswith("+") and not x.startswith("+++")]
    assert len(added) == 3
    assert V2_HASH in added[0] and added[2].strip() == 'res = "qwen2"'


def test_g0_expectations_follow_the_release_kinds(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{- '<|bos|>' -}}", encoding="utf-8")
    base, inst = G.expected("base", tmp_path, "Q8_0"), G.expected("instruct", tmp_path, "F32")
    assert base["tokenizer.ggml.eos_token_id"] == 1 and inst["tokenizer.ggml.eos_token_id"] == 3
    assert base["general.name"] == "LiteGPT-V2-Base" and inst["general.name"] == "LiteGPT-Llama-V2-Instruct"
    assert base["general.file_type"] == 7 and inst["general.file_type"] == 0
    assert base["qwen3.rope.freq_base"] == 10000.0 and base["tokenizer.ggml.pre"] == "qwen2" and base["tokenizer.ggml.add_bos_token"] is True
    assert G.gguf_name("instruct", "Q4_K_M") == "LiteGPT-Llama-V2-Instruct-Q4_K_M.gguf"


def test_check_metadata_reports_every_difference(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("t", encoding="utf-8")
    want = G.expected("base", tmp_path, "BF16")
    assert G.check_metadata(dict(want), want) == []
    bad = {**want, "qwen3.rope.freq_base": 1000000.0, "tokenizer.ggml.add_bos_token": False}
    bad.pop("tokenizer.chat_template")
    problems = G.check_metadata(bad, want)
    assert len(problems) == 3 and any(p.startswith("qwen3.rope.freq_base") for p in problems)
    assert G.check_metadata({**want, "qwen3.rope.freq_base": 10000.0000001}, want) == []


def test_parse_perplexity_plain_and_kl_divergence():
    plain = "perplexity: calculating perplexity over 40 chunks\n[1]19.1,[2]19.4,\nFinal estimate: PPL = 19.3949 +/- 0.81234\n"
    assert C.parse_perplexity(plain) == {"ppl": 19.3949}
    kld = ("====== Perplexity statistics ======\nMean PPL(Q)                   :  19.410054 ±   0.812\nMean PPL(base)                :  19.392275 ±   0.811\n"
           "====== KL divergence statistics ======\nMean    KLD:   0.000448 ±   0.000010\n====== Token probability statistics ======\n"
           "Same top p: 98.729 ± 0.248 %\n")
    out = C.parse_perplexity(kld)
    assert out == {"ppl_q": 19.410054, "ppl_base": 19.392275, "kld_mean": 0.000448, "same_top_pct": 98.729, "ppl": 19.410054}
    with pytest.raises(RuntimeError):
        C.parse_perplexity("you need at least 4096 tokens to evaluate perplexity")


def test_torch_perplexity_follows_the_llama_cpp_chunk_rule():
    torch.manual_seed(0)
    model = LiteGPT(get_config("litegpt-tiny", vocab_size=512, block_size=64, attention_backend="sdpa")).eval()
    ids = torch.randint(5, 512, (3 * 16 + 5,)).tolist()
    out = C.torch_perplexity(model, ids, bos=0, n_ctx=16, chunks=2)
    assert out["chunks"] == 2 and out["scored_tokens"] == 2 * (16 - 1 - 8)
    nll = 0.0
    with torch.no_grad():
        for i in range(2):
            chunk = ids[i * 16:(i + 1) * 16]
            chunk[0] = 0
            logits, _ = model(torch.tensor([chunk]), logits_all=True)
            lp = torch.log_softmax(logits[0].float(), -1)
            nll -= sum(float(lp[p, chunk[p + 1]]) for p in range(8, 15))
    assert out["ppl"] == pytest.approx(math.exp(nll / 14), rel=1e-6)


def test_verdict_applies_the_registered_limits():
    def row(torch_ppl, f32, bf16, q8=None, q4=None):
        r = {"torch": {"ppl": torch_ppl}, "F32": {"ppl": f32}, "BF16": {"ppl": bf16}}
        if q8:
            r["Q8_0"], r["Q4_K_M"] = {"ppl": q8}, {"ppl": q4}
        return r
    ppl = {d: row(10.0, 10.005, 10.02, 10.09, 10.4) if d in C.QUANT_DOMAINS else row(10.0, 10.005, 10.02) for d in C.DOMAINS}
    res = {"perplexity": ppl, "G1": {"passed": True}, "G2_greedy": {"identical": 21, "prompts": 23}}
    v = C.verdict(res, "base")
    assert v == {"G1": True, "G2": True, "G2_ppl": True, "G2_greedy": True, "G3_BF16": True, "G3_Q8_0": True, "G3_Q4_K_M": True}
    ppl["web"]["F32"]["ppl"] = 10.02                       # +0.2%, over the 0.1% line
    ppl["code"]["Q4_K_M"]["ppl"] = 10.6                    # +5.8% against BF16
    res["G2_greedy"]["identical"] = 20                     # 87% < 90%
    v = C.verdict(res, "base")
    assert not v["G2_ppl"] and not v["G2_greedy"] and not v["G2"] and not v["G3_Q4_K_M"] and v["G3_Q8_0"]


def test_edge_strings_cover_the_risky_cases():
    s = C.edge_strings()
    assert 90 <= len(s) <= 110
    assert "café" in s and "<|im_start|>user\nhi<|im_end|>\n" in s and "\r\n" in s and any("🚀" in x for x in s)


def test_gguf_rows_pair_prefill_with_decode_after_the_same_context():
    from benchmarks import cpu_bench as B
    e = lambda t, p, g, d, ts: {"n_threads": t, "n_prompt": p, "n_gen": g, "n_depth": d, "samples_ts": ts}
    entries = [e(4, 128, 0, 0, [1000.0, 1200.0, 1100.0]), e(4, 512, 0, 0, [900.0]), e(4, 0, 64, 128, [200.0, 180.0, 190.0]),
               e(4, 0, 64, 512, [150.0]), e(8, 128, 0, 0, [1500.0])]                     # 8 threads has no decode entry: skipped
    rows = B.gguf_rows(entries)
    assert [(r["threads"], r["prompt_tokens"]) for r in rows] == [(4, 128), (4, 512)]
    r = rows[0]
    assert (r["prefill_tok_s"], r["prefill_tok_s_min"], r["prefill_tok_s_max"]) == (1100.0, 1000.0, 1200.0)
    assert (r["decode_tok_s"], r["decode_tok_s_min"], r["decode_tok_s_max"]) == (190.0, 180.0, 200.0)
    assert r["ttft_s"] == pytest.approx(128 / 1100) and r["ttft_s_min"] == pytest.approx(128 / 1200)


def test_pi_compute_cap_applies_only_when_asked():
    from benchmarks import cpu_bench as B
    model = {"weight_bytes": 215_000_000, "peak_rss_gib": 0.3, "rows": [{"threads": 4, "prompt_tokens": 128, "prefill_tok_s": 4000.0, "decode_tok_s": 240.0}]}
    pi5 = lambda bw, cap: B.estimate_pi(model, bandwidth_4t_gbs=bw, compute_cap=cap)["Raspberry Pi 5"]["prompts"][128]["decode_tok_s"]
    # laptop at 60 GB/s: bandwidth-scaled 55.2 binds before compute (60), so the cap changes nothing
    assert pi5(60.0, False) == pytest.approx(240.0 * 13.8 / 60.0) and pi5(60.0, True) == pytest.approx(240.0 * 13.8 / 60.0)
    # laptop at 40 GB/s: bandwidth-scaled 82.8 is cut to the 64.2 roofline without the cap, and to 240 / 4 = 60 with it
    assert pi5(40.0, False) == pytest.approx(13.8e9 / 215e6) and pi5(40.0, True) == pytest.approx(60.0)
    pi3 = B.estimate_pi(model, bandwidth_4t_gbs=60.0, compute_cap=True)["Raspberry Pi 3 Model B"]["prompts"][128]["decode_tok_s"]
    assert pi3 == pytest.approx(min(240.0 * 2.2 / 60.0, 2.2e9 / 215e6, 240.0 / 16.0))  # 8.8


def test_strip_end_drops_only_a_trailing_end_token():
    assert C.strip_end([5, 6, 3]) == [5, 6] and C.strip_end([5, 6, 1]) == [5, 6]
    assert C.strip_end([3, 5]) == [3, 5] and C.strip_end([]) == []
