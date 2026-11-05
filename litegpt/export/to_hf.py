"""Export a V2 checkpoint to the Hugging Face Qwen3 layout and check the result against LiteGPT's own code (EXP-M5-EXPORT).

    python -m litegpt.export.to_hf --checkpoint milestone-final.pt --kind base --out LiteGPT-V2-Base

The folder holds safetensors at bfloat16, the dtype every benchmark was scored at, with tied embeddings stored once; config, tokenizer and
chat template; `litegpt.json` metadata; and the model card, notice and licence files. Parameter names already match, so weights are copied.
Needs the `hf` dependency group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from litegpt import api, chat_template
from litegpt.inference import Engine, GenerationConfig
from litegpt.spec import REPO_ROOT, spec_version
from litegpt.tokenizer import BOS, DEFAULT_PATH, EOT, IM_END, IM_START, LiteTokenizer, PAD

RELEASE = REPO_ROOT / "docs" / "release"
KINDS = {
    "base": {"name": "LiteGPT-V2-Base", "card": "model_card_base.md", "notice": "NOTICE.txt", "licenses": ["QWEN.txt"], "eos": "<|endoftext|>", "eos_ids": [EOT]},
    "instruct": {"name": "LiteGPT-Llama-V2-Instruct", "card": "model_card_instruct.md", "notice": "NOTICE_instruct.txt", "licenses": ["QWEN.txt", "LLAMA-3.1.txt"], "eos": "<|im_end|>", "eos_ids": [IM_END, EOT]},
}
HF_OWNER = "kmkrworks"
GITHUB = "https://github.com/kmkrofficial/LiteGPT/blob/main/docs/"
GITHUB_RAW = "https://raw.githubusercontent.com/kmkrofficial/LiteGPT/main/docs/"
LOGIT_TOLERANCE = 1e-3
PROMPT_LENGTHS = (1, 16, 512, 2048, 3968)
GREEDY_LENGTHS = (16, 512, 3968)
GREEDY_NEW_TOKENS = 64
TOKENIZER_STRINGS = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def hf_config(cfg, kind: dict) -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": cfg.vocab_size, "hidden_size": cfg.n_embd, "intermediate_size": cfg.ffn_hidden,
        "num_hidden_layers": cfg.n_layer, "num_attention_heads": cfg.n_head, "num_key_value_heads": cfg.n_kv_head, "head_dim": cfg.head_dim, "hidden_act": "silu",
        "max_position_embeddings": cfg.block_size, "initializer_range": cfg.init_std, "rms_norm_eps": cfg.norm_eps, "use_cache": True, "tie_word_embeddings": cfg.tie_embeddings,
        "rope_parameters": {"rope_type": "default", "rope_theta": cfg.rope_theta}, "attention_bias": False, "attention_dropout": 0.0, "use_sliding_window": False,
        "layer_types": ["full_attention"] * cfg.n_layer, "bos_token_id": BOS, "eos_token_id": kind["eos_ids"], "pad_token_id": PAD, "dtype": "bfloat16",
    }


def write_tokenizer(out: Path, kind: dict, context: int) -> None:
    """The frozen tokenizer, plus a post-processor so that plain `tokenizer(text)` starts with <|bos|> as V2 was trained. Chat templates already write it."""
    from tokenizers import Tokenizer, processors
    tok = Tokenizer.from_file(str(DEFAULT_PATH))
    tok.post_processor = processors.TemplateProcessing(single="<|bos|> $A", pair="<|bos|> $A $B", special_tokens=[("<|bos|>", BOS)])
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": kind["eos"], "pad_token": "<|pad|>", "model_max_length": context,
                                                           "clean_up_tokenization_spaces": False}, indent=1), encoding="utf-8")
    shutil.copy(chat_template.JINJA_PATH, out / "chat_template.jinja")


def card_text(kind: dict) -> str:
    """The model card for the Hub: front matter added, the draft line removed, links made absolute."""
    s = (RELEASE / kind["card"]).read_text(encoding="utf-8").replace("\r\n", "\n")
    s = re.sub(r"^> Draft[^\n]*\n\n", "", s, flags=re.M).replace(" (draft model card, not released)", "")
    s = re.sub(r"\]\(\.\./figures/([^)]+)\)", lambda m: f"]({GITHUB_RAW}figures/{m.group(1)})", s)
    s = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"]({GITHUB}{m.group(1)})", s)
    s = re.sub(r"\]\(NOTICE[^)]*\)", "](NOTICE)", s)
    s = re.sub(r"\]\(model_card_(base|instruct)\.md\)", lambda m: f"](https://huggingface.co/{HF_OWNER}/{KINDS[m.group(1)]['name']})", s)
    front = ["---", "license: apache-2.0", "library_name: transformers", "pipeline_tag: text-generation", "language:", "- en", "tags:", "- litegpt", "- qwen3"]
    if kind["name"] == KINDS["instruct"]["name"]:
        front += ["- llama", f"base_model: {HF_OWNER}/{KINDS['base']['name']}"]
    return "\n".join(front + ["---", "", s])


def export(checkpoint, out, kind_name: str) -> dict:
    """Write the folder. Returns the metadata written to litegpt.json."""
    from safetensors.torch import save_file
    kind = KINDS[kind_name]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = api.load(checkpoint, device="cpu", dtype=torch.bfloat16)
    cfg = m.model.config
    state = {k: v.clone().contiguous() for k, v in m.model.state_dict().items() if k != "lm_head.weight" or not cfg.tie_embeddings}
    save_file(state, out / "model.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(hf_config(cfg, kind), indent=1), encoding="utf-8")
    (out / "generation_config.json").write_text(json.dumps({"bos_token_id": BOS, "eos_token_id": kind["eos_ids"], "pad_token_id": PAD, "do_sample": False}, indent=1), encoding="utf-8")
    write_tokenizer(out, kind, cfg.block_size)
    (out / "README.md").write_text(card_text(kind), encoding="utf-8", newline="\n")
    notice = (RELEASE / kind["notice"]).read_text(encoding="utf-8").replace(" (draft notice, not released)", "")
    (out / "NOTICE").write_text(notice, encoding="utf-8", newline="\n")
    shutil.copy(RELEASE / "licenses" / "Apache-2.0.txt", out / "LICENSE")
    (out / "licenses").mkdir(exist_ok=True)
    for name in kind["licenses"]:
        shutil.copy(RELEASE / "licenses" / name, out / "licenses" / name)
    meta = {"format": 1, "name": kind["name"], "kind": kind_name, "architecture": "Qwen3ForCausalLM layout", "dtype": "bfloat16", "parameters": sum(p.numel() for p in m.model.parameters()), "model_config": cfg.to_dict(),
            "step": m.step, "tokens": m.tokens, "chat_template": chat_template.VERSION, "spec_version": spec_version(), "source_sha256": sha256(Path(checkpoint)), "weights_sha256": sha256(out / "model.safetensors"),
            "tokenizer_sha256": sha256(DEFAULT_PATH), "licence": "apache-2.0"}
    (out / "litegpt.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def _prompts(vocab: int, lengths) -> dict[int, list[int]]:
    rng = np.random.default_rng(20261105)
    return {n: [BOS] + rng.integers(IM_START + 40, vocab, size=n - 1).tolist() for n in lengths}


def _doc_strings(limit: int) -> list[str]:
    out = []
    for p in sorted((REPO_ROOT / "docs").rglob("*.md")):
        if "private" in p.parts:
            continue
        out += [x for x in p.read_text(encoding="utf-8").splitlines() if len(x) >= 40]
        if len(out) >= limit:
            break
    return out[:limit]


@torch.no_grad()
def check(checkpoint, out, lengths=PROMPT_LENGTHS, greedy_lengths=GREEDY_LENGTHS, new_tokens: int = GREEDY_NEW_TOKENS, strings: int = TOKENIZER_STRINGS) -> dict:
    """The registered checks. Both sides use the bfloat16 weights, computed in float32 on CPU, so bf16 rounding can't hide a difference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path(out)
    src = api.load(checkpoint, device="cpu", dtype=torch.bfloat16)
    src.model = src.model.float().eval()
    src.model.set_context(src.model.config.block_size, src.model.config.rope_theta)      # .to(bfloat16) also rounds the RoPE tables; only weights should be
    hf = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False).eval()
    res: dict = {"parameters": {"litegpt": sum(p.numel() for p in src.model.parameters()), "transformers": hf.num_parameters()}, "logits": {}, "greedy": {}}
    prompts = _prompts(src.model.config.vocab_size, sorted({*lengths, *greedy_lengths}))
    worst = 0.0
    for n in lengths:
        ids = torch.tensor([prompts[n]])
        a = src.model(ids, logits_all=True)[0]
        b = hf(input_ids=ids, use_cache=False).logits
        d = float((a - b).abs().max())
        res["logits"][str(n)] = d
        worst = max(worst, d)
    res["logits_max"] = worst
    res["logits_pass"] = worst <= LOGIT_TOLERANCE
    for n in greedy_lengths:
        ids = prompts[n]
        cut = min(new_tokens, src.model.config.block_size - n)
        a = Engine(src.model, src.tok).generate_ids([ids], GenerationConfig(max_new_tokens=cut, stop_token_ids=()))[0].ids
        b = hf.generate(torch.tensor([ids]), attention_mask=torch.ones(1, n, dtype=torch.long), max_new_tokens=cut, min_new_tokens=cut, do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=PAD)[0, n:].tolist()
        res["greedy"][str(n)] = {"new_tokens": cut, "identical": a == b, "first_difference": next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)}
    res["greedy_pass"] = all(g["identical"] for g in res["greedy"].values())
    res["parameters_pass"] = res["parameters"]["litegpt"] == res["parameters"]["transformers"]

    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=False)
    lite = LiteTokenizer.load()
    texts = _doc_strings(strings)
    plain = sum(tok(t, add_special_tokens=False)["input_ids"] == lite.encode_text(t) for t in texts)
    bos = sum(tok(t)["input_ids"] == [BOS] + lite.encode_text(t) for t in texts)
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name three colours."}]
    want = chat_template.render_ids(lite, msgs, add_generation_prompt=True)
    got = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    res["tokenizer"] = {"strings": len(texts), "identical": plain, "identical_with_bos": bos, "chat_template_identical": list(got) == want}
    res["tokenizer_pass"] = plain == len(texts) == bos and list(got) == want

    round_trip = api.load(out, device="cpu", dtype=torch.float32)
    ids = torch.tensor([prompts[lengths[min(2, len(lengths) - 1)]]])
    res["round_trip_pass"] = bool(torch.equal(round_trip.model(ids, logits_all=True)[0], src.model(ids, logits_all=True)[0]))
    res["passed"] = all(res[k] for k in ("parameters_pass", "logits_pass", "greedy_pass", "tokenizer_pass", "round_trip_pass"))
    res["at_bfloat16"] = _bfloat16_difference(checkpoint, out, prompts, greedy_lengths, new_tokens)
    return res


@torch.no_grad()
def _bfloat16_difference(checkpoint, out, prompts: dict, lengths, new_tokens: int) -> dict:
    """Reported, no pass line. How people will really run it: LiteGPT rounds its RoPE tables to bfloat16 here and Transformers doesn't, so the two can differ a little."""
    from transformers import AutoModelForCausalLM
    a = api.load(checkpoint, device="cpu", dtype=torch.bfloat16).model
    b = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False).eval()
    res = {}
    for n in lengths:
        ids = prompts[n]
        t = torch.tensor([ids])
        cut = min(new_tokens, a.config.block_size - n)
        x = Engine(a).generate_ids([ids], GenerationConfig(max_new_tokens=cut, stop_token_ids=()))[0].ids
        y = b.generate(t, attention_mask=torch.ones(1, n, dtype=torch.long), max_new_tokens=cut, min_new_tokens=cut, do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=PAD)[0, n:].tolist()
        res[str(n)] = {"logits_max_difference": float((a(t, logits_all=True)[0].float() - b(input_ids=t, use_cache=False).logits.float()).abs().max()), "greedy_tokens": cut, "greedy_identical": x == y,
                       "greedy_matching_tokens": sum(p == q for p, q in zip(x, y))}
    return res


def main(argv=None) -> int:
    from litegpt.records import create_run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = create_run("export", KINDS[a.kind]["name"], config={"checkpoint": a.checkpoint, "kind": a.kind, "experiment": "EXP-M5-EXPORT"})
    meta = export(a.checkpoint, a.out, a.kind)
    res = check(a.checkpoint, a.out)
    meta["checks"] = res
    (Path(a.out) / "litegpt.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rec.write_json("export.json", meta)
    print(json.dumps({k: res[k] for k in ("passed", "parameters", "logits", "logits_max", "greedy", "tokenizer", "round_trip_pass", "at_bfloat16")}, indent=1))
    print("record:", rec.path.name)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
