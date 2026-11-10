"""EXP-M5-GGUF gates G1 to G4: the GGUF files against V2's bf16 weights computed in float32 by PyTorch.

    python -m benchmarks.gguf_check --kind base --hf-dir ~/m5_export/LiteGPT-V2-Base --gguf-dir ~/m5_gguf --llama-cpp ~/llama.cpp \
        --val-dir runs/20261011T173054Z-data-prepare-1f005bc/shards/val_text

G1 tokenizer: llama.cpp's tokenizer (through llama-server /tokenize) against AutoTokenizer on the export's tokenizer.json.
G2 fidelity: F32 GGUF perplexity under llama-perplexity's chunk rule, replicated in PyTorch, and greedy 64-token continuations.
G3 quantization: BF16 against PyTorch, then Q8_0 and Q4_K_M against BF16 with llama-perplexity --kl-divergence.
G4 chat (instruct only): llama.cpp's applied template, BOS count, stop token and greedy replies against PyTorch.
Thresholds are the registered ones (registry EXP-M5-GGUF). One record per run: eval-gguf-check-<name>.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

N_CTX = 2048
CHUNKS = 40                     # per domain: 40 chunks of 2,048 tokens, 40,920 tokens scored
DOMAINS = ("chat", "code", "math", "synthetic", "web")
QUANT_DOMAINS = ("code", "math", "web")
THREADS = 16
DOCS_PER_DOMAIN = 200
GREEDY_NEW = 64
REAL_PROMPTS_PER_DOMAIN = 4
REAL_PROMPT_TOKENS = 128
LIMITS = {"g2_ppl": 0.001, "g2_greedy": 0.90, "bf16_ppl": 0.005, "q8_ppl": 0.01, "q4_ppl": 0.05, "g4_greedy": 9}

CONVERSATIONS = [
    [{"role": "user", "content": "Give me three tips for staying focused while studying."}],
    [{"role": "system", "content": "You are a concise assistant."}, {"role": "user", "content": "What is the capital of Japan?"}],
    [{"role": "user", "content": "Write a Python function that returns the square of a number."}],
    [{"role": "user", "content": "Summarise in one sentence: The meeting moved from Monday to Wednesday because the room was booked."}],
    [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! How can I help you today?"}, {"role": "user", "content": "Tell me a fun fact about octopuses."}],
    [{"role": "system", "content": "Answer with valid JSON only."}, {"role": "user", "content": "Give the name and age of a fictional person as JSON."}],
    [{"role": "user", "content": "Translate 'good morning' into French."}],
    [{"role": "user", "content": "List the planets of the solar system in order."}],
    [{"role": "user", "content": "Explain what a hash table is to a beginner."}],
    [{"role": "user", "content": "What is 12 multiplied by 7?"}, {"role": "assistant", "content": "84."}, {"role": "user", "content": "And divided by 4?"}],
]


def edge_strings() -> list[str]:
    """About 100 strings that stress the split regex, byte fallback, normalisation and special tokens."""
    return ["", " ", "  ", "\t", "\n", "\r\n", "\n\n\n", " \n \n", "a\r\nb", "x \t y", "   leading", "trailing   ", "tab\tseparated\tvalues",
            "0", "7", "42", "123456789", "3.14159", "1,000,000", "2026-11-10", "v2.0.0", "0x1F", "1e-6", "10%", "$5.99", "#42",
            "I'm", "I'M", "you're", "YOU'RE", "it's", "IT'S", "we've", "they'll", "she'd", "'s", "'S", "rock'n'roll", "don't", "DON'T",
            "café", "cafe\u0301", "naïve", "nai\u0308ve", "Ångström", "A\u030angstro\u0308m", "ﬁ ligature", "Ⅻ", "²³", "①②",
            "日本語のテキスト", "我想在apple工作", "한국어 문장", "Русский текст", "Ελληνικά", "العربية نص", "עברית", "हिन्दी वाक्य", "ภาษาไทย",
            "🚀", "😶\u200d🌫️", "👩\u200d👩\u200d👧\u200d👦", "🇮🇳🇯🇵", "✅ done", "🦙🦙🦙",
            "def f(x):\n    return x * 2\n", "if (a && b) { return; }", "SELECT * FROM t WHERE id = 1;", "<div class=\"x\">hi</div>",
            "https://example.com/a?b=c&d=e", "user@example.com", "C:\\Users\\path\\file.txt", "~/.config/app.json", "a-b_c.d/e",
            "------======", "......!!!!!!??????", "```python\nprint(1)\n```", "\"\"\"docstring\"\"\"", "''''''", "(((nested)))",
            "<|bos|>", "<|endoftext|>", "<|im_start|>user\nhi<|im_end|>\n", "<|reserved_3|>", "<|im_", "im_end|>", "text<|im_end|>more",
            "<|pad|>", "<| bos |>", "a" * 300, " " * 50 + "x", "\n" * 20, "ab" * 200, "1" * 100, "😀" * 30,
            "Mixed CASE and 123 numbers, with punctuation!", "The quick brown fox jumps over the lazy dog.", "e\u0301e\u0301e\u0301",
            "\u00a0non-breaking\u00a0space", "zero\u200bwidth", "soft\u00adhyphen", "line\u2028separator", "BOM\ufeffinside",
            "Tab\u3000ideographic space", "\u2126 ohm vs \u03a9 omega", "\u212a kelvin"]


# ---- llama.cpp processes ----

class Server:
    """A llama-server on one GGUF file, for /tokenize, /completion, /apply-template and /v1/chat/completions."""

    def __init__(self, llama_cpp: Path, gguf: Path, port: int, threads: int = THREADS):
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen([str(llama_cpp / "b" / "llama-server"), "-m", str(gguf), "--port", str(port), "-t", str(threads), "-c", "4096",
                                      "-np", "1", "--jinja", "--no-ui"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        for _ in range(240):
            try:
                if json.loads(urllib.request.urlopen(self.url + "/health", timeout=2).read()).get("status") == "ok":
                    return
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited: " + (self.proc.stderr.read() or "")[-2000:])
            time.sleep(0.5)
        raise RuntimeError("llama-server did not become ready")

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=600).read())

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_perplexity(llama_cpp: Path, gguf: Path, text_file: Path, extra: list[str] | None = None) -> dict:
    cmd = [str(llama_cpp / "b" / "llama-perplexity"), "-m", str(gguf), "-f", str(text_file), "-c", str(N_CTX), "-b", str(N_CTX),
           "--chunks", str(CHUNKS), "-t", str(THREADS), *(extra or [])]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed ({p.returncode}): {log[-2000:]}")
    return parse_perplexity(log)


def parse_perplexity(log: str) -> dict:
    """Numbers from llama-perplexity output: the final PPL, and with --kl-divergence PPL(Q), PPL(base), mean KLD and same-top-token %."""
    if re.search(r"tokenizes to only|you need at least", log):
        raise RuntimeError("text too short for perplexity")
    num = r"([0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?)"
    out = {}
    if m := re.search(rf"Final estimate: PPL = {num}", log):
        out["ppl"] = float(m.group(1))
    for key, pat in (("ppl_q", rf"Mean PPL\(Q\)\s*:\s*{num}"), ("ppl_base", rf"Mean PPL\(base\)\s*:\s*{num}"),
                     ("kld_mean", rf"Mean\s+KLD:\s*{num}"), ("same_top_pct", rf"Same top p:\s*{num}")):
        if m := re.search(pat, log):
            out[key] = float(m.group(1))
    if "ppl" not in out and "ppl_q" in out:
        out["ppl"] = out["ppl_q"]
    return out


# ---- PyTorch side ----

def torch_perplexity(model, ids: list[int], bos: int, n_ctx: int = N_CTX, chunks: int = CHUNKS) -> dict:
    """llama-perplexity's rule: chunk i is ids[i*n_ctx:(i+1)*n_ctx] with BOS at its first position; positions n_ctx/2 .. n_ctx-2 are scored."""
    import torch
    first = n_ctx // 2
    n_chunk = min(chunks, len(ids) // n_ctx)
    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(n_chunk):
            chunk = list(ids[i * n_ctx:(i + 1) * n_ctx])
            chunk[0] = bos
            logits, _ = model(torch.tensor([chunk]), logits_all=True)
            lp = torch.log_softmax(logits[0, first:n_ctx - 1].float(), dim=-1)
            target = torch.tensor(chunk[first + 1:n_ctx])
            nll -= float(lp.gather(1, target[:, None]).sum())
            count += n_ctx - 1 - first
    return {"ppl": math.exp(nll / count), "scored_tokens": count, "chunks": n_chunk}


def domain_text(val_dir: Path, domain: str, tok, need_tokens: int) -> str:
    from litegpt.records import read_jsonl
    parts, n = [], 0
    for row in read_jsonl(val_dir / f"{domain}.jsonl"):
        parts.append(row["text"])
        n += len(tok(row["text"], add_special_tokens=False)["input_ids"]) + 2
        if n >= need_tokens:
            break
    return "\n\n".join(parts)


def rel(a: float, b: float) -> float:
    return a / b - 1.0


# ---- gates ----

def gate_tokenizer(server: Server, tok, val_dir: Path) -> dict:
    from litegpt.export.to_hf import _doc_strings
    from litegpt.records import read_jsonl
    sets = {d: [r["text"] for r, _ in zip(read_jsonl(val_dir / f"{d}.jsonl"), range(DOCS_PER_DOMAIN))] for d in DOMAINS}
    sets["doc_strings"] = _doc_strings(200)
    sets["edge"] = edge_strings()
    out, passed = {}, True
    for name, texts in sets.items():
        mismatches = []
        for i, text in enumerate(texts):
            want = tok(text)["input_ids"]
            got = server.post("/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
            if got != want:
                j = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                mismatches.append({"index": i, "text": text[:120] if name == "edge" else None, "first_diff": j, "llama": got[j:j + 6], "hf": want[j:j + 6]})
        out[name] = {"strings": len(texts), "identical": len(texts) - len(mismatches), "mismatches": mismatches[:20]}
        if name != "edge":
            passed &= not mismatches
        print(f"G1 {name}: {len(texts) - len(mismatches)}/{len(texts)} identical", flush=True)
    return {"passed": passed, "sets": out}


def greedy_prompts(tok, val_dir: Path) -> dict[str, list[int]]:
    from litegpt.export.to_hf import GREEDY_LENGTHS, _prompts
    from litegpt.records import read_jsonl
    prompts = {f"random-{n}": ids for n, ids in _prompts(len(tok), GREEDY_LENGTHS).items()}
    for d in DOMAINS:
        rows = [r["text"] for r in read_jsonl(val_dir / f"{d}.jsonl")]
        picked = [t for t in rows[-60:] if len(tok(t, add_special_tokens=False)["input_ids"]) >= REAL_PROMPT_TOKENS][:REAL_PROMPTS_PER_DOMAIN]
        for k, t in enumerate(picked):
            prompts[f"{d}-{k}"] = tok(t)["input_ids"][:REAL_PROMPT_TOKENS]
    return prompts


def gate_greedy(server: Server, engine, prompts: dict) -> dict:
    from litegpt.inference import GenerationConfig
    rows, same = {}, 0
    for name, ids in prompts.items():
        want = engine.generate_ids([ids], GenerationConfig(max_new_tokens=GREEDY_NEW, stop_token_ids=()))[0].ids
        r = server.post("/completion", {"prompt": ids, "n_predict": GREEDY_NEW, "temperature": 0, "top_k": 1, "ignore_eos": True,
                                        "cache_prompt": False, "return_tokens": True, "samplers": ["top_k"]})
        got = r.get("tokens", [])
        j = None if got == want else next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
        same += got == want
        rows[name] = {"identical": got == want, "first_diff": j, "prompt_tokens": len(ids), "evaluated": r.get("tokens_evaluated")}
    print(f"greedy: {same}/{len(prompts)} identical", flush=True)
    return {"identical": same, "prompts": len(prompts), "rows": rows}


def gate_chat(server: Server, tok, engine) -> dict:
    from litegpt.inference import GenerationConfig
    rows, template_ok, bos_ok, stop_ok, same = [], True, True, True, 0
    for conv in CONVERSATIONS:
        want_text = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        want_ids = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
        want_ids = list(want_ids["input_ids"] if hasattr(want_ids, "keys") else want_ids)
        got_text = server.post("/apply-template", {"messages": conv})["prompt"]
        reply_ids = engine.generate_ids([want_ids], GenerationConfig(max_new_tokens=128))[0].ids
        want_reply = tok.decode(reply_ids, skip_special_tokens=True)
        r = server.post("/v1/chat/completions", {"messages": conv, "temperature": 0, "top_k": 1, "max_tokens": 128, "samplers": ["top_k"]})
        got_reply = r["choices"][0]["message"]["content"] or ""
        prompt_tokens = r["usage"]["prompt_tokens"]
        t_ok = got_text == want_text
        b_ok = prompt_tokens == len(want_ids) and want_ids[0] == 0 and want_ids[1] != 0
        s_ok = "<|im_end|>" not in got_reply and (r["choices"][0]["finish_reason"] == "stop" or len(reply_ids) >= 128)
        template_ok &= t_ok
        bos_ok &= b_ok
        stop_ok &= s_ok
        same += got_reply.strip() == want_reply.strip()
        rows.append({"template_identical": t_ok, "prompt_tokens": prompt_tokens, "expected_tokens": len(want_ids), "stop_ok": s_ok,
                     "finish_reason": r["choices"][0]["finish_reason"], "reply_identical": got_reply.strip() == want_reply.strip(),
                     "llama_reply": got_reply[:300], "torch_reply": want_reply[:300]})
    print(f"G4 template {template_ok} one-BOS {bos_ok} stop {stop_ok} replies {same}/{len(CONVERSATIONS)}", flush=True)
    return {"template_identical": template_ok, "one_bos": bos_ok, "stops": stop_ok, "replies_identical": same, "rows": rows}


def verdict(res: dict, kind: str) -> dict:
    ppl = res["perplexity"]
    g2_ppl = all(abs(rel(r["F32"]["ppl"], r["torch"]["ppl"])) <= LIMITS["g2_ppl"] for r in ppl.values())
    g2_greedy = res["G2_greedy"]["identical"] >= LIMITS["g2_greedy"] * res["G2_greedy"]["prompts"]
    out = {"G1": res["G1"]["passed"], "G2": g2_ppl and g2_greedy, "G2_ppl": g2_ppl, "G2_greedy": g2_greedy,
           "G3_BF16": all(abs(rel(r["BF16"]["ppl"], r["torch"]["ppl"])) <= LIMITS["bf16_ppl"] for r in ppl.values()),
           "G3_Q8_0": all(rel(ppl[d]["Q8_0"]["ppl"], ppl[d]["BF16"]["ppl"]) <= LIMITS["q8_ppl"] for d in QUANT_DOMAINS),
           "G3_Q4_K_M": all(rel(ppl[d]["Q4_K_M"]["ppl"], ppl[d]["BF16"]["ppl"]) <= LIMITS["q4_ppl"] for d in QUANT_DOMAINS)}
    if kind == "instruct":
        g = res["G4"]
        out["G4"] = g["template_identical"] and g["one_bos"] and g["stops"] and g["replies_identical"] >= LIMITS["g4_greedy"]
    return out


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer

    from litegpt import api
    from litegpt.export.to_gguf import gguf_name
    from litegpt.export.to_hf import KINDS
    from litegpt.records import create_run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=list(KINDS))
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--port", type=int, default=8931)
    a = ap.parse_args(argv)
    hf_dir, gdir, llama_cpp, val_dir = (Path(x).expanduser() for x in (a.hf_dir, a.gguf_dir, a.llama_cpp, a.val_dir))
    files = {q: gdir / gguf_name(a.kind, q) for q in ("F32", "BF16", "Q8_0", "Q4_K_M")}
    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    m = api.load(hf_dir, device="cpu", dtype=torch.float32)
    engine = m.engine
    res: dict = {"settings": {"n_ctx": N_CTX, "chunks": CHUNKS, "threads": THREADS, "domains": DOMAINS, "quant_domains": QUANT_DOMAINS, "limits": LIMITS}}

    with Server(llama_cpp, files["F32"], a.port) as s:
        res["G1"] = gate_tokenizer(s, tok, val_dir)
        res["G2_greedy"] = gate_greedy(s, engine, greedy_prompts(tok, val_dir))
        if a.kind == "instruct":
            res["G4"] = gate_chat(s, tok, engine)
    if a.kind == "instruct":
        with Server(llama_cpp, files["Q8_0"], a.port) as s:
            res["G4_q8"] = gate_chat(s, tok, engine)

    ppl: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        for d in DOMAINS:
            text = domain_text(val_dir, d, tok, (CHUNKS + 1) * N_CTX)
            f = Path(tmp) / f"{d}.txt"
            f.write_text(text, encoding="utf-8")
            row = {"torch": torch_perplexity(m.model, tok(text)["input_ids"], tok.bos_token_id)}
            row["F32"] = run_perplexity(llama_cpp, files["F32"], f)
            kld = Path(tmp) / f"{d}.kld"
            row["BF16"] = run_perplexity(llama_cpp, files["BF16"], f, ["--kl-divergence-base", str(kld)])
            if d in QUANT_DOMAINS:
                for q in ("Q8_0", "Q4_K_M"):
                    row[q] = run_perplexity(llama_cpp, files[q], f, ["--kl-divergence-base", str(kld), "--kl-divergence"])
            kld.unlink(missing_ok=True)
            ppl[d] = row
            print(d, json.dumps({k: v.get("ppl") for k, v in row.items()}), flush=True)
    res["perplexity"] = ppl
    res["verdict"] = verdict(res, a.kind)
    rec = create_run("eval", f"gguf-check-{KINDS[a.kind]['name']}", config={"experiment": "EXP-M5-GGUF", "gates": "G1-G4", "kind": a.kind,
                                                                            "llama_cpp": "b12084", "files": {q: p.name for q, p in files.items()}})
    rec.write_json("eval/gguf_check.json", res)
    print("verdict:", json.dumps(res["verdict"]), "record:", rec.path.name, flush=True)
    return 0 if all(v for k, v in res["verdict"].items() if k != "G3_Q4_K_M") else 1


if __name__ == "__main__":
    raise SystemExit(main())
