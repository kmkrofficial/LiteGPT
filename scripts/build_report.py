"""
Build the LiteGPT V2 report from committed data. Nothing in the output is typed by hand.

  python -m scripts.build_report            # writes runs/report-latest/REPORT.md and runs/report-latest/figures/*.svg
  python -m scripts.build_report --check    # exit 1 if the committed report differs from a fresh build (stale report)

Inputs: docs/spec/v2.yaml, the model code (parameter / FLOP introspection on the meta device), the V1 model code
(scripts/legacy_v1/model.py, for the V1 column), and the run records in runs/<run-id>/ (run.json + bench/*.json).
Every input record is validated (litegpt.records.validate_run_record) and listed in the report with its sha256.
Output is deterministic: no timestamps, fixed SVG hash salt, sorted inputs. Generated outputs live in runs/ like every other
generated artifact; there is no separate top-level reports/ directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

import warnings

import matplotlib

warnings.filterwarnings("ignore", message="Tight layout not applied")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from litegpt.config import get_config  # noqa: E402
from litegpt.records import RUNS_ROOT, validate_run_record  # noqa: E402
from litegpt.spec import REPO_ROOT, load_spec, spec_sha256  # noqa: E402

RUNS_DIR = RUNS_ROOT
OUTPUT_DIR = RUNS_ROOT / "report-latest"  # generated artifact, not an experiment run (it has no run.json)

# Reference palette (dataviz skill, categorical slots 1-2 + neutrals), light surface.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "svg.hashsalt": "litegpt-v2-report", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sha_short(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def save_svg(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


# ---------------------------------------------------------------- model introspection ------------------------------
def _load_v1():
    spec = importlib.util.spec_from_file_location("litegpt_v1_model", REPO_ROOT / "scripts" / "legacy_v1" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_stats() -> list[dict]:
    import torch
    from litegpt.model import LiteGPT
    rows = []
    v1 = _load_v1()
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        # V1 as trained: bias=False (train.py default, config/train_fineweb.py does not override), vocab 50304, T=1024
        m1 = v1.GPT(v1.GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, bias=False))
    c1 = m1.config
    n1_total = m1.get_num_params(non_embedding=False)
    emb1 = m1.transformer.wte.weight.numel() + m1.transformer.wpe.weight.numel()
    attn1 = 12 * c1.n_layer * c1.n_head * (c1.n_embd // c1.n_head) * c1.block_size
    rows.append({"name": "V1 (GPT-2 Small)", "layers": c1.n_layer, "d": c1.n_embd, "heads": f"{c1.n_head} MHA",
                 "ffn": "GELU 4d (3072)", "pos": "learned", "norm": "LayerNorm", "vocab": c1.vocab_size,
                 "ctx": c1.block_size, "total": n1_total, "embedding": emb1, "non_embedding": n1_total - emb1,
                 "flops_full": 6 * m1.get_num_params(True) + attn1, "flops_causal": 6 * m1.get_num_params(True) + attn1 / 2})
    for key, label in (("litegpt-v2-24l", "V2 24L (fallback)"), ("litegpt-v2-28l", "V2 28L (approved)")):
        cfg = get_config(key)
        with torch.device("meta"):
            m = LiteGPT(cfg)
        emb = m.model.embed_tokens.weight.numel()
        rows.append({"name": label, "layers": cfg.n_layer, "d": cfg.n_embd, "heads": f"{cfg.n_head}Q/{cfg.n_kv_head}KV GQA",
                     "ffn": f"SwiGLU ({cfg.ffn_hidden})", "pos": "RoPE", "norm": "RMSNorm + QK-norm",
                     "vocab": cfg.vocab_size, "ctx": cfg.block_size, "total": m.num_params(), "embedding": emb,
                     "non_embedding": m.num_params() - emb, "flops_full": m.flops_per_token(cfg.block_size, "full"),
                     "flops_causal": m.flops_per_token(cfg.block_size, "causal")})
    return rows


def v2_module_breakdown() -> list[tuple[str, int]]:
    import torch
    from litegpt.model import LiteGPT
    with torch.device("meta"):
        m = LiteGPT(get_config("litegpt-v2-28l"))
    groups = {"token embedding (tied with output)": 0, "attention projections (q,k,v,o)": 0, "FFN (SwiGLU)": 0,
              "norms (RMSNorm + QK-norm)": 0}
    for n, p in m.named_parameters():
        if "embed_tokens" in n or n.startswith("lm_head"):
            groups["token embedding (tied with output)"] += p.numel()
        elif any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj")):
            groups["attention projections (q,k,v,o)"] += p.numel()
        elif "mlp" in n:
            groups["FFN (SwiGLU)"] += p.numel()
        else:
            groups["norms (RMSNorm + QK-norm)"] += p.numel()
    assert sum(groups.values()) == m.num_params()
    return list(groups.items())


# ---------------------------------------------------------------- charts --------------------------------------------
def chart_params(stats, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    names = [s["name"] for s in stats][::-1]
    non_emb = [s["non_embedding"] / 1e6 for s in stats][::-1]
    emb = [s["embedding"] / 1e6 for s in stats][::-1]
    ax.barh(names, non_emb, color=BLUE, height=0.5, edgecolor=SURFACE, linewidth=2, label="transformer layers")
    ax.barh(names, emb, left=non_emb, color=ORANGE, height=0.5, edgecolor=SURFACE, linewidth=2, label="embeddings")
    for i, (a, b) in enumerate(zip(non_emb, emb)):
        ax.text(a + b + 2, i, f"{a + b:.1f}M", va="center", color=INK, fontsize=9)
    ax.set_xlabel("parameters (millions)")
    ax.set_xlim(0, max(a + b for a, b in zip(non_emb, emb)) * 1.18)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Parameters: V1 vs V2", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_svg(fig, out / "params.svg")
    return "figures/params.svg"


def chart_tokps(results, required, title, fname, out: Path) -> str | None:
    ok = [r for r in results if r.get("status") == "ok" and r.get("tok_per_s") and not r.get("vram_pressure")]
    if not ok:
        return None
    labels = [f"{r['config'].replace('litegpt-', '')} mb{r['micro_batch']}x{r['seq_len']} {r['backend']}"
              f"{' +compile' if r['compile'] else ''}" for r in ok][::-1]
    vals = [r["tok_per_s"] for r in ok][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 + 0.42 * len(ok)))
    ax.barh(labels, vals, color=BLUE, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v / 1000:.1f}k", va="center", fontsize=9, color=INK)
    ax.axvline(required, color=MUTED, linestyle="--", linewidth=1)
    ax.text(required, len(vals) - 0.35, f" required {required / 1000:.1f}k tok/s", color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(0, max(max(vals), required) * 1.15)
    ax.set_xlabel("training tokens / second (higher is better)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    save_svg(fig, out / fname)
    return f"figures/{fname}"


# ---------------------------------------------------------------- sections ------------------------------------------
def load_bench_runs() -> list[dict]:
    runs = []
    for run_json in sorted(RUNS_DIR.glob("*/run.json")):
        bench = run_json.parent / "bench" / "train_mem.json"
        if bench.exists():
            run = json.loads(run_json.read_text(encoding="utf-8"))
            problems = validate_run_record(run)
            if problems:
                raise ValueError(f"{run_json.parent.name}: invalid run record: {problems}")
            runs.append({"run": run, "bench": json.loads(bench.read_text(encoding="utf-8")), "files": [run_json, bench]})
    return runs


def load_equivalence_runs() -> list[dict]:
    out = []
    for run_json in sorted(RUNS_DIR.glob("*/run.json")):
        eq = run_json.parent / "bench" / "compile_equivalence.json"
        if eq.exists():
            run = json.loads(run_json.read_text(encoding="utf-8"))
            problems = validate_run_record(run)
            if problems:
                raise ValueError(f"{run_json.parent.name}: invalid run record: {problems}")
            out.append({"run": run, "eq": json.loads(eq.read_text(encoding="utf-8")), "files": [run_json, eq]})
    return out


def load_sweep_runs() -> list[dict]:
    out = []
    for run_json in sorted(RUNS_DIR.glob("*/run.json")):
        sw = run_json.parent / "bench" / "power_sweep.json"
        if sw.exists():
            run = json.loads(run_json.read_text(encoding="utf-8"))
            problems = validate_run_record(run)
            if problems:
                raise ValueError(f"{run_json.parent.name}: invalid run record: {problems}")
            out.append({"run": run, "sweep": json.loads(sw.read_text(encoding="utf-8")), "files": [run_json, sw]})
    return out


def chart_sweep(sweep: dict, title: str, fname: str, out: Path) -> str | None:
    """2x2: tok/s vs power, tok/s/W vs power, tok/s vs max temperature, Pareto frontier (throughput vs thermal load)."""
    pts = [e for e in sweep["points"] if e.get("valid")]
    if not pts:
        return None
    m = [e["metrics"] for e in pts]
    el = [bool(e.get("eligible")) for e in pts]
    labels = [e["spec"] for e in pts]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.2))
    panels = [(axes[0][0], "power_w_mean", "tok_per_s", "mean GPU power (W)", "tok/s"),
              (axes[0][1], "power_w_mean", "tok_per_s_per_w", "mean GPU power (W)", "tok/s per W"),
              (axes[1][0], "temp_c_max", "tok_per_s", "max GPU temperature (C)", "tok/s"),
              (axes[1][1], "temp_c_max", "tok_per_s", "max GPU temperature (C) (thermal load)", "tok/s")]
    for ax, xk, yk, xl, yl in panels:
        xs, ys = [d[xk] for d in m], [d[yk] for d in m]
        for x, y, e in zip(xs, ys, el):
            ax.scatter([x], [y], s=46, color=BLUE if e else SURFACE, edgecolors=BLUE, linewidths=1.5, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=INK2)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
    front = sorted([d for d, e in zip(m, pts) if e["spec"] in sweep.get("pareto_front", [])], key=lambda d: d["temp_c_max"])
    if len(front) >= 2:
        axes[1][1].plot([d["temp_c_max"] for d in front], [d["tok_per_s"] for d in front], color=INK2, linestyle="--",
                        linewidth=1, zorder=2)
    axes[1][1].set_title("Pareto frontier (dashed)", fontsize=8, loc="left", color=INK2)
    fig.suptitle(title + "   (filled = production candidate, hollow = valid but not eligible)", fontsize=9, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout()
    save_svg(fig, out / fname)
    return f"figures/{fname}"


# ---------------------------------------------------------------- data (M1) ----------------------------------------
def load_data_run() -> dict | None:
    """Latest data-prepare run that has a mix manifest, plus the frozen tokenizer's training report."""
    cands = sorted(p for p in RUNS_DIR.glob("*-data-prepare-*") if (p / "mix_manifest.json").exists())
    if not cands:
        return None
    run_dir = cands[-1]
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if validate_run_record(run):
        return None
    d = {"run_dir": run_dir, "run": run, "files": [run_dir / "run.json", run_dir / "mix_manifest.json"],
         "mix": json.loads((run_dir / "mix_manifest.json").read_text(encoding="utf-8"))}
    st = run_dir / "data_stats.json"
    d["stats"] = json.loads(st.read_text(encoding="utf-8")) if st.exists() else None
    if d["stats"] is not None:
        d["files"].append(st)
    spec = load_spec()
    tok_run = spec["tokenizer"].get("frozen_from_run")
    rep = RUNS_DIR / str(tok_run) / "tokenizer_report.json" if tok_run else None
    d["tok"] = json.loads(rep.read_text(encoding="utf-8")) if rep and rep.exists() else None
    if d["tok"] is not None:
        d["files"].append(rep)
    import yaml
    d["sources"] = yaml.safe_load((REPO_ROOT / "docs" / "spec" / "data_sources.yaml").read_text(encoding="utf-8"))
    return d


def chart_mixture(mixture: dict, out: Path) -> str:
    names = sorted(set(mixture["stable"]) | set(mixture["decay"]), key=lambda n: -mixture["stable"].get(n, 0))
    y = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(names) + 1.2))
    h = 0.38
    ax.barh([v - h / 2 for v in y], [100 * mixture["stable"].get(n, 0) for n in names], height=h, color=BLUE, label="stable (to 9B / 14B)")
    ax.barh([v + h / 2 for v in y], [100 * mixture["decay"].get(n, 0) for n in names], height=h, color=ORANGE, label="decay (last 1B)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("share of training tokens (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Pretraining mixture (spec 2.0.0-spec.6)", loc="left", fontsize=11)
    fig.tight_layout()
    save_svg(fig, out / "data_mixture.svg")
    return "figures/data_mixture.svg"


def data_section(A, out: Path) -> list:
    d = load_data_run()
    A("## 3c. Data and tokenizer (M1)\n")
    if d is None:
        A("_No mixed data run yet (M1 step 6)._\n")
        return []
    mix, stats, src_cfg = d["mix"], d["stats"], d["sources"]
    m = src_cfg["mixture"]
    st_tok, dc_tok = m["stable_tokens"], m["decay_tokens"]
    tokens = (stats or {}).get("tokens", {})
    A(f"Data run `{d['run']['run_id']}`; sources file sha256 `{d['run']['config'].get('sources_sha256', '')[:16]}…`.\n")
    A("### Tokens prepared vs plan (millions of tokens)\n")
    rows = []
    for sub in sorted(set(m["stable"]) | set(m["decay"])):
        need_s, need_d = m["stable"].get(sub, 0) * st_tok / 1e6, m["decay"].get(sub, 0) * dc_tok / 1e6
        have_s, have_d, have_v = (tokens.get(f"{sub}/{p}", 0) / 1e6 for p in ("stable", "decay", "val"))
        ratios = [h / n for h, n in ((have_s, need_s), (have_d, need_d)) if n]
        rows.append([sub, f"{need_s:,.0f}", f"{have_s:,.0f}", f"{need_d:,.0f}", f"{have_d:,.0f}", f"{have_v:,.1f}",
                     f"{min(ratios):.2f}x" if ratios else "-"])
    A(md_table(["source", "stable need", "stable have", "decay need", "decay have", "val have", "min margin"], rows))
    A("")
    A(f"![mixture]({chart_mixture({'stable': m['stable'], 'decay': m['decay']}, out / 'figures')})\n")
    if d["tok"]:
        A("### Tokenizer (frozen) compression, bytes per token (higher is better)\n")
        comp = d["tok"]["compression"]
        A(md_table(["domain", "LiteGPT 32k", "GPT-2 50k", "SmolLM2 49k", "vs GPT-2"],
                   [[k, f"{v['litegpt_bytes_per_token']:.3f}", f"{v.get('gpt2_bytes_per_token', 0):.3f}",
                     f"{v.get('smollm2_bytes_per_token', 0):.3f}",
                     f"{v['litegpt_bytes_per_token'] / v['gpt2_bytes_per_token']:.2f}x" if v.get("gpt2_bytes_per_token") else "-"]
                    for k, v in comp.items()]))
        A(f"\nTokenizer sha256 `{d['tok']['sha256'][:16]}…`; trained on {d['tok']['train']['bytes'] / 1e9:.2f} GB "
          f"({d['tok']['train']['docs']:,} documents).\n")
    if stats:
        A("### Code filters (documents dropped per rule)\n")
        drops = stats["dropped"]
        rules = sorted({k.split(":")[1] for k in drops if k.startswith("code_")})
        langs = sorted({k.split(":")[0] for k in drops if k.startswith("code_")})
        A(md_table(["language"] + rules, [[l] + [f"{drops.get(f'{l}:{r}', 0):,}" for r in rules] for l in langs]))
        A(f"\nE-mail addresses redacted in {stats['emails_redacted_docs']:,} documents. `keep_fraction` drops only bound disk use.\n")
        A("### Decontamination (documents dropped, 13-gram overlap with the 18 pinned benchmarks)\n")
        dec = stats["decontam"]
        subs = sorted({k.split(":")[0] for k in dec})
        benches = sorted({k.split(":")[1] for k in dec}, key=lambda b: -sum(v for k, v in dec.items() if k.endswith(":" + b)))
        rows = []
        for s in subs:
            kept = sum(v for k, v in stats["bytes"].items() if k.startswith(s + "/") and not k.endswith("decontam_dropped"))
            dropped = stats["bytes"].get(f"{s}/decontam_dropped", 0)
            rows.append([s, f"{100 * dropped / max(kept + dropped, 1):.2f}%"] + [f"{dec.get(f'{s}:{b}', 0):,}" for b in benches[:8]])
        A(md_table(["source", "bytes dropped"] + benches[:8], rows))
        A("")
    A("### Checks (EXP-M1-DATA)\n")
    dup = mix.get("duplicates", {})
    A(md_table(["source", "train docs", "within-source dup", "cross-source dup", "val docs", "val also in training"],
               [[s, f"{r['train_docs']:,}", f"{100 * r['within_rate']:.2f}%", f"{100 * r['cross_rate']:.2f}%", f"{r['val_docs']:,}",
                 r["val_docs_also_in_training"]] for s, r in sorted(dup.items())]))
    keys = mix.get("val_key_check", {})
    pp = mix.get("prefix_proportions_stable", {})
    A("")
    A(md_table(["check", "result"], [
        ["validation keys outside the val hash range", sum(v["violations"] for v in keys.values())],
        ["stable index: max deviation from weights at 1M-token prefixes", f"{100 * pp.get('max_abs_deviation', 0):.3f} pp"],
        ["loader resume equals uninterrupted run (stable index)", mix.get("resume_check_stable")],
        ["shortfalls", ", ".join(mix.get("shortfalls") or []) or "none"],
    ]))
    A("\n### Frozen hashes\n")
    spec = load_spec()
    A(md_table(["item", "sha256"], [
        ["tokenizer", f"`{spec['tokenizer']['sha256']}`"],
        ["val_core (token content)", f"`{mix['val_core']['sha256']}`"],
        ["data order (stable + decay indexes)", f"`{mix['data_order_sha256']}`"],
    ] + [[f"val_text/{k}.jsonl ({v['docs']:,} docs, {v['bytes'] / 1e6:.1f} MB)", f"`{v['sha256']}`"] for k, v in sorted(mix.get("val_text", {}).items())]))
    A("")
    return d["files"]


def env_label(run: dict) -> str:
    e = run["environment"]
    return f"WSL2 ({e['wsl_distro_and_kernel']})" if e.get("wsl_distro_and_kernel") else f"{e['os']} native"


def build(out: Path) -> None:
    spec = load_spec()
    stats = model_stats()
    p = spec["training"]["pretraining"]
    g = spec["training"]["extension_gate"]
    L: list[str] = []
    A = L.append

    A("# LiteGPT V2: generated report\n")
    A("> Generated by `python -m scripts.build_report` from the run records in `runs/`. Do not edit by hand.\n")
    A("## 1. Specification\n")
    A(md_table(["item", "value"], [
        ["spec version", f"`{spec['spec_version'].split()[0]}`"], ["spec sha256", f"`{spec_sha256()[:16]}…`"],
        ["code licence / model licence", f"{spec['licensing']['code']} / {spec['licensing']['model']}"],
        ["python / torch (pinned)", f"{spec['environment']['python']} / {spec['environment']['pins']['torch']}"],
        ["attention", f"{spec['environment']['attention']['mandatory']} mandatory; "
                      f"{spec['environment']['attention']['optional']['package']} "
                      f"{spec['environment']['attention']['optional']['version']} optional ({spec['environment']['attention']['optional']['status']})"],
        ["pretraining budget", f"{p['budget_initial_tokens'] / 1e9:.0f}B tokens (max {p['budget_max_tokens'] / 1e9:.0f}B), "
                               f"~{p['target_wall_clock_days']} days, requires {p['required_sustained_tok_per_s']:,} tok/s sustained"],
        ["extension gate (pre-registered)",
         f"Δ val_bpb over [{g['bpb_window_a'][0] / 1e9}–{g['bpb_window_a'][-1] / 1e9}]B vs "
         f"[{g['bpb_window_b'][0] / 1e9}–{g['bpb_window_b'][-1] / 1e9}]B ≥ {g['threshold_bpb']} → extend to 15B, else stop at 10B"],
    ]))

    A("\n## 2. Architecture and parameters\n")
    A(md_table(["model", "layers", "d_model", "attention", "FFN", "positions", "norm", "vocab", "context"],
               [[s["name"], s["layers"], s["d"], s["heads"], s["ffn"], s["pos"], s["norm"], f"{s['vocab']:,}", f"{s['ctx']:,}"]
                for s in stats]))
    A("")
    A(md_table(["model", "total params", "embedding params", "non-embedding params", "FLOPs/token (full-attn, MFU convention)",
                "FLOPs/token (causal)"],
               [[s["name"], f"{s['total'] / 1e6:.2f}M", f"{s['embedding'] / 1e6:.2f}M", f"{s['non_embedding'] / 1e6:.2f}M",
                 f"{s['flops_full'] / 1e9:.3f} G", f"{s['flops_causal'] / 1e9:.3f} G"] for s in stats]))
    A("\nV1 is shown as trained (bias-free, vocab 50,304, context 1,024); its embedding count includes the learned position table.\n")
    A(f"![parameters]({chart_params(stats, out / 'figures')})\n")
    A("V2 28L parameter breakdown:\n")
    bd = v2_module_breakdown()
    tot = sum(n for _, n in bd)
    A(md_table(["component", "params", "share"], [[k, f"{n / 1e6:.2f}M", f"{100 * n / tot:.1f}%"] for k, n in bd]))

    A("\n## 3. Training throughput and memory (M0 benchmarks)\n")
    runs = load_bench_runs()
    if not runs:
        A("_No benchmark runs found in runs/ (`python -m benchmarks.train_mem sweep --preset m0`)._\n")
    for i, r in enumerate(runs):
        run, bench = r["run"], r["bench"]
        env = run["environment"]
        A(f"### {env_label(run)}: run `{run['run_id']}`\n")
        A(f"GPU {env.get('gpu_name')} ({env.get('gpu_vram_gib')} GiB), driver {env.get('nvidia_driver_version')}, "
          f"torch {env['torch_version']} (CUDA {env['torch_cuda_version']}), Python {env['python']}, "
          f"uv.lock `{str(env['uv_lock_sha256'])[:12]}`, git `{str(run['git_commit'])[:7]}`"
          f"{' (dirty)' if run['git_dirty'] else ''}. Measured bf16 matmul ceiling: "
          f"{(bench.get('matmul_ceiling') or {}).get('tflops', 'n/a')} TFLOPS.\n")
        rows = []
        for x in bench["results"]:
            rows.append([x["config"].replace("litegpt-", ""), f"{x['micro_batch']}×{x['seq_len']}", x["backend"],
                         "yes" if x["compile"] else "no",
                         x["status"] + (" ⚠ VRAM pressure (spill risk, not a valid throughput)" if x.get("vram_pressure") else ""),
                         f"{x['tok_per_s']:,.0f}" if x.get("tok_per_s") else "–",
                         f"{x['tflops_full']:.1f}" if x.get("tflops_full") else "–",
                         f"{100 * x['mfu_full_vs_matmul_ceiling']:.0f}%" if x.get("mfu_full_vs_matmul_ceiling") else "–",
                         f"{x['peak_alloc_gib']:.2f}" if x.get("peak_alloc_gib") is not None else "–",
                         {True: "yes", False: "no", None: "n/a"}[x.get("meets_16_5k_tok_per_s")]])
        A(md_table(["config", "micro-batch×seq", "attention", "compile", "status", "tok/s", "TFLOPS (full)",
                    "MFU vs ceiling", "peak VRAM GiB", "≥16.5k tok/s"], rows))
        A("")
        fig = chart_tokps(bench["results"], bench["required_tok_per_s"], f"Training throughput: {env_label(run)}",
                          f"throughput_{i}.svg", out / "figures")
        if fig:
            A(f"![throughput]({fig})\n")
        ab = bench.get("ab_summary")
        if ab:
            keys = [k for k in ab if isinstance(ab[k], dict)]
            A(md_table(["side", "runs", "tok/s mean (min-max)", "peak VRAM GiB", "warmup s", "final loss"],
                       [[k, ab[k]["n"], f"{ab[k]['tok_per_s_mean']:,.0f} ({ab[k]['tok_per_s_min']:,.0f}-{ab[k]['tok_per_s_max']:,.0f})",
                         ab[k]["peak_alloc_gib_max"], ab[k].get("warmup_s_mean", "–"), ab[k].get("final_loss_mean", "–")]
                        for k in keys]))
            extra = f"**Speedup B over A: {ab['speedup_B_over_A']:.3f}x**" if "speedup_B_over_A" in ab else ""
            if "final_loss_rel_diff_B_vs_A" in ab:
                extra += f" (final-loss relative difference {100 * ab['final_loss_rel_diff_B_vs_A']:.1f}%, training drift on a repeated synthetic batch)"
            A("\n" + extra + "\n")
        errs = [x for x in bench["results"] if x.get("error")]
        for x in errs:
            A(f"- `{x['config']}` mb{x['micro_batch']}×{x['seq_len']} {x['backend']}{' compile' if x['compile'] else ''}: "
              f"`{x['error'][:200].replace(chr(10), ' ')}`")
        sus = [x for x in bench["results"] if x.get("sustained")]
        for x in sus:
            s = x["sustained"]
            A(f"- Sustained {s['seconds']} s: first-minute {s['first_window']:,.0f} → last-minute {s['last_window']:,.0f} tok/s "
              f"(ratio {s['last_over_first']}); power-capped {100 * x['telemetry'].get('frac_power_capped', 0):.0f}% / thermally "
              f"throttled {100 * x['telemetry'].get('frac_thermal_throttled', 0):.0f}% of busy samples; mean GPU power "
              f"{x['telemetry'].get('power_w_mean', 0):.0f} W"
              + (f", enforced power limit {x['telemetry']['power_limit_w_min']:.0f}-{x['telemetry']['power_limit_w_max']:.0f} W"
                 if x['telemetry'].get('power_limit_w_max') else " (power limit not recorded by this run)") + ".")
        A("")

    eqs = load_equivalence_runs()
    if eqs:
        A("### Compiled vs eager equivalence (identical weights, bf16 autocast)\n")
        rows = []
        for r in eqs:
            e = r["eq"]
            rows.append([env_label(r["run"]), f"{e['config'].replace('litegpt-', '')} {e['micro_batch']}x{e['seq_len']}",
                         f"{e['loss_abs_diff']:.4f}", f"{100 * e['logits_rel_l2_diff']:.2f}%", f"{100 * e['grad_norm_rel_diff']:.3f}%",
                         "pass" if e["passed"] else "FAIL"])
        A(md_table(["environment", "config", "loss abs diff", "logits rel L2 diff", "grad-norm rel diff", "result"], rows))
        A("")

    sweeps = load_sweep_runs()
    A("## 3b. GPU operating-point sweep (EXP-M0-POWER)\n")
    if not sweeps:
        A("_No sweep runs yet (`python -m benchmarks.power_sweep screen`)._\n")
    for i, r in enumerate(sweeps):
        run, sw = r["run"], r["sweep"]
        host = run.get("host", {})
        A(f"### {sw['stage']}: {env_label(run)}, run `{run['run_id']}`\n")
        A(f"Workload: {sw['workload']}. Elevated: {host.get('windows_elevated')}; AC power: {host.get('ac_power_online')}; "
          f"Lenovo mode: {host.get('lenovo_smart_fan_mode')}; voltage readout: {host.get('voltage_readout')}. "
          f"Notes: {sw.get('notes') or '-'}\n")
        if sw.get("protocol"):
            A(f"Protocol: {sw['protocol']}; natural clock {sw.get('natural_clock_mhz', '-')} MHz"
              + (f"; {sw['caps_skipped_reason']}" if sw.get("caps_skipped_reason") else "") + "\n")
        rows = []
        for e in sw.get("segments") or sw["points"]:
            mm = e.get("metrics", {})
            f = (lambda k, fmt="{:,.0f}": fmt.format(mm[k]) if mm.get(k) is not None else "-")
            rows.append([e["spec"], e["status"], "yes" if e.get("valid") else "no", "yes" if e.get("eligible") else "no",
                         f("tok_per_s"), f("tok_per_s_per_w", "{:.1f}"), f("power_w_mean", "{:.0f}") + " / " + f("power_w_peak", "{:.0f}"),
                         f("power_limit_w_min", "{:.0f}") + "-" + f("power_limit_w_max", "{:.0f}"), f("sm_clock_mhz_mean"),
                         f("temp_c_mean", "{:.0f}") + " / " + f("temp_c_max", "{:.0f}"), f("util_pct_mean", "{:.0f}"),
                         f("vram_used_mib_max"), f("frac_thermal_throttled", "{:.0%}"), f("frac_power_capped", "{:.0%}"),
                         f("measured_seconds", "{:.0f}"), "; ".join(e.get("invalid_reasons") or e.get("ineligible_reasons") or e.get("notes") or []) or "-"])
        A(md_table(["point", "status", "valid", "candidate", "tok/s", "tok/s/W", "GPU W mean/peak", "power limit W",
                    "SM MHz mean", "temp C mean/max", "util %", "VRAM MiB", "thermal thr.", "power cap", "measured s",
                    "reasons"], rows))
        A(f"\nRecommended (pre-registered rule): **{sw.get('recommended') or 'none'}**; Pareto front: "
          f"{', '.join(sw.get('pareto_front') or []) or '-'}"
          + (f"; **soak passed: {sw['soak_passed']}**" if "soak_passed" in sw else "") + "\n")
        fig = chart_sweep(sw, f"{sw['stage']} sweep", f"power_sweep_{i}.svg", out / "figures")
        if fig:
            A(f"![power sweep]({fig})\n")

    data_files = data_section(A, out)
    A("## 4. Data provenance of this report\n")
    files = ([f for r in runs for f in r["files"]] + [f for r in eqs for f in r["files"]]
             + [f for r in sweeps for f in r["files"]] + data_files)
    A(md_table(["file", "sha256 (12)"], [[f"`runs/{f.relative_to(RUNS_DIR).as_posix()}`", sha_short(f)] for f in files])
      if files else "_No input run records._")
    A("\n## 5. Reproduce\n")
    A("```bash\nuv sync --locked\nuv run pytest\nuv run python -m benchmarks.train_mem sweep --preset m0\n"
      "uv run python -m scripts.build_report\n```\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes().replace(b"\r\n", b"\n")
            for p in sorted(root.rglob("*")) if p.is_file()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if runs/report-latest is stale")
    a = ap.parse_args(argv)
    if a.check:
        with tempfile.TemporaryDirectory() as td:
            fresh = Path(td) / "latest"
            build(fresh)
            want, have = _tree(fresh), (_tree(OUTPUT_DIR) if OUTPUT_DIR.exists() else {})
            bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        if bad:
            print("report is stale; rebuild with `python -m scripts.build_report`. Differing files:", *bad, sep="\n  ")
            return 1
        print("report is up to date")
        return 0
    if OUTPUT_DIR.exists():
        import shutil
        assert OUTPUT_DIR.name.startswith("report-") and not (OUTPUT_DIR / "run.json").exists(), \
            "refusing to delete a directory that is not a generated report"
        shutil.rmtree(OUTPUT_DIR)
    build(OUTPUT_DIR)
    print("wrote", OUTPUT_DIR / "REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
