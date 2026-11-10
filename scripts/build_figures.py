"""Build every chart in docs/figures/ from the run records in runs/. Nothing is typed by hand, and the output is deterministic.

    python -m scripts.build_figures            # writes docs/figures/
    python -m scripts.build_figures --check    # exit 1 if a committed figure differs from a fresh build

Needs matplotlib (`uv sync --group train`). V2 is always blue and V1 always orange, in every chart. Other models are gray, named on the axis.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "figures"
if "--check" in sys.argv:
    COMMITTED, OUT = OUT, Path(tempfile.mkdtemp())

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#8a8984"
V2, V1 = "LiteGPT V2", "LiteGPT V1"
ENTITY = {V2: BLUE, "V2-Instruct": BLUE, V1: ORANGE, "V1-Instruct": ORANGE, "V2": BLUE}

plt.rcParams.update({
    "svg.hashsalt": "litegpt-v2-figures", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "axes.titlelocation": "left", "axes.titlecolor": INK2, "lines.linewidth": 2, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
INDEX: list[tuple[str, str, str, str, str]] = []


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def metrics(p: Path) -> list[dict]:
    """Training-step rows of a metrics file; a run also logs other kinds of row there."""
    return [r for r in jl(p) if "loss" in r and "tokens" in r]


def runs(kind: str, name: str) -> list[Path]:
    """Runs of exactly this kind and name, oldest first."""
    pat = re.compile(rf"^\d{{8}}T\d{{6}}Z-{kind}-{re.escape(name)}-[0-9a-f]{{7}}$")
    return sorted(p for p in RUNS.iterdir() if p.is_dir() and pat.match(p.name))


def color(name: str) -> str:
    return ENTITY.get(name, GRAY)


def finish(fig, name: str, section: str, title: str, sub: str, source: str, top: float = 0.86) -> None:
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.012, 0.985 - 0.32 / fig.get_figheight(), sub, ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, top))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    INDEX.append((name, section, title, sub, source))


def grid(ax, axis="x") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbars(ax, items: list[tuple[str, float]], fmt: str = "{:.3f}", title: str = "", xmax=None, label_names=(V2, V1, "V2-Instruct", "V1-Instruct", "V2")) -> None:
    """Thin bars from zero, best first as given. Only the leader and the LiteGPT models carry a value; the tables in the docs have the rest."""
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    y = list(range(len(items)))[::-1]
    ax.barh(y, vals, height=0.62, color=[color(n) for n in names], edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, names)
    for lab, n in zip(ax.get_yticklabels(), names):
        if n in ENTITY:
            lab.set_fontweight("bold")
            lab.set_color(INK)
    top = xmax or max(max(vals), 1e-9) * 1.22
    ax.set_xlim(0, top)
    for i, (n, v) in enumerate(items):
        if i == 0 or n in label_names:
            ax.text(v + top * 0.012, y[i], fmt.format(v), va="center", fontsize=7.5, color=INK)
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)
    grid(ax)


def scatter_labels(ax, pts: list[tuple[str, float, float]], offsets: dict | None = None) -> None:
    offsets = offsets or {}
    for n, x, y in pts:
        ax.scatter([x], [y], s=52 if n in ENTITY else 34, color=color(n), edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(n, (x, y), textcoords="offset points", xytext=offsets.get(n, (6, 4)), fontsize=8, color=INK if n in ENTITY else INK2, fontweight="bold" if n in ENTITY else "normal")
    grid(ax, "both")


# ---- records ----

BASE = {"Qwen3-0.6B-Base": "Qwen3-0.6B", "Qwen2.5-0.5B": "Qwen2.5-0.5B", "SmolLM2-360M": "SmolLM2-360M", "SmolLM2-135M": "SmolLM2-135M", "gemma-3-270m": "Gemma 3 270M",
        "litegpt-final": V2, "pythia-410m": "Pythia-410M", "gpt2-medium": "GPT-2 Medium", "LiteGPT-Base": V1, "gpt2": "GPT-2", "pythia-160m": "Pythia-160M"}
CHAT = {"Qwen3-0.6B": "Qwen3-0.6B", "SmolLM2-360M-Instruct": "SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct", "gemma-3-270m-it": "Gemma 3 270M-it",
        "SmolLM2-135M-Instruct": "SmolLM2-135M-Instruct", "litegpt-final": "V2-Instruct", "LiteGPT-Instruct": "V1-Instruct"}
ZERO_SHOT = {"hellaswag": ("HellaSwag", 0.25), "arc_easy": ("ARC-Easy", 0.25), "arc_challenge": ("ARC-Challenge", 0.25), "piqa": ("PIQA", 0.5), "openbookqa": ("OpenBookQA", 0.25),
             "commonsense_qa": ("CommonsenseQA", 0.2), "winogrande": ("Winogrande", 0.5), "mmlu_continuation": ("MMLU", 0.25), "truthfulqa_mc1": ("TruthfulQA MC1", None), "truthfulqa_mc2": ("TruthfulQA MC2", None)}
FEW = {"triviaqa": "TriviaQA", "gsm8k": "GSM8K", "humaneval": "HumanEval", "mbpp": "MBPP"}
T2D = {"qwen3": "Qwen3-0.6B", "gemma": "Gemma 3 270M", "smollm2": "SmolLM2-135M", "v2": "V2"}
SHORT = {"qwen3": "Qwen3", "gemma": "Gemma", "smollm2": "SmolLM2", "v2": "V2"}


def base_scores() -> dict[str, dict[str, float]]:
    out = {}
    for run, name in BASE.items():
        for d in reversed(runs("eval-base", run)):
            s = d / "eval" / "summary.json"
            if s.exists() and load(d / "run.json")["config"].get("limit") is None and "hellaswag" in load(s)["headline"]:
                out[name] = {k: v["value"] for k, v in load(s)["headline"].items()}
                break
    out = {k: {**v, "mean": statistics.fmean(v[t] for t in ZERO_SHOT)} for k, v in out.items()}
    return out


def native(kind: str, table: dict) -> dict[str, dict]:
    out = {}
    for run, name in table.items():
        out[name] = load(runs(f"eval-{kind}", run)[-1] / "eval" / f"{kind}.json")
    return out


def chat_scores() -> dict[str, dict[str, float]]:
    out = {}
    for run, name in CHAT.items():
        m = {}
        for d in runs("eval-instruct", run):      # later records correct earlier ones
            s = d / "eval" / "summary.json"
            if not s.exists():
                continue
            for k, v in load(s)["headline"].items():
                if k.startswith("gsm8k"):
                    continue
                m["humaneval" if k.startswith("humaneval") else "mbpp" if k.startswith("mbpp") else k] = v["value"]
            g = d / "eval" / "gsm8k_chat.json"
            if g.exists():                        # last-number scoring: strict match gives every model 0, so older headlines can't be used
                m["gsm8k"] = load(g)["results"]["gsm8k_cot_zeroshot"]["exact_match,flexible-extract"]
        out[name] = m
    return out


BASE_S = base_scores()
BPB = native("bpb", BASE)
EFF = native("efficiency", BASE)
XST = native("xstest", CHAT)
CHAT_S = chat_scores()
TEST = load(RUNS / "m4_t2d" / "test_results.json")
ROUTES = load(RUNS / "m4_t2d" / "routes.json")
CHOICES = load(RUNS / "m4_t2d" / "lr_choices.json")
ORDER = list(BASE.values())


def params(name: str) -> float:
    return EFF[name]["params"]["total"] / 1e6


def eff_row(name: str, batch: int, prompt: int = 512) -> dict:
    return next(r for r in EFF[name]["rows"] if r["batch"] == batch and r["prompt_tokens"] == prompt)


def seeds(model: str, key: str, group: str = "seeds") -> list[float]:
    return [s[key] for s in TEST["models"][model][group]]


# ---- text-to-data ----

def fig_t2d_scores() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    ms = list(T2D)
    y = list(range(len(ms)))[::-1]
    for yy, m in zip(y, ms):
        c = color(T2D[m])
        ax.scatter(seeds(m, "score"), [yy + 0.14] * 3, s=20, facecolor="none", edgecolor=c, linewidth=1, zorder=2)
        ax.scatter([statistics.fmean(seeds(m, "score"))], [yy + 0.14], s=78, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(seeds(m, "score", "zero_match_seeds"), [yy - 0.14] * 3, s=20, marker="D", facecolor="none", edgecolor=c, linewidth=1, zorder=2)
        ax.scatter([statistics.fmean(seeds(m, "score", "zero_match_seeds"))], [yy - 0.14], s=62, marker="D", color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.text(max(seeds(m, "score")) + 0.0025, yy + 0.14, f"{statistics.fmean(seeds(m, 'score')):.3f}", fontsize=8, color=INK, va="center")
    ax.set_yticks(y, [T2D[m] for m in ms])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-0.6, len(ms) - 0.3)
    ax.margins(x=0.08)
    ax.set_xlabel("text-to-data score (axis does not start at zero)")
    grid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=8, label="All test items, mean"), Line2D([], [], marker="D", color=MUTED, linestyle="", markersize=7, label="Zero-match items, mean"),
                       Line2D([], [], marker="o", markerfacecolor="none", color=MUTED, linestyle="", markersize=5, label="Each seed")], frameon=False, fontsize=8, loc="upper left")
    finish(fig, "t2d_scores", "text-to-data", "V2 scored lowest of the four, on every seed", "Mean of CUAD, MASSIVE and Banking77. Fine-tuned by one recipe, test scored once.", "runs/m4_t2d/test_results.json", 0.84)


def fig_t2d_gap() -> None:
    rivals = {"smollm2": "SmolLM2-135M", "gemma": "Gemma 3 270M", "qwen3": "Qwen3-0.6B"}
    metrics = [("score", "Score", "comparisons"), ("cuad", "CUAD", "comparisons"), ("massive", "MASSIVE", "comparisons"), ("banking77", "Banking77", "comparisons"), ("score", "Zero-match score", "zero_match_comparisons")]
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.2), sharey=True, sharex=True)
    y = list(range(len(metrics)))[::-1]
    for ax, (r, name) in zip(axes, rivals.items()):
        for yy, (k, _, group) in zip(y, metrics):
            c = TEST[group][r][k]
            ax.errorbar([c["diff"]], [yy], xerr=[2 * c["se"]], color=BLUE, capsize=2.5, linewidth=1.5, zorder=2)
            lost = c["verdict"] == "loss"
            ax.scatter([c["diff"]], [yy], s=52, color=BLUE if lost else SURFACE, edgecolor=BLUE, linewidth=1.5, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(f"V2 minus {name}")
        grid(ax)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [m[1] for m in metrics])
    axes[1].set_xlabel("difference, with 2 standard errors (0 means level)")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=7, label="Loss"), Line2D([], [], marker="o", markerfacecolor=SURFACE, color=BLUE, linestyle="", markersize=7, label="Tie")], frameon=False, fontsize=8, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 0.97))
    finish(fig, "t2d_gap", "text-to-data", "No win: every point sits left of zero", "A loss is more than 2 standard errors behind. Paired bootstrap over test items, 10,000 rounds.", "runs/m4_t2d/test_results.json", 0.84)


def fig_t2d_tasks() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)
    ms = list(T2D)
    y = list(range(len(ms)))[::-1]
    for ax, (k, title) in zip(axes, (("cuad", "CUAD, contract fields (F1)"), ("massive", "MASSIVE, request slots (F1)"), ("banking77", "Banking77, intents (accuracy)"))):
        for yy, m in zip(y, ms):
            c = color(T2D[m])
            ax.scatter(seeds(m, k), [yy] * 3, s=20, facecolor="none", edgecolor=c, linewidth=1, zorder=2)
            v = statistics.fmean(seeds(m, k))
            ax.scatter([v], [yy], s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{v:.3f}", (v, yy), textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center", color=INK)
        ax.set_title(title)
        ax.margins(x=0.18)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylim(-0.5, len(ms) - 0.3)
        grid(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [T2D[m] for m in ms])
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    finish(fig, "t2d_tasks", "text-to-data", "Close on short text, furthest behind on contracts", "Mean and each of 3 seeds. Each axis is zoomed separately and does not start at zero.", "runs/m4_t2d/test_results.json", 0.83)


def fig_t2d_cuad() -> None:
    fields = [("document_name", "Document name"), ("parties", "Parties"), ("agreement_date", "Agreement date"), ("governing_law", "Governing law")]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 3.9))
    ms = list(T2D)
    for ax, (k, title) in zip(axes.flat, fields):
        hbars(ax, sorted(((SHORT[m], statistics.fmean(s["cuad_per_field"][k] for s in TEST["models"][m]["seeds"])) for m in ms), key=lambda x: -x[1]), title=f"{title} (F1)", xmax=1.2)
    hbars(axes.flat[4], sorted(((SHORT[m], statistics.fmean(seeds(m, "cuad_valid_json"))) for m in ms), key=lambda x: -x[1]), title="CUAD: valid JSON", xmax=1.3)
    hbars(axes.flat[5], sorted(((SHORT[m], statistics.fmean(seeds(m, "massive_valid_json"))) for m in ms), key=lambda x: -x[1]), fmt="{:.4f}", title="MASSIVE: valid JSON", xmax=1.3)
    finish(fig, "t2d_cuad_fields", "text-to-data", "Parties are the hardest contract field for all four models", "Mean of 3 seeds, 98 contracts.", "runs/m4_t2d/test_results.json", 0.88)


def fig_t2d_cpu() -> None:
    files = {m: load(RUNS / "m4_t2d" / f"cpu-{m}.json") for m in T2D}
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 3.8))
    for ax, (k, t) in zip(axes[0], (("cuad", "CUAD, docs per minute"), ("massive", "MASSIVE, docs per minute"), ("banking77", "Banking77, docs per minute"))):
        hbars(ax, sorted(((SHORT[m], files[m]["tasks"][k]["documents_per_minute"]) for m in T2D), key=lambda x: -x[1]), "{:.1f}", t)
    for ax, (k, t) in zip(axes[1], (("cuad", "CUAD, first token, s"), ("massive", "MASSIVE, first token, s"), ("banking77", "Banking77, first token, s"))):
        hbars(ax, sorted(((SHORT[m], files[m]["tasks"][k]["time_to_first_token_s"]) for m in T2D), key=lambda x: x[1]), "{:.2f}", t)
    ram = ", ".join(f"{SHORT[m]} {files[m]['peak_ram_gib']:.2f}" for m in sorted(T2D, key=lambda m: files[m]["peak_ram_gib"]))
    finish(fig, "t2d_cpu", "text-to-data", "On CPU V2 is mid-pack: light, but not the fastest", f"8 threads, one run, small samples. Peak RAM, GiB: {ram}.", "runs/m4_t2d/cpu-*.json", 0.88)


def fig_t2d_size() -> None:
    files = {m: load(RUNS / "m4_t2d" / f"cpu-{m}.json") for m in T2D}
    sizes = {"qwen3": "Qwen3-0.6B", "gemma": "Gemma 3 270M", "smollm2": "SmolLM2-135M", "v2": V2}
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    score = {m: statistics.fmean(seeds(m, "score")) for m in T2D}
    off = {"SmolLM2-135M": (-30, 8), "Gemma 3 270M": (6, -11), "V2": (6, -11)}
    scatter_labels(axes[0], [(T2D[m], params(sizes[m]), score[m]) for m in T2D], off)
    axes[0].set_xlabel("parameters, millions")
    axes[0].set_ylabel("text-to-data score")
    axes[0].margins(x=0.22, y=0.14)
    scatter_labels(axes[1], [(T2D[m], files[m]["peak_ram_gib"], score[m]) for m in T2D], {"SmolLM2-135M": (-18, 8), "V2": (6, -11), "Gemma 3 270M": (6, -11)})
    axes[1].set_xlabel("peak RAM on CPU, GiB")
    axes[1].margins(x=0.22, y=0.14)
    finish(fig, "t2d_size", "text-to-data", "More size bought more score, and V2 sits below the trend", "SmolLM2-135M is smaller, uses about the same RAM, and scores higher.", "runs/m4_t2d/test_results.json, cpu-*.json", 0.84)


def fig_t2d_dev_test() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ms = list(T2D)
    y = list(range(len(ms)))[::-1]
    for yy, m in zip(y, ms):
        c = color(T2D[m])
        dev, test = ROUTES[m]["chosen"]["mean"], statistics.fmean(seeds(m, "score"))
        ax.plot([dev, test], [yy, yy], color=c, linewidth=2, zorder=1)
        ax.scatter([dev], [yy], s=60, facecolor=SURFACE, edgecolor=c, linewidth=1.5, zorder=2)
        ax.scatter([test], [yy], s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(y, [T2D[m] for m in ms])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("score (axis does not start at zero)")
    ax.set_ylim(-0.5, len(ms) - 0.5)
    grid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", markerfacecolor=SURFACE, color=MUTED, linestyle="", markersize=7, label="Dev, chosen route"), Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=8, label="Test")], frameon=False, fontsize=8, loc="lower right")
    finish(fig, "t2d_dev_vs_test", "text-to-data", "Dev scores predicted the order, and test confirmed it", "Routes and learning rates were chosen on dev only, before test was touched.", "runs/m4_t2d/routes.json, test_results.json", 0.8)


def fig_t2d_lr() -> None:
    models = {"V2": ("v2-base", "v2-chat"), "SmolLM2-135M": ("smollm2-base", "smollm2-chat"), "Gemma 3 270M": ("gemma-base", "gemma-chat"), "Qwen3-0.6B": ("qwen3-base", None)}
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.9), sharey=True)
    for ax, (name, (base, chat)) in zip(axes, models.items()):
        for arm, style, lab in ((base, "-", "From base"), (chat, "--", "From chat")):
            if not arm:
                continue
            rows = CHOICES[arm]["rows"]
            ax.plot([r["lr"] for r in rows], [r["score"] for r in rows], color=color(name) if name == "V2" else INK2, linestyle=style, marker="o", markersize=4, linewidth=1.8, label=lab)
            pick = next(r for r in rows if r["lr"] == CHOICES[arm]["chosen_lr"])
            ax.scatter([pick["lr"]], [pick["score"]], s=70, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1e-5, 3e-5, 1e-4], ["1e-5", "3e-5", "1e-4"])
        ax.minorticks_off()
        ax.set_title(name)
        grid(ax, "both")
    axes[0].set_ylabel("dev score, seed 1")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    finish(fig, "t2d_lr_grid", "text-to-data", "V2 and SmolLM2 chose the top of the grid, Gemma and Qwen3 the bottom", "Each could have scored more just past the fixed grid. The ring marks the choice.", "runs/m4_t2d/lr_choices.json", 0.84)


def fig_t2d_loss() -> None:
    models = {"V2": "v2", "SmolLM2-135M": "smollm2", "Gemma 3 270M": "gemma", "Qwen3-0.6B": "qwen3"}
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.5), sharey=True)
    for ax, (name, m) in zip(axes, models.items()):
        c = ROUTES[m]["chosen"]
        for seed in (1, 2, 3):
            d = next(p for p in sorted(RUNS.glob(f"*-train-m4-t2d-{c['arm']}-lr{c['lr']:g}-s{seed}-*")))
            rows = metrics(d / "metrics.jsonl")
            ax.plot([r["step"] / r["total_steps"] * 100 for r in rows], [r["loss"] for r in rows], color=color(name) if name == "V2" else GRAY, linewidth=1.1, alpha=0.75)
        ax.set_title(f"{name}\nfrom {c['arm'].split('-')[1]}, LR {c['lr']:g}")
        ax.set_ylim(0, 2)
        ax.set_xlabel("% of run")
        grid(ax, "both")
    axes[0].set_ylabel("training loss")
    finish(fig, "t2d_finetune_loss", "text-to-data", "Fine-tuning runs were stable for every model", "Chosen route, all 3 seeds. Loss is on each model's own tokens, so levels aren't comparable across models.", "runs/*-train-m4-t2d-*/metrics.jsonl", 0.84)


def fig_billsum() -> None:
    names = SHORT
    b = {m: load(RUNS / "m4_t2d" / f"billsum-{m}.json") for m in names}
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.2))
    hbars(axes[0], sorted(((names[m], b[m]["rouge_l"]) for m in names), key=lambda x: -x[1]), title="ROUGE-L, all items", xmax=0.65)
    hbars(axes[1], sorted(((names[m], b[m]["rouge_l_zero_match"]) for m in names), key=lambda x: -x[1]), title="ROUGE-L, zero-match items", xmax=0.65)
    hbars(axes[2], sorted(((names[m], b[m]["generation"]["seconds"]) for m in names), key=lambda x: x[1]), "{:.0f}", "Generation, seconds")
    finish(fig, "t2d_billsum", "text-to-data", "BillSum: V2 wrote its summaries fastest, and they scored lowest", "Reported only, not part of the score. Chosen route, seed 1, 1,000 items, on the A100.", "runs/m4_t2d/billsum-*.json", 0.78)


# ---- base models ----

def fig_base_mean() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    hbars(ax, sorted(((n, BASE_S[n]["mean"]) for n in ORDER), key=lambda x: -x[1]), xmax=0.62)
    ax.set_xlabel("mean of 10 zero-shot tasks (higher is better)")
    finish(fig, "base_zero_shot_mean", "base", "As a base model V2 ranks 6th of 11", "Ahead of V1, behind SmolLM2-135M and Gemma 3 270M, the two nearest its size.", "runs/*-eval-base-*/eval/summary.json", 0.88)


def fig_base_tasks() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    tasks = list(ZERO_SHOT)
    y = list(range(len(tasks)))[::-1]
    for yy, t in zip(y, tasks):
        chance = ZERO_SHOT[t][1]
        if chance:
            ax.plot([chance, chance], [yy - 0.3, yy + 0.3], color=INK2, linewidth=1.2, zorder=1)
        for n in ORDER:
            if n not in (V2, V1):
                ax.scatter([BASE_S[n][t]], [yy], s=22, color=GRAY, alpha=0.65, edgecolor=SURFACE, linewidth=0.8, zorder=2)
        ax.scatter([BASE_S[V1][t]], [yy], s=52, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.scatter([BASE_S[V2][t]], [yy], s=66, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.set_yticks(y, [ZERO_SHOT[t][0] for t in tasks])
    ax.set_xlabel("score (higher is better)")
    grid(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=8, label=V2), Line2D([], [], marker="o", color=ORANGE, linestyle="", markersize=7, label=V1), Line2D([], [], marker="o", color=GRAY, linestyle="", markersize=5, label="9 other models"), Line2D([], [], marker="|", color=INK2, linestyle="", markersize=9, label="About chance")], frameon=False, fontsize=8, loc="lower right")
    finish(fig, "base_zero_shot_tasks", "base", "V2 is above chance on some tasks and close to it on others", "Each dot is one model. Chance lines are approximate. TruthfulQA has no single chance level.", "runs/*-eval-base-*/eval/summary.json", 0.88)


def fig_base_few() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.5))
    for ax, (k, t) in zip(axes, FEW.items()):
        hbars(ax, sorted(((n, BASE_S[n][k]) for n in ORDER), key=lambda x: -x[1]), title=t)
        ax.tick_params(axis="y", labelsize=6.8)
    finish(fig, "base_few_shot_code", "base", "Math, code and fact recall are where V2 is weakest", "Higher is better. TriviaQA and GSM8K few-shot; HumanEval and MBPP pass rate.", "runs/*-eval-base-*/eval/summary.json", 0.87)


def fig_base_size() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    off = {"Pythia-410M": (6, -10), "GPT-2 Medium": (-4, -13), V1: (6, 4), "GPT-2": (-30, -12), "Pythia-160M": (6, -9), "SmolLM2-135M": (6, 4), "Gemma 3 270M": (6, 3), V2: (-22, -14)}
    scatter_labels(ax, [(n, params(n), BASE_S[n]["mean"]) for n in ORDER], off)
    ax.set_xscale("log")
    ax.set_xticks([125, 200, 300, 400, 600], ["125", "200", "300", "400", "600"])
    ax.minorticks_off()
    ax.set_xlabel("parameters, millions (log scale)")
    ax.set_ylabel("zero-shot mean")
    ax.margins(x=0.1, y=0.1)
    finish(fig, "base_size_vs_quality", "base", "V2 sits below the line modern small models set", "SmolLM2-135M beats it with fewer parameters.", "runs/*-eval-base-*, runs/*-eval-efficiency-*", 0.88)


def fig_base_bpb() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.5))
    for ax, (k, t) in zip(axes, (("web", "Web"), ("synthetic", "Synthetic"), ("code", "Code"), ("math", "Math"))):
        hbars(ax, sorted(((n, BPB[n]["domains"][k]["bpb"]) for n in ORDER), key=lambda x: x[1]), title=t)
        ax.tick_params(axis="y", labelsize=6.8)
    finish(fig, "base_bits_per_byte", "base", "V2 predicts code text well for its size, and web text less well", "Bits per byte, lower is better, best on top. Chat is left out: V2's chat text overlaps its training mix.", "runs/*-eval-bpb-*/eval/bpb.json", 0.87)


def fig_base_speed() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.2))
    a = axes.flat
    hbars(a[0], sorted(((n, eff_row(n, 1)["decode_tok_s"]) for n in ORDER), key=lambda x: -x[1]), "{:.0f}", "Decode tok/s, batch 1")
    hbars(a[1], sorted(((n, eff_row(n, 8)["decode_tok_s"]) for n in ORDER), key=lambda x: -x[1]), "{:.0f}", "Decode tok/s, batch 8")
    hbars(a[2], sorted(((n, eff_row(n, 1)["prefill_tok_s"]) for n in ORDER), key=lambda x: -x[1]), "{:,.0f}", "Prefill tok/s")
    hbars(a[3], sorted(((n, eff_row(n, 1)["ttft_s"] * 1000) for n in ORDER), key=lambda x: x[1]), "{:.1f}", "First token, ms")
    hbars(a[4], sorted(((n, eff_row(n, 8)["peak_vram_GiB"]) for n in ORDER), key=lambda x: x[1]), "{:.2f}", "Peak VRAM, GiB, batch 8")
    hbars(a[5], sorted(((n, eff_row(n, 1)["decode_bytes_s"]) for n in ORDER), key=lambda x: -x[1]), "{:,.0f}", "Decode bytes/s, batch 1")
    for ax in a:
        ax.tick_params(axis="y", labelsize=7)
    finish(fig, "base_speed", "base", "V2 uses the least GPU memory, and older, shallower models decode faster", "Laptop GPU, prompt 512, 128 new tokens. V2 uses its own engine, the others HF generate: not like for like.", "runs/*-eval-efficiency-*/eval/efficiency.json", 0.92)


def fig_base_speed_quality() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    off = {"GPT-2": (-6, -13), V1: (-46, 6), "Pythia-160M": (6, 3), "Qwen3-0.6B": (6, 3), "SmolLM2-135M": (6, 2), "Gemma 3 270M": (6, -10), "GPT-2 Medium": (6, -10), "Pythia-410M": (6, 5), V2: (-48, -13), "SmolLM2-360M": (6, 3), "Qwen2.5-0.5B": (6, 3)}
    scatter_labels(ax, [(n, eff_row(n, 1)["decode_tok_s"], BASE_S[n]["mean"]) for n in ORDER], off)
    ax.set_xlabel("decode tok/s on the laptop GPU, batch 1")
    ax.set_ylabel("zero-shot mean")
    ax.margins(x=0.14, y=0.1)
    finish(fig, "base_speed_vs_quality", "base", "No model here is both fastest and best", "V2 is the fastest of the modern models and sits mid-table on quality. Engines differ.", "runs/*-eval-efficiency-*, runs/*-eval-base-*", 0.88)


CPU_MODELS = {"v2-litegpt": V2, "v2-hf": "V2 (Transformers)", "smollm2": "SmolLM2-135M", "gemma": "Gemma 3 270M", "v1": V1}


def cpu(model: str, dtype: str) -> dict:
    return load(runs("bench", f"cpu-{model}-{dtype}")[-1] / "bench" / "cpu.json")


def cpu_row(r: dict, threads: int, prompt: int = 128) -> dict:
    return next(x for x in r["rows"] if x["threads"] == threads and x["prompt_tokens"] == prompt)


def fig_cpu_threads() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    style = {V2: (BLUE, "-"), "V2 (Transformers)": (BLUE, ":"), V1: (ORANGE, "-"), "SmolLM2-135M": (GRAY, "-"), "Gemma 3 270M": (GRAY, "--")}
    for ax, dtype, title in zip(axes, ("float32", "bfloat16"), ("float32", "bfloat16 (no native bf16 maths on this CPU)")):
        for m, n in CPU_MODELS.items():
            rows = sorted((x for x in cpu(m, dtype)["rows"] if x["prompt_tokens"] == 128), key=lambda x: x["threads"])
            c, ls = style[n]
            ax.plot([x["threads"] for x in rows], [x["decode_tok_s"] for x in rows], color=c, linestyle=ls, marker="o", markersize=3.5, label=n)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16], ["1", "4", "8", "16"])
        ax.set_xlabel("threads")
        ax.set_title(title)
        grid(ax, "both")
    axes[0].set_ylabel("decode tok/s, batch 1")
    axes[0].set_ylim(bottom=0)
    axes[1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    finish(fig, "cpu_threads", "cpu", "On CPU, 4 to 8 threads is the sweet spot for decode",
           "Laptop CPU (i9-14900HX), prompt 128, 64 new tokens, median of 3. Past 8 threads the slower E-cores join.", "runs/*-bench-cpu-*/bench/cpu.json", 0.86)


def fig_cpu_ram_speed() -> None:
    names = {m: n for m, n in CPU_MODELS.items() if m != "v2-hf"}
    f32 = {m: cpu(m, "float32") for m in names}
    pi = load(runs("bench", "cpu-pi-estimate")[-1] / "bench" / "pi_estimate.json")["models"]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.4))
    a = axes.flat
    hbars(a[0], sorted(((names[m], f32[m]["peak_rss_gib"]) for m in names), key=lambda x: x[1]), "{:.2f}", "Peak RAM, GiB (whole process)")
    hbars(a[1], sorted(((names[m], cpu_row(f32[m], 4)["decode_tok_s"]) for m in names), key=lambda x: -x[1]), "{:.1f}", "Decode tok/s, 4 threads")
    hbars(a[2], sorted(((names[m], cpu_row(f32[m], 4)["prefill_tok_s"]) for m in names), key=lambda x: -x[1]), "{:,.0f}", "Prefill tok/s, 4 threads")
    hbars(a[3], sorted(((names[m], pi[f"{m}/float32"]["estimate"]["Raspberry Pi 5"]["prompts"]["128"]["decode_tok_s"]) for m in names), key=lambda x: -x[1]),
          "{:.1f}", "Raspberry Pi 5, decode tok/s (estimate)")
    for ax in a:
        ax.tick_params(axis="y", labelsize=7.5)
    finish(fig, "cpu_ram_speed", "cpu", "On CPU, V2 needs under 1.7 GiB and keeps pace with models of its size",
           "float32, prompt 128. The Pi 5 figure is an estimate from laptop runs and published Pi bandwidth, not a measurement.", "runs/*-bench-cpu-*/bench/", 0.88)


GGUF_V2 = (("V2 PyTorch fp32", None), ("V2 GGUF BF16", "BF16"), ("V2 GGUF Q8_0", "Q8_0"), ("V2 GGUF Q4_K_M", "Q4_K_M"))
ENTITY.update({n: BLUE for n, _ in GGUF_V2} | {"V2 Q8_0": BLUE, "V2 Q4_K_M": BLUE})


def gguf(model: str, quant: str) -> dict:
    return load(runs("bench", f"cpu-gguf-{model}-{quant}")[-1] / "bench" / "cpu.json")


def fig_gguf_ram_speed() -> None:
    recs = {n: (cpu("v2-litegpt", "float32") if q is None else gguf("v2", q)) for n, q in GGUF_V2}
    names = [n for n, _ in GGUF_V2]
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.4))
    a = axes.flat
    hbars(a[0], [(n, recs[n]["weight_bytes"] / 1e6) for n in names], "{:,.0f}", "Weights, MB", label_names=names)
    hbars(a[1], [(n, recs[n]["peak_rss_gib"]) for n in names], "{:.2f}", "Peak RAM, GiB (whole process)", label_names=names)
    hbars(a[2], [(n, cpu_row(recs[n], 4)["decode_tok_s"]) for n in names], "{:.0f}", "Decode tok/s, 4 threads", label_names=names)
    hbars(a[3], [(n, cpu_row(recs[n], 4)["prefill_tok_s"]) for n in names], "{:,.0f}", "Prefill tok/s, 4 threads", label_names=names)
    for ax in a:
        ax.tick_params(axis="y", labelsize=7.5)
    finish(fig, "gguf_ram_speed", "cpu", "On llama.cpp, V2 needs a fraction of the RAM and decodes several times faster",
           "Laptop CPU, prompt 128, 64 new tokens, median of 3. PyTorch fp32 through LiteGPT's engine against the GGUF files on llama.cpp.",
           "runs/*-bench-cpu-gguf-v2-*, runs/*-bench-cpu-v2-litegpt-float32-*", 0.88)


def fig_gguf_rivals() -> None:
    labels = {"v2": "V2", "smollm2": "SmolLM2-135M", "gemma": "Gemma 3 270M"}
    pi = load(runs("bench", "cpu-gguf-pi-estimate")[-1] / "bench" / "pi_estimate.json")["models"]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.4))
    for row, q in enumerate(("Q8_0", "Q4_K_M")):
        recs = {f"{labels[m]} {q}": gguf(m, q) for m in labels}
        keys = {f"{labels[m]} {q}": f"{m}/{q}" for m in labels}
        hbars(axes[row, 0], sorted(((n, cpu_row(r, 4)["decode_tok_s"]) for n, r in recs.items()), key=lambda x: -x[1]), "{:.0f}",
              f"{q}: decode tok/s, 4 threads", label_names=tuple(recs))
        hbars(axes[row, 1], sorted(((n, r["weight_bytes"] / 1e6) for n, r in recs.items()), key=lambda x: x[1]), "{:,.0f}",
              f"{q}: file, MB", label_names=tuple(recs))
        hbars(axes[row, 2], sorted(((n, pi[keys[n]]["estimate"]["Raspberry Pi 5"]["prompts"]["128"]["decode_tok_s"]) for n in recs), key=lambda x: -x[1]),
              "{:.0f}", f"{q}: Pi 5 decode (estimate)", label_names=tuple(recs))
    for ax in axes.flat:
        ax.tick_params(axis="y", labelsize=7)
    finish(fig, "gguf_rivals", "cpu", "On llama.cpp, V2 sits between SmolLM2-135M and Gemma 3 270M",
           "All three converted with the same llama.cpp build. Laptop CPU, prompt 128. Pi 5 figures are estimates, not measurements.",
           "runs/*-bench-cpu-gguf-*", 0.88)


def fig_v1_v2() -> None:
    tasks = [(k, v[0]) for k, v in ZERO_SHOT.items()] + list(FEW.items())
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
    y = list(range(len(tasks)))[::-1]
    ax = axes[0]
    for yy, (k, _) in zip(y, tasks):
        a, b = BASE_S[V1][k], BASE_S[V2][k]
        ax.plot([a, b], [yy, yy], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter([a], [yy], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.scatter([b], [yy], s=56, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_yticks(y, [t for _, t in tasks])
    ax.set_title("Benchmark scores (higher is better)")
    ax.tick_params(axis="y", length=0)
    grid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", color=ORANGE, linestyle="", markersize=7, label=V1), Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=8, label=V2)], frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    doms = [("web", "Web"), ("synthetic", "Synthetic"), ("code", "Code"), ("math", "Math")]
    yb = list(range(len(doms)))[::-1]
    for yy, (k, _) in zip(yb, doms):
        a, b = BPB[V1]["domains"][k]["bpb"], BPB[V2]["domains"][k]["bpb"]
        ax.plot([a, b], [yy, yy], color=GRID, linewidth=2.5, zorder=1)
        ax.scatter([a], [yy], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=2)
        ax.scatter([b], [yy], s=56, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.annotate(f"{b:.3f}", (b, yy), textcoords="offset points", xytext=(-3, 8), fontsize=7.5, ha="right", color=INK)
        ax.annotate(f"{a:.3f}", (a, yy), textcoords="offset points", xytext=(3, 8), fontsize=7.5, ha="left", color=INK)
    ax.set_yticks(yb, [t for _, t in doms])
    ax.set_ylim(-0.6, len(doms) - 0.3)
    ax.set_title("Bits per byte (lower is better)")
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.1)
    grid(ax)
    finish(fig, "v1_vs_v2", "base", "V2 against V1: much better on code text, smaller gains elsewhere", "V2 is also 1.6 times larger, on different data. This shows the whole model changing, not code alone.", "runs/*-eval-base-*, runs/*-eval-bpb-*", 0.89)


def fig_checkpoints() -> None:
    names = {"2.5B": "tokens-2.5B", "5B": "tokens-5B", "9B": "tokens-9B", "10B": "final"}
    bs = {t: load(next(reversed(runs("eval-base", f"litegpt-{n}"))) / "eval" / "summary.json")["headline"] for t, n in names.items()}
    bp = {t: load(runs("eval-bpb", f"litegpt-{n}")[-1] / "eval" / "bpb.json")["domains"] for t, n in names.items()}
    xs = list(names)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    axes[0].plot(xs, [statistics.fmean(bs[t][k]["value"] for k in ZERO_SHOT) for t in xs], color=BLUE, marker="o")
    axes[0].set_title("Zero-shot mean")
    axes[1].plot(xs, [bs[t]["triviaqa"]["value"] for t in xs], color=BLUE, marker="o")
    axes[1].set_title("TriviaQA")
    for (k, c) in (("web", BLUE), ("synthetic", ORANGE), ("code", AQUA), ("math", YELLOW)):
        vals = [bp[t][k]["bpb"] for t in xs]
        axes[2].plot(xs, vals, color=c, marker="o", markersize=4, label=k.capitalize())
    axes[2].set_title("Bits per byte (lower is better)")
    axes[2].legend(frameon=False, fontsize=7, loc="center", ncol=2)
    for ax in axes:
        grid(ax, "both")
        ax.set_xlabel("tokens trained")
    finish(fig, "m3_checkpoints", "base", "The last billion tokens moved the most", "Learning-rate decay, data mix and context all changed there, so one run can't separate the causes.", "runs/*-eval-base-litegpt-*, runs/*-eval-bpb-litegpt-*", 0.83)


# ---- chat ----

def fig_chat() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.6))
    for ax, (k, t) in zip(axes.flat, (("ifeval", "IFEval"), ("gsm8k", "GSM8K chat"), ("humaneval", "HumanEval chat"), ("mbpp", "MBPP chat"))):
        hbars(ax, sorted(((n, CHAT_S[n][k]) for n in CHAT.values()), key=lambda x: -x[1]), title=t)
    finish(fig, "chat_tasks", "chat", "V2-Instruct beats V1-Instruct and trails the rest on instructions", "Higher is better. IFEval is the registered gate for chat.", "runs/*-eval-instruct-*/eval/summary.json", 0.89)


def fig_xstest() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.6))
    hbars(axes[0], sorted(((n, XST[n]["unsafe_refusal_rate"] * 100) for n in CHAT.values()), key=lambda x: -x[1]), "{:.1f}%", "Unsafe prompts refused (higher is safer)", xmax=115)
    hbars(axes[1], sorted(((n, XST[n]["safe_over_refusal_rate"] * 100) for n in CHAT.values()), key=lambda x: x[1]), "{:.1f}%", "Safe prompts refused (lower is better)", xmax=65)
    finish(fig, "chat_xstest", "chat", "V2-Instruct refuses nothing, so it has no safety behaviour", "XSTest, 200 unsafe and 250 safe prompts. Refusals are detected by string match.", "runs/*-eval-xstest-*/eval/xstest.json", 0.8)


# ---- training ----

M3 = next(p for p in runs("train", "m3") if p.name.endswith("673cd85"))


def rolling(values: list[float], n: int) -> list[float]:
    return [statistics.median(values[max(0, i - n):i + 1]) for i in range(len(values))]


def phases(ax) -> None:
    ax.axvline(9.0, color=INK2, linewidth=1, linestyle=":")
    ax.axvline(9.6, color=INK2, linewidth=1, linestyle=":")


def fig_m3_loss() -> None:
    rows = metrics(M3 / "metrics.jsonl")
    x = [r["tokens"] / 1e9 for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.plot(x, [r["loss"] for r in rows], color=GRAY, linewidth=0.5, alpha=0.5, label="Each log")
    ax.plot(x, rolling([r["loss"] for r in rows], 25), color=BLUE, linewidth=2, label="Rolling median")
    phases(ax)
    ax.set_ylim(2, 5)
    ax.text(9.0, 4.75, "decay ", ha="right", fontsize=8, color=INK2)
    ax.text(9.62, 4.75, "4,096", ha="left", fontsize=8, color=INK2)
    ax.set_xlabel("tokens trained, billions")
    ax.set_ylabel("training loss")
    grid(ax, "both")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    finish(fig, "m3_loss", "training", "Pretraining ran stably for 37 hours, with no restarts or skipped steps", "Dotted lines: decay and new data mix from 9.0B, context 4,096 from 9.6B. The first steps are above the axis.", M3.name + "/metrics.jsonl", 0.86)


def fig_m3_signals() -> None:
    rows = metrics(M3 / "metrics.jsonl")
    x = [r["tokens"] / 1e9 for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.6), sharex=True)
    axes[0].plot(x, [r["lr"] for r in rows], color=BLUE, linewidth=1.6)
    axes[0].set_title("Learning rate")
    axes[1].plot(x, rolling([r["grad_norm"] for r in rows], 25), color=BLUE, linewidth=1.4)
    axes[1].set_title("Gradient norm, rolling median")
    axes[1].set_ylim(0, 0.3)
    axes[2].plot(x, rolling([r["tok_per_s"] / 1000 for r in rows], 15), color=BLUE, linewidth=1.4)
    axes[2].set_title("Speed, thousand tokens per second, rolling median")
    axes[2].set_ylim(0, 95)
    axes[2].set_xlabel("tokens trained, billions")
    for ax in axes:
        grid(ax, "both")
        phases(ax)
    finish(fig, "m3_signals", "training", "Learning rate, gradients and speed across pretraining", "Speed drops at 9.6B, when the context doubles. The early gradient norms are above the axis.", M3.name + "/metrics.jsonl", 0.91)


def fig_m3_bpb() -> None:
    rows = jl(M3 / "eval.jsonl")
    x = [r["tokens"] / 1e9 for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for k, c, lab in (("web", BLUE, "Web"), ("math", ORANGE, "Math"), ("code", AQUA, "Code"), ("synthetic", YELLOW, "Synthetic"), ("chat", MAGENTA, "Chat")):
        ys = [r[f"text_{k}"]["bpb"] for r in rows]
        ax.plot(x, ys, color=c, label=lab, linewidth=2)
        ax.annotate(f"{lab} {ys[-1]:.3f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(5, {"math": 5, "web": -5, "code": 5, "synthetic": -6, "chat": 0}[k]), fontsize=7.5, va="center", color=INK)
    ax.axvline(9.0, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlim(0, 11.6)
    ax.set_xlabel("tokens trained, billions")
    ax.set_ylabel("bits per byte (lower is better)")
    grid(ax, "both")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center")
    finish(fig, "m3_bits_per_byte", "training", "Every domain improved, most of all in the final billion tokens", "Chat's drop partly reflects overlap with the new data mix, so it isn't a fair measure of chat ability.", M3.name + "/eval.jsonl", 0.87)


def fig_m3_gate() -> None:
    g = load(M3 / "gate" / "extension_gate.json")
    rows = [r for r in jl(M3 / "eval.jsonl") if 5.5e9 <= r["tokens"] <= 9.05e9]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.plot([r["tokens"] / 1e9 for r in rows], [r["val"]["bpb"] for r in rows], color=GRAY, marker="o", markersize=3.5, linewidth=1.2, zorder=1)
    for w, c, lab in ((g["inputs"]["window_a"], ORANGE, "Window A"), (g["inputs"]["window_b"], BLUE, "Window B")):
        xs, ys = [float(k) / 1e9 for k in w], list(w.values())
        ax.scatter(xs, ys, s=52, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=f"{lab}, mean {statistics.fmean(ys):.4f}")
        ax.plot([min(xs), max(xs)], [statistics.fmean(ys)] * 2, color=c, linewidth=2, zorder=2)
    ax.set_xlabel("tokens trained, billions")
    ax.set_ylabel("validation bits per byte")
    grid(ax, "both")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "m3_gate", "training", f"The registered gate said stop: {g['delta']:.5f} gained, {g['threshold_bpb']:.3f} needed", "Extending past 10B needed the window means to differ by the registered threshold. Decided by script.", M3.name + "/gate/extension_gate.json", 0.82)


def fig_m3_lr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=False)
    labels = {"m3lr-6e-4": "6e-4", "m3lr-1.2e-3": "1.2e-3", "m3lr-2.4e-3": "2.4e-3", "m3lr-0.0048": "4.8e-3"}
    finals = []
    for (n, lab), c in zip(labels.items(), (ORANGE, AQUA, BLUE, YELLOW)):
        rows = jl(runs("train", n)[-1] / "eval.jsonl")
        axes[0].plot([r["tokens"] / 1e6 for r in rows], [r["val"]["bpb"] for r in rows], color=c, marker="o", markersize=4, label=lab)
        finals.append((lab, rows[-1]["val"]["bpb"]))
    axes[0].set_xlabel("tokens, millions")
    axes[0].set_ylabel("validation bits per byte")
    axes[0].legend(frameon=False, fontsize=8)
    grid(axes[0], "both")
    axes[0].set_title("Real V2, four peak learning rates")
    hbars(axes[1], [("V2" if v == min(f[1] for f in finals) else lab, v) for lab, v in finals], title="Final (lower is better); V2's choice in blue", xmax=2.1)
    axes[1].set_yticks(axes[1].get_yticks(), [lab for lab, _ in finals])
    finish(fig, "m3_lr_sweep", "training", "Peak learning rate 2.4e-3 was chosen from a short sweep", "200M tokens each, with a short warm-up. Chosen before pretraining began.", "runs/*-train-m3lr-*/eval.jsonl", 0.83)


LADDER = [("s2", "+ RoPE"), ("s3", "+ RMSNorm, QK-norm"), ("s4", "+ SwiGLU"), ("s5", "+ grouped-query attention"), ("s6", "+ deeper, thinner"), ("s7", "+ 32k tokenizer"), ("s8a", "+ full data mix")]


def ladder_final(name: str) -> dict:
    return jl(runs("train", f"ladder-{name}")[-1] / "eval.jsonl")[-1]


def fig_ladder() -> None:
    s1 = [ladder_final("s1-seed1"), ladder_final("s1-seed3")]
    rows = [("V1 baseline", {k: statistics.fmean(s[f"text_{k}"]["bpb"] for s in s1) for k in ("web", "code", "math", "synthetic")})]
    rows += [(lab, {k: ladder_final(n)[f"text_{k}"]["bpb"] for k in ("web", "code", "math", "synthetic")}) for n, lab in LADDER]
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 3.3), sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, (k, t) in zip(axes, (("web", "Web"), ("synthetic", "Synthetic"), ("code", "Code"), ("math", "Math"))):
        vals = [r[1][k] for r in rows]
        ax.plot(vals, y, color=BLUE, linewidth=1.6, marker="o", markersize=5)
        for v, yy in zip((vals[0], vals[-1]), (y[0], y[-1])):
            ax.annotate(f"{v:.2f}", (v, yy), textcoords="offset points", xytext=(0, 7), fontsize=7.5, ha="center", color=INK)
        ax.set_title(t)
        ax.margins(x=0.18, y=0.1)
        grid(ax, "both")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [r[0] for r in rows])
    finish(fig, "ladder_steps", "training", "Architecture steps gained a little; the data mix transformed code", "One change at a time on a small proxy model, 1B tokens each. Bits per byte, lower is better.", "runs/*-train-ladder-*/eval.jsonl", 0.84)


def fig_ladder_seeds() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    for n, c, lab in (("s1-seed1", BLUE, "Seed 1"), ("s1-seed2", ORANGE, "Seed 2"), ("s1-seed3", AQUA, "Seed 3")):
        d = runs("train", f"ladder-{n}")[-1]
        rows = metrics(d / "metrics.jsonl")
        axes[0].plot([r["tokens"] / 1e9 for r in rows], rolling([r["loss"] for r in rows], 15), color=c, label=lab, linewidth=1.3)
        e = jl(d / "eval.jsonl")
        axes[1].plot([r["tokens"] / 1e9 for r in e], [r["text_web"]["bpb"] for r in e], color=c, marker="o", markersize=4, label=lab)
    axes[0].set_title("Training loss, rolling median")
    axes[0].set_ylim(3, 6)
    axes[1].set_title("Web bits per byte")
    for ax in axes:
        ax.set_xlabel("tokens, billions")
        grid(ax, "both")
    axes[0].legend(frameon=False, fontsize=8)
    finish(fig, "ladder_seed_divergence", "training", "One baseline seed diverged late, so a stability rule was added", "Taken as noise, seed 2 would have set a threshold no step could pass. Recorded before use.", "runs/*-train-ladder-s1-*", 0.83)


def fig_proxy_lr() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    for n, c, lab in (("lrsweep-1e-3", ORANGE, "1e-3"), ("lrsweep-2e-3", BLUE, "2e-3, chosen"), ("lrsweep-4e-3", AQUA, "4e-3, diverged")):
        e = jl(runs("train", f"ladder-{n}")[-1] / "eval.jsonl")
        ax.plot([r["tokens"] / 1e6 for r in e], [r["text_web"]["bpb"] for r in e], color=c, marker="o", markersize=5, label=lab)
    ax.set_xlabel("tokens, millions")
    ax.set_ylabel("web bits per byte")
    grid(ax, "both")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "ladder_lr_sweep", "training", "The ladder's learning rate came from a small registered sweep", "The proxy model, 0.25B tokens each. The lowest web bits per byte won.", "runs/*-train-ladder-lrsweep-*", 0.82)


SFT = next(p for p in sorted(RUNS.glob("*-train-m4-sft-0034095")))


def fig_sft() -> None:
    rows = metrics(SFT / "metrics.jsonl")
    x = [r["tokens"] / 1e9 for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.6), sharex=True)
    axes[0].plot(x, [r["loss"] for r in rows], color=GRAY, linewidth=0.4, alpha=0.45)
    axes[0].plot(x, rolling([r["loss"] for r in rows], 25), color=BLUE, linewidth=1.8)
    axes[0].set_title("Training loss, assistant tokens only")
    axes[1].plot(x, [r["lr"] for r in rows], color=BLUE, linewidth=1.6)
    axes[1].set_title("Learning rate")
    axes[2].plot(x, rolling([r["tok_per_s"] / 1000 for r in rows], 15), color=BLUE, linewidth=1.4)
    axes[2].axhline(55, color=INK2, linewidth=1, linestyle=":")
    axes[2].text(x[0], 56.5, "registered 55k line", fontsize=7.5, color=INK2, va="bottom")
    axes[2].set_ylim(0, 65)
    axes[2].set_title("Speed, thousand token slots per second, rolling median")
    axes[2].set_xlabel("token slots, billions")
    for ax in axes:
        grid(ax, "both")
    finish(fig, "sft_training", "training", "Chat fine-tuning ran 3 epochs with no skipped steps", "Peak learning rate 6e-4, on flash attention at micro-batch 8.", SFT.name + "/metrics.jsonl", 0.91)


def fig_sft_heldout() -> None:
    s = load(SFT / "train_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.4))
    axes[0].plot([f"Epoch {h['epoch'] + 1}" for h in s["history"]], [h["loss"] for h in s["history"]], color=BLUE, marker="o")
    for i, h in enumerate(s["history"]):
        axes[0].annotate(f"{h['loss']:.4f}", (i, h["loss"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=INK)
    axes[0].set_title("Held-out loss by epoch")
    axes[0].margins(x=0.12, y=0.25)
    grid(axes[0], "both")
    sweep = sorted((load(d / "train_summary.json") for d in RUNS.glob("*-train-m4-sft-sweep-*")), key=lambda r: r["lr"])
    best = min(r["history"][-1]["loss"] for r in sweep)
    labs = [f"{r['lr']:g}" for r in sweep]
    axes[1].plot(labs, [r["history"][-1]["loss"] for r in sweep], color=INK2, marker="o")
    for i, r in enumerate(sweep):
        v = r["history"][-1]["loss"]
        axes[1].annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5, color=INK, fontweight="bold" if v == best else "normal")
    axes[1].set_title("Learning-rate sweep: held-out loss, 3% of the data")
    axes[1].margins(x=0.12, y=0.25)
    grid(axes[1], "both")
    finish(fig, "sft_heldout", "training", "Held-out loss fell every epoch, and the sweep chose the top of its grid", "Lower is better. This chooses the epoch and says nothing about quality against other models.", SFT.name + "/train_summary.json", 0.8)


def fig_tune() -> None:
    t = load(sorted(RUNS.glob("*-bench-tune-throughput-*"))[0] / "bench" / "tune.json")
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    for backend, c, lab in (("sdpa", BLUE, "SDPA"), ("flash", ORANGE, "flash-attn")):
        ok = [r for r in t["screen"] if r["backend"] == backend and r["status"] == "ok"]
        ax.plot([str(r["micro_batch"]) for r in ok], [r["tok_per_s"] / 1000 for r in ok], color=c, marker="o", label=lab, linestyle="-" if backend == "sdpa" else "--")
    ax.set_xlabel("micro-batch (32 ran out of memory on both)")
    ax.set_ylabel("thousand tokens per second")
    ax.set_ylim(0, 95)
    grid(ax, "both")
    ax.legend(frameon=False, fontsize=8)
    finish(fig, "a100_tuning", "training", "On the rented A100, the two attention backends were nearly equal", "Compiled, fixed-length rows, synthetic data, 2,048 context. Real conversations ran slower.", sorted(RUNS.glob("*-bench-tune-throughput-*"))[0].name, 0.82)


def write_index() -> None:
    sections = (("text-to-data", "Text-to-data comparison"), ("base", "Base models"), ("cpu", "CPU and Raspberry Pi"), ("chat", "Chat models"), ("training", "Training"))
    lines = ["# Figures", "", "Generated from the run records by `python -m scripts.build_figures`. Don't edit by hand. V2 is always blue and V1 always orange. The tables behind most charts are in [results.md](../results.md).", ""]
    for key, title in sections:
        lines += [f"## {title}", ""]
        for name, sec, t, sub, src in INDEX:
            if sec == key:
                lines += [f"![{t}]({name}.svg)", "", f"*{sub}* Source: `{src}`.", ""]
    (OUT / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


ALL = [fig_t2d_scores, fig_t2d_gap, fig_t2d_tasks, fig_t2d_cuad, fig_t2d_cpu, fig_t2d_size, fig_t2d_dev_test, fig_t2d_lr, fig_t2d_loss, fig_billsum, fig_base_mean, fig_base_tasks, fig_base_few, fig_base_size, fig_base_bpb,
       fig_base_speed, fig_base_speed_quality, fig_cpu_threads, fig_cpu_ram_speed, fig_gguf_ram_speed, fig_gguf_rivals, fig_v1_v2, fig_checkpoints, fig_chat, fig_xstest, fig_m3_loss, fig_m3_signals, fig_m3_bpb, fig_m3_gate, fig_m3_lr, fig_ladder, fig_ladder_seeds, fig_proxy_lr, fig_sft, fig_sft_heldout, fig_tune]


def main() -> int:
    for f in ALL:
        f()
    write_index()
    if "--check" in sys.argv:
        new = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in OUT.iterdir()}
        old = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in COMMITTED.iterdir()}
        stale = sorted(n for n in new.keys() | old.keys() if new.get(n) != old.get(n))
        print(f"stale figures: {stale}" if stale else "figures are up to date")
        return 1 if stale else 0
    print(f"{len(ALL)} figures in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
