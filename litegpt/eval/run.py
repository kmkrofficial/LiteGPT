"""Run evaluation suite v0 on one model and write a run record (EXP-M1B-SUITE / EXP-M1B-BASELINES).

    python -m litegpt.eval.run --model hf:openai-community/gpt2 --suite base
    python -m litegpt.eval.run --model hf:HuggingFaceTB/SmolLM2-135M-Instruct --suite instruct

HF models run through lm-eval's `hf` backend with the model revision pinned to the commit resolved at start. Native LiteGPT
checkpoints (`litegpt:<path>`) use `litegpt.eval.lm_adapter` (added to the CLI when the first checkpoint exists, M2).
`--limit` is for smoke tests only and marks the record as not comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SUITE = Path(__file__).resolve().parent / "suite_v0.yaml"


def load_suite() -> dict:
    return yaml.safe_load(SUITE.read_text(encoding="utf-8"))


def resolve_model(spec: str) -> dict:
    kind, _, name = spec.partition(":")
    if kind != "hf":
        raise SystemExit("only hf:<repo> models are supported until the first LiteGPT checkpoint exists (M2)")
    from huggingface_hub import HfApi
    info = HfApi().model_info(name)
    return {"kind": "hf", "repo": name, "revision": info.sha, "gated": info.gated,
            "license": (info.card_data.to_dict() if info.card_data else {}).get("license")}


GEN_LENGTH_KEYS = ("max_new_tokens", "max_length")


def generation_length_override(model: dict) -> dict:
    """Length keys in the model's generation_config.json; transformers lets them override lm-eval's generation limit."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(model["repo"], "generation_config.json", revision=model["revision"])
    except Exception:
        return {}
    cfg = json.loads(Path(p).read_text(encoding="utf-8"))
    return {k: cfg[k] for k in GEN_LENGTH_KEYS if cfg.get(k) is not None}


def local_copy(model: dict, template_file: str | None = None) -> tuple[str, dict]:
    """Snapshot of the pinned revision, adjusted for the suite protocol: a chat template for models published without one
    (V1-Instruct), and generation-length keys removed from generation_config.json so lm-eval's limits apply."""
    from huggingface_hub import snapshot_download
    dest = Path.home() / ".cache" / "litegpt" / "adjusted" / f"{model['repo'].replace('/', '__')}@{model['revision'][:12]}"
    snapshot_download(model["repo"], revision=model["revision"], local_dir=dest)
    if template_file:
        cfg_path = dest / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg["chat_template"] = Path(template_file).read_text(encoding="utf-8")
        cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    removed = {}
    gc_path = dest / "generation_config.json"
    if gc_path.exists():
        gc = json.loads(gc_path.read_text(encoding="utf-8"))
        removed = {k: gc.pop(k) for k in GEN_LENGTH_KEYS if gc.get(k) is not None}
        gc_path.write_text(json.dumps(gc, indent=1), encoding="utf-8")
    return str(dest), removed


def task_dataset_revisions(task_names: list[str]) -> dict:
    """Resolved Hub commit of each task's dataset (recorded; compared with the decontamination pins)."""
    from huggingface_hub import HfApi
    from lm_eval.tasks import TaskManager, get_task_dict
    api, out = HfApi(), {}
    tasks = get_task_dict(task_names, TaskManager())

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                cfg = getattr(v, "config", None)
                path = getattr(cfg, "dataset_path", None) if cfg is not None else None   # TaskConfig is falsy
                if path and path not in out:
                    try:
                        out[path] = api.dataset_info(path).sha
                    except Exception as e:                 # local / non-Hub datasets
                        out[path] = f"unresolved: {type(e).__name__}"
    walk(tasks)
    return out


def triviaqa_subset(size: int, seed: int) -> list[int]:
    from lm_eval.tasks import TaskManager, get_task_dict
    task = get_task_dict(["triviaqa"], TaskManager())["triviaqa"]
    n = len(list(task.eval_docs))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), min(size, n)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="hf:<repo>")
    ap.add_argument("--suite", choices=["base", "instruct"], required=True)
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--groups", default=None, help="comma-separated subset of suite groups (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only: record is marked non-comparable")
    ap.add_argument("--extra-model-args", default="", help="appended to lm-eval model_args, e.g. enable_thinking=False (Qwen3)")
    ap.add_argument("--chat-template-file", default=None,
                    help="for chat models without a template (V1-Instruct): a local copy of the pinned snapshot gets this template")
    args = ap.parse_args()

    import lm_eval
    import torch
    from litegpt.records import create_run
    if torch.cuda.is_available():                 # fail instead of silently spilling into shared memory (WDDM, also in WSL2)
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.1, (free - 2**30) / total))
    suite = load_suite()
    model = resolve_model(args.model)
    groups = [g for g in suite[args.suite] if not args.groups or g["name"] in args.groups.split(",")]
    unsafe = any(g.get("unsafe_code") for g in groups)
    if unsafe and platform.system() != "Linux":
        raise SystemExit("code-execution tasks run only in WSL2 (Linux); use --groups to skip them on Windows")
    if unsafe:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    rec = create_run("eval", f"{args.suite}-{model['repo'].split('/')[-1]}", config={
        "suite_version": suite["suite_version"], "suite": args.suite, "groups": [g["name"] for g in groups], "model": model,
        "batch_size": args.batch_size, "limit": args.limit, "comparable": args.limit is None,
        "extra_model_args": args.extra_model_args,
        "chat_template": Path(args.chat_template_file).read_text(encoding="utf-8") if args.chat_template_file else None})
    seed = suite["protocol"]["seed"]
    pretrained = f"pretrained={model['repo']},revision={model['revision']}"
    override = generation_length_override(model)
    removed = {}
    if args.chat_template_file or override:
        path, removed = local_copy(model, args.chat_template_file)
        pretrained = f"pretrained={path}"
    rec.update(generation_config_removed=removed)            # recorded in run.json (protocol correction, EXP-M1B-SUITE)
    model_args = f"{pretrained},dtype=bfloat16,trust_remote_code=False" + (f",{args.extra_model_args}" if args.extra_model_args else "")
    results, revisions = {}, {}
    for g in groups:
        if g.get("native"):
            print(f"[{g['name']}] native task: run with litegpt.eval.xstest (separate entry point)", flush=True)
            continue
        samples = None
        if g.get("subset") and args.limit is None:
            samples = {t: triviaqa_subset(g["subset"]["size"], g["subset"]["seed"]) for t in g["tasks"]}
        revisions.update(task_dataset_revisions(g["tasks"]))
        print(f"[{g['name']}] {g['tasks']} shots={g['num_fewshot']}", flush=True)
        bs = int(args.batch_size)
        while True:                                # batch 16; halved only if it exceeds the VRAM cap (large vocabularies)
            try:
                out = lm_eval.simple_evaluate(
                    model="hf", model_args=model_args, tasks=g["tasks"], num_fewshot=g["num_fewshot"], batch_size=bs,
                    device="cuda", limit=args.limit, samples=samples, log_samples=True, apply_chat_template=bool(g.get("chat")),
                    random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed, fewshot_random_seed=seed,
                    confirm_run_unsafe_code=bool(g.get("unsafe_code")), gen_kwargs=g.get("gen_kwargs"))
                break
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                bs //= 2
                print(f"[{g['name']}] out of memory: retrying with batch size {bs}", flush=True)
        results[g["name"]] = {"batch_size": bs, "results": out["results"], "n_samples": out.get("n-samples"), "versions": out.get("versions"),
                              "configs": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in out.get("configs", {}).items()}}
        rec.write_json(f"eval/{g['name']}.json", results[g["name"]])
        for task, samples_list in (out.get("samples") or {}).items():
            with open(rec.path / "eval" / f"{task}.samples.jsonl", "w", encoding="utf-8") as f:
                for s in samples_list:
                    f.write(json.dumps(s, default=str, ensure_ascii=False) + "\n")
    head = suite["headline_metrics"]
    summary = {}
    for gname, r in results.items():
        for task, metrics in r["results"].items():
            m = head.get(task)
            if m:
                key = next((k for k in metrics if k.startswith(m + ",")), None)
                if key:
                    summary[task] = {"metric": m, "value": metrics[key], "stderr": metrics.get(key.replace(m + ",", m + "_stderr,"))}
    rec.write_json("eval/summary.json", {"model": model, "suite_version": suite["suite_version"], "task_dataset_revisions": revisions,
                                         "headline": summary, "comparable": args.limit is None})
    print(json.dumps(summary, indent=1))
    print("record:", rec.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
