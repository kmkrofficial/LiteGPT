"""Running the text-to-data comparison (EXP-M4-T2D): one interface over LiteGPT and Hugging Face models, the fine-tune and the scoring.

Every arm gets the same batches, loss rule, schedule and decoding. Micro-batches differ only to fit memory, and each target token weighs
the same whatever the split, so they do not change the update. Model output is parsed as data and never executed.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from litegpt import chat_template, t2d as X
from litegpt.records import RunRecord, create_run


def expand_paths(cfg: dict) -> dict:
    for k in ("items_dir", "results_dir"):
        cfg[k] = os.path.expanduser(cfg[k])
    for arm in cfg["arms"].values():
        if arm.get("checkpoint"):
            arm["checkpoint"] = os.path.expanduser(arm["checkpoint"])
    return cfg


class LiteArm:
    """A LiteGPT checkpoint, or a preset for smoke runs."""

    def __init__(self, arm: dict, device):
        from litegpt.config import ModelConfig, get_config
        from litegpt.model import LiteGPT
        from litegpt.tokenizer import LiteTokenizer
        self.arm, self.device, self.tok = arm, device, LiteTokenizer.load()
        if arm.get("checkpoint"):
            state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False, mmap=True)      # own training files only
            cfg = ModelConfig.from_dict({**state["model_config"], "attention_backend": "sdpa"})
            self.model = LiteGPT(cfg)
            self.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
        else:
            self.model = LiteGPT(get_config(arm["preset"], vocab_size=32768, block_size=arm.get("block_size", 1024), attention_backend="sdpa"))
        self.model = self.model.to(device)
        self.chat = arm["layout"] == "chat"

    def encode(self, ex: dict) -> tuple[list[int], list[int]]:
        from litegpt.tokenizer import BOS, EOT, IM_END
        if self.chat:
            prompt = chat_template.render_ids(self.tok, [{"role": "user", "content": ex["user"]}], add_generation_prompt=True)
            return prompt, self.tok.encode_text(ex["target"]) + [IM_END]
        return [BOS] + self.tok.encode_text(ex["user"] + X.ANSWER_CUE), self.tok.encode_text(ex["target"]) + [EOT]

    def logits(self, ids, mask):
        return self.model(ids, logits_all=True)[0]        # causal, so right padding cannot reach a real token

    def context(self) -> int:
        return self.model.config.block_size

    def generate(self, prompts: list[list[int]], new_tokens: int) -> list[str]:
        from litegpt.inference import Engine, GenerationConfig
        from litegpt.tokenizer import EOT, IM_END
        cfg = GenerationConfig(max_new_tokens=new_tokens, temperature=0.0, stop_token_ids=(EOT, IM_END))
        return [r.text for r in Engine(self.model, self.tok).generate_ids(prompts, cfg)]

    def save(self, dest: Path, meta: dict) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        torch.save({"format": 1, "model": self.model.state_dict(), "model_config": self.model.config.to_dict(), "step": meta.get("step", 0), "tokens": meta.get("tokens", 0),
                    "spec_sha256": None, "t2d": meta}, dest / "weights.pt")

    def as_eval(self):
        self.model = self.model.to(dtype=torch.bfloat16).eval()


class HFArm:
    """A Hugging Face model at a pinned revision, with no repository code."""

    def __init__(self, arm: dict, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.arm, self.device = arm, device
        src = arm.get("local") or arm["repo"]
        kw = {} if arm.get("local") else {"revision": arm["revision"]}
        self.tok = AutoTokenizer.from_pretrained(src, trust_remote_code=False, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, trust_remote_code=False, **kw).to(device)
        self.chat = arm["layout"] == "chat"
        stop = arm.get("stop_token")
        self.stop = self.tok.convert_tokens_to_ids(stop) if stop else self.tok.eos_token_id
        if self.stop is None:
            raise SystemExit(f"{arm['name']}: no stop token")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.stop

    def encode(self, ex: dict) -> tuple[list[int], list[int]]:
        if self.chat:
            prompt = self.tok.apply_chat_template([{"role": "user", "content": ex["user"]}], add_generation_prompt=True, tokenize=True, return_dict=False)
        else:
            prompt = self.tok(ex["user"] + X.ANSWER_CUE, add_special_tokens=False)["input_ids"]
            if self.tok.bos_token_id is not None:
                prompt = [self.tok.bos_token_id] + prompt
        return list(prompt), self.tok(ex["target"], add_special_tokens=False)["input_ids"] + [self.stop]

    def logits(self, ids, mask):
        return self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits

    def context(self) -> int:
        return int(getattr(self.model.config, "max_position_embeddings", 2048))

    def generate(self, prompts: list[list[int]], new_tokens: int) -> list[str]:
        from transformers import GenerationConfig
        L = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), L), self.pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(prompts), L), dtype=torch.long, device=self.device)
        for r, p in enumerate(prompts):
            ids[r, L - len(p):] = torch.tensor(p, device=self.device)
            mask[r, L - len(p):] = 1
        cfg = GenerationConfig(max_new_tokens=new_tokens, do_sample=False, eos_token_id=self.stop, pad_token_id=self.pad)   # nothing inherited from the repository's file
        out = self.model.generate(input_ids=ids, attention_mask=mask, generation_config=cfg)[:, L:]
        return [self.tok.decode(o, skip_special_tokens=True) for o in out]

    def save(self, dest: Path, meta: dict) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(dest)
        self.tok.save_pretrained(dest)
        (dest / "t2d_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def as_eval(self):
        self.model = self.model.to(dtype=torch.bfloat16).eval()


def load_arm(arm: dict, device, weights: Path | None = None):
    """An arm's starting weights, or the fine-tuned weights saved in `weights`."""
    arm = dict(arm)
    if weights is not None:
        if arm["kind"] == "litegpt":
            arm["checkpoint"] = str(weights / "weights.pt")
        else:
            arm["local"] = str(weights)
    return LiteArm(arm, device) if arm["kind"] == "litegpt" else HFArm(arm, device)


def cap_memory(device) -> None:
    """An overflow must raise, so the micro-batch can halve, instead of spilling into shared memory and crawling (WDDM, also under WSL2)."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.05, (free - 2**30) / total))


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.05, final_frac: float = 0.1) -> float:
    warmup = max(1, round(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def split_micro(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Positions of one batch, longest first, into micro-batches of at most `max_tokens` padded tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > max_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def micro_tensors(encoded: list[tuple[list[int], list[int]]], device, pad: int = 0):
    L = max(len(p) + len(t) for p, t in encoded)
    ids = torch.full((len(encoded), L), pad, dtype=torch.long)
    mask = torch.zeros((len(encoded), L), dtype=torch.long)
    tgt = torch.full((len(encoded), L), -1, dtype=torch.long)
    for r, (p, t) in enumerate(encoded):
        seq = p + t
        ids[r, :len(seq)] = torch.tensor(seq)
        mask[r, :len(seq)] = 1
        tgt[r, len(p) - 1:len(seq) - 1] = torch.tensor(t)   # each output token is predicted from the position before it
    return ids.to(device), mask.to(device), tgt.to(device)


def build_examples(items_dir, manifest: dict, tasks: list[str], split: str, limit: int | None = None) -> list[dict]:
    out = []
    for task in tasks:
        for it in X.load_items(items_dir, manifest, task, split)[:limit]:
            out.append({"task": task, "id": it["id"], "user": X.user_text(task, it), "target": X.target_text(task, it["gold"]), "gold": it["gold"]})
    return out


def _optimizer(model, lr: float, recipe: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": recipe["weight_decay"]}, {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(recipe["betas"]), fused=next(model.parameters()).is_cuda)


def fine_tune(arm_obj, examples: list[dict], lr: float, seed: int, recipe: dict, rec: RunRecord | None, device) -> dict:
    """The registered recipe. A batch that overflows memory is redone with smaller micro-batches, which does not change the update."""
    torch.manual_seed(seed)
    encoded = [arm_obj.encode(e) for e in examples]
    model = arm_obj.model.train()
    opt = _optimizer(model, lr, recipe)
    epochs = [X.make_batches(examples, seed, e, recipe["max_examples"], recipe["max_chars"]) for e in range(recipe["epochs"])]
    total = sum(map(len, epochs))
    step, seen, t0, skipped, budget = 0, 0, time.time(), 0, recipe["micro_tokens"]
    autocast = {"device_type": device.type, "dtype": torch.bfloat16, "enabled": device.type == "cuda"}
    last_log = time.time()
    for epoch, batches in enumerate(epochs):
        for batch in batches:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, lr)
            rows = [encoded[i] for i in batch]
            n = sum(len(t) for _, t in rows)
            while True:
                try:
                    loss_sum = 0.0
                    for micro in split_micro([len(p) + len(t) for p, t in rows], budget):
                        ids, mask, tgt = micro_tensors([rows[i] for i in micro], device)
                        with torch.autocast(**autocast):
                            logits = arm_obj.logits(ids, mask)
                            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1, reduction="sum")
                        (loss / n).backward()
                        loss_sum += float(loss.detach())
                    break
                except torch.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True)
                    ids = mask = tgt = logits = loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    if budget <= max(len(p) + len(t) for p, t in rows):
                        raise
                    budget //= 2
                    if rec:
                        rec.log("events", event="micro_tokens_halved", step=step, micro_tokens=budget)
            value = loss_sum / max(n, 1)
            if not math.isfinite(value):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                if rec:
                    rec.log("events", event="nonfinite_loss_skipped", step=step)
                if skipped > 20:
                    return {"status": "aborted_nonfinite", "steps": step, "total_steps": total, "seconds": time.time() - t0}
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            seen += sum(len(p) + len(t) for p, t in rows)
            if rec and (step % 20 == 0 or time.time() - last_log > 60 or step == total):
                rec.log("metrics", step=step, total_steps=total, epoch=epoch, tokens=seen, loss=value, lr=opt.param_groups[0]["lr"], tok_per_s=seen / max(time.time() - t0, 1e-9))
                last_log = time.time()
    return {"status": "completed", "steps": step, "total_steps": total, "tokens": seen, "seconds": time.time() - t0, "skipped_steps": skipped, "micro_tokens_final": budget}


@torch.no_grad()
def predict(arm_obj, examples: list[dict], recipe: dict, heartbeat=None) -> tuple[list[str], dict]:
    """Greedy outputs in the examples' order, batched by length within the token budget, at the suite's dtype."""
    encoded = [arm_obj.encode(e)[0] for e in examples]
    n_new, limit = recipe["new_tokens"], arm_obj.context()
    too_long = [e["id"] for e, p in zip(examples, encoded) if len(p) + n_new > limit]
    if too_long:
        raise SystemExit(f"{len(too_long)} prompts do not fit the context of {limit} with {n_new} new tokens, e.g. {too_long[:3]}")
    order = sorted(range(len(examples)), key=lambda i: -len(encoded[i]))
    out: list = [None] * len(examples)
    t0, at, budget, done, finished = time.time(), 0, recipe["gen_tokens"], 0, 0
    while at < len(order):
        size = 1
        while at + size < len(order) and size < recipe["gen_items"] and (size + 1) * (len(encoded[order[at]]) + n_new) <= budget:
            size += 1
        chunk = order[at:at + size]
        try:
            texts = arm_obj.generate([encoded[i] for i in chunk], n_new)
        except torch.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if budget <= len(encoded[order[at]]) + n_new:
                raise
            budget //= 2
            continue
        for i, t in zip(chunk, texts):
            out[i] = t
        at += size
        done += size
        if heartbeat:
            heartbeat(done, len(order))
    elapsed = time.time() - t0
    chars = sum(len(t) for t in out)
    return out, {"items": len(out), "seconds": elapsed, "items_per_s": len(out) / max(elapsed, 1e-9), "output_chars": chars, "chars_per_s": chars / max(elapsed, 1e-9), "gen_tokens_budget_final": budget}


def score_outputs(examples: list[dict], outputs: list[str]) -> dict[str, list[dict]]:
    per: dict[str, list[dict]] = {t: [] for t in X.PRIMARY}
    for e, o in zip(examples, outputs):
        per[e["task"]].append({"id": e["id"], **X.score_item(e["task"], o, e["gold"])})
    return per


def run_fit(cfg: dict, arm_name: str, lr: float, seed: int, runs_root: Path, device, name: str | None = None, limit: int | None = None) -> dict:
    """One fine-tune with its dev score. Returns at once if a completed run of this name exists; an interrupted one starts again."""
    from litegpt.training.sft import find_run
    arm = {**cfg["arms"][arm_name], "name": arm_name}
    name = name or run_name(arm_name, lr, seed)
    old = find_run(runs_root, name)
    if old and (old / "dev" / "score.json").exists():
        return json.loads((old / "dev" / "score.json").read_text(encoding="utf-8"))
    cap_memory(device)
    manifest = X.load_manifest(cfg["manifest"])
    train = build_examples(cfg["items_dir"], manifest, list(X.PRIMARY), "train", limit)
    dev = build_examples(cfg["items_dir"], manifest, list(X.PRIMARY), "dev", limit)
    rec = create_run("train", name, config={"name": name, "experiment": "EXP-M4-T2D", "arm": arm, "lr": lr, "seed": seed, "recipe": cfg["recipe"], "instructions": X.INSTRUCTIONS, "manifest_sha256": X.manifest_sha256(cfg["manifest"])}, root=runs_root)
    obj = load_arm(arm, device)
    params = sum(p.numel() for p in obj.model.parameters())
    res = fine_tune(obj, train, lr, seed, cfg["recipe"], rec, device)
    rec.write_json("train_summary.json", {**res, "params": params})
    if res["status"] != "completed":
        return {"status": res["status"], "arm": arm_name, "lr": lr, "seed": seed, "score": None, "run": rec.path.name}
    obj.as_eval()
    obj.save(rec.path / "weights", {"arm": arm_name, "lr": lr, "seed": seed, "step": res["steps"], "tokens": res["tokens"]})
    outputs, speed = predict(obj, dev, cfg["recipe"], heartbeat=lambda d, n: rec.log("metrics", dev_done=d, dev_total=n))
    write_predictions(rec.path / "dev" / "predictions.jsonl", dev, outputs)
    per = score_outputs(dev, outputs)
    score = {"status": "completed", "arm": arm_name, "lr": lr, "seed": seed, "run": rec.path.name, **X.summarize(per), "generation": speed, "train": res, "params": params}
    rec.write_json("dev/score.json", score)
    del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def run_name(arm: str, lr: float, seed: int) -> str:
    return f"m4-t2d-{arm}-lr{lr:g}-s{seed}"


def write_predictions(path: Path, examples: list[dict], outputs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e, o in zip(examples, outputs):
            f.write(json.dumps({"task": e["task"], "id": e["id"], "output": o}, ensure_ascii=False) + "\n")


def dev_score(runs_root: Path, name: str) -> dict | None:
    from litegpt.training.sft import find_run
    run = find_run(runs_root, name)
    p = run / "dev" / "score.json" if run else None
    return json.loads(p.read_text(encoding="utf-8")) if p and p.exists() else None


def choose_lr(cfg: dict, arm: str, runs_root: Path) -> dict:
    """The dev score of each grid point at seed 1. Highest wins; ties go to the lower rate. A diverged point is out."""
    rows = []
    for lr in cfg["recipe"]["lr_grid"]:
        s = dev_score(runs_root, run_name(arm, lr, 1))
        rows.append({"lr": lr, "score": s["score"] if s else None})
    alive = [r for r in rows if r["score"] is not None]
    if not alive:
        raise SystemExit(f"{arm}: no grid point finished")
    return {"arm": arm, "rows": rows, "chosen_lr": max(alive, key=lambda r: (r["score"], -r["lr"]))["lr"]}


def choose_routes(cfg: dict, runs_root: Path, choices: dict) -> dict:
    """Each model's route by the mean dev score over its seeds. A route without all its seeds is out (V2's chat route if G2 failed)."""
    out = {}
    for model, routes in cfg["models"].items():
        rows = []
        for arm in routes:
            if arm not in choices:
                continue
            lr = choices[arm]["chosen_lr"]
            scores = [dev_score(runs_root, run_name(arm, lr, s)) for s in cfg["recipe"]["seeds"]]
            if all(s and s["score"] is not None for s in scores):
                rows.append({"arm": arm, "lr": lr, "dev_scores": [s["score"] for s in scores], "mean": float(np.mean([s["score"] for s in scores]))})
        if rows:                                    # a model with no finished route is left out, and the results show that
            out[model] = {"routes": rows, "chosen": max(rows, key=lambda r: r["mean"])}
    if not out:
        raise SystemExit("no model has a finished route")
    return out


def score_test(cfg: dict, routes: dict, runs_root: Path, out_dir: Path, device) -> dict:
    """Test outputs of every chosen route and seed, once. Refused if a result already exists, so nothing can be re-scored quietly."""
    from litegpt.training.sft import find_run
    result_file = out_dir / "test_results.json"
    if result_file.exists():
        raise SystemExit(f"{result_file} exists: the test set is scored once")
    cap_memory(device)
    manifest = X.load_manifest(cfg["manifest"])
    test = build_examples(cfg["items_dir"], manifest, list(X.PRIMARY), "test")
    per_model, speeds = {}, {}
    for model, r in routes.items():
        arm, lr = r["chosen"]["arm"], r["chosen"]["lr"]
        per_model[model] = {t: [] for t in X.PRIMARY}
        for seed in cfg["recipe"]["seeds"]:
            run = find_run(runs_root, run_name(arm, lr, seed))
            file = out_dir / "predictions" / f"{model}-{arm}-s{seed}.jsonl"
            if file.exists():                   # a restart continues; the outputs are fixed once written
                outputs = [json.loads(x)["output"] for x in file.read_text(encoding="utf-8").splitlines()]
            else:
                obj = load_arm({**cfg["arms"][arm], "name": arm}, device, weights=run / "weights")
                obj.as_eval()
                outputs, speeds[f"{model}-s{seed}"] = predict(obj, test, cfg["recipe"], heartbeat=lambda d, n: (out_dir / "progress").write_text(f"{model} s{seed} {d} of {n}", encoding="utf-8"))
                write_predictions(file, test, outputs)
                del obj
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            per = score_outputs(test, outputs)
            for t in X.PRIMARY:
                per_model[model][t].append(per[t])
    def only(per_model_one: dict, keep: dict) -> dict:
        return {t: [[s for s in seed if s["id"] not in keep[t]] for seed in per_model_one[t]] for t in X.PRIMARY}

    flagged = {t: set(manifest["tasks"][t]["test"]["flagged_kept_ids"]) for t in X.PRIMARY}
    res = {"models": {}, "comparisons": {}, "zero_match_comparisons": {}, "speeds": speeds, "routes": routes, "test_items": {t: len(per_model[next(iter(per_model))][t][0]) for t in X.PRIMARY}}
    for model, per in per_model.items():
        res["models"][model] = {"seeds": [X.summarize({t: per[t][i] for t in X.PRIMARY}) for i in range(len(cfg["recipe"]["seeds"]))],
                                "zero_match_seeds": [X.summarize({t: only(per, flagged)[t][i] for t in X.PRIMARY}) for i in range(len(cfg["recipe"]["seeds"]))]}
    for rival in cfg["compare_against"]:
        if rival in per_model and "v2" in per_model:
            res["comparisons"][rival] = X.compare(per_model["v2"], per_model[rival])
            res["zero_match_comparisons"][rival] = X.compare(only(per_model["v2"], flagged), only(per_model[rival], flagged))
    result_file.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def run_billsum(cfg: dict, model: str, route: dict, runs_root: Path, out_dir: Path, device, limit: int | None = None) -> dict:
    """BillSum, reported only: the chosen route, one seed, from the arm's starting weights, last."""
    file = out_dir / f"billsum-{model}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    arm_name, lr = route["arm"], route["lr"]
    arm, seed = {**cfg["arms"][arm_name], "name": arm_name}, cfg["recipe"]["seeds"][0]
    cap_memory(device)
    manifest = X.load_manifest(cfg["manifest"])
    train = build_examples(cfg["items_dir"], manifest, ["billsum"], "train", limit)
    test = build_examples(cfg["items_dir"], manifest, ["billsum"], "test", limit)
    rec = create_run("train", f"m4-billsum-{model}", config={"name": f"m4-billsum-{model}", "experiment": "EXP-M4-T2D", "task": "billsum", "arm": arm, "lr": lr, "seed": seed, "recipe": cfg["recipe"]}, root=runs_root)
    obj = load_arm(arm, device)
    res = fine_tune(obj, train, lr, seed, cfg["recipe"], rec, device)
    rec.write_json("train_summary.json", res)
    if res["status"] != "completed":
        return {"model": model, "status": res["status"]}
    obj.as_eval()
    outputs, speed = predict(obj, test, cfg["recipe"], heartbeat=lambda d, n: rec.log("metrics", test_done=d, test_total=n))
    write_predictions(rec.path / "test" / "predictions.jsonl", test, outputs)
    rouge = [X.rouge_l(o, e["gold"]) for e, o in zip(test, outputs)]
    flagged = set(manifest["tasks"]["billsum"]["test"]["flagged_kept_ids"])
    zero = [r for r, e in zip(rouge, test) if e["id"] not in flagged]
    out = {"model": model, "arm": arm_name, "lr": lr, "seed": seed, "status": "completed", "rouge_l": float(np.mean(rouge)), "rouge_l_zero_match": float(np.mean(zero)) if zero else None, "items": len(rouge), "zero_match_items": len(zero),
           "generation": speed, "train": res, "run": rec.path.name}
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out, indent=1), encoding="utf-8")
    del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def check_fits(cfg: dict, device_for_load="cpu", arms: list[str] | None = None) -> dict:
    """Every prompt plus 256 new tokens must fit each arm's context, and each arm must say what its prompt looks like. Weights are loaded on the CPU."""
    manifest = X.load_manifest(cfg["manifest"])
    out = {}
    for name in arms or cfg["arms"]:
        arm = {**cfg["arms"][name], "name": name}
        if arm["kind"] == "litegpt" and arm.get("checkpoint") and not Path(arm["checkpoint"]).exists():
            out[name] = {"skipped": f"{arm['checkpoint']} does not exist yet"}
            continue
        obj = load_arm(arm, torch.device(device_for_load))
        row = {"context": obj.context()}
        for task in ("cuad", "massive", "banking77", "billsum"):
            longest = 0
            for split in ("train", "dev", "test"):
                for e in build_examples(cfg["items_dir"], manifest, [task], split):
                    p, t = obj.encode(e)
                    longest = max(longest, len(p) + (cfg["recipe"]["new_tokens"] if split != "train" else len(t)))
            row[f"{task}_longest"] = longest
        row["fits"] = all(row[f"{t}_longest"] <= row["context"] for t in ("cuad", "massive", "banking77", "billsum"))
        sample = obj.encode({"user": "U", "target": "T", "task": "banking77"})
        row["sample_prompt"] = (obj.tok.decode(sample[0]))
        row["sample_target"] = (obj.tok.decode(sample[1]))
        out[name] = row
        del obj
        gc.collect()
    return out
