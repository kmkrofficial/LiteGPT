"""Chat SFT training (EXP-M4-SFT): assistant-only loss on whole conversations, from a pretraining checkpoint.

The M3 loop stays as it was; this reuses its optimizer grouping, checkpoints and records. A step is `batch_rows` rows of `row_len`
tokens, split into micro-batches only to fit memory. The learning rate warms up linearly over `warmup_tokens` and then follows a cosine
to `final_lr_frac` of the peak. Held-out assistant-only loss is measured after each epoch and picks the checkpoint; no benchmark is
ever used for that. Training stops early if held-out loss rises after an epoch.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from litegpt import sft as S
from litegpt.checkpoint import CheckpointManager, GracefulInterrupt, config_hash
from litegpt.config import ModelConfig
from litegpt.model import LiteGPT
from litegpt.records import RunRecord, create_run
from litegpt.spec import spec_sha256
from litegpt.training.loop import fit_micro_batch


@dataclass
class SFTConfig:
    name: str
    data_dir: str
    init_checkpoint: str | None           # the pretraining checkpoint; None with `model_preset` (tiny smoke runs only)
    lr: float
    model_preset: str | None = None
    attention_backend: str = "sdpa"
    row_len: int = 2048
    batch_rows: int = 64                  # 131,072 tokens per step
    micro_batch: int = 16
    epochs: int = 3
    warmup_tokens: float = 5e6
    final_lr_frac: float = 0.1
    keep_fraction: float = 1.0            # the sweep's fixed subset
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1
    compile: bool = True
    compile_mode: str | None = None
    eval_micro_batch: int = 32
    checkpoint_every_minutes: float = 30.0
    keep_last: int = 2
    device: str = "cuda"
    log_every: int = 10
    max_eval_conversations: int | None = None   # tiny smoke runs only

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def load_config(path, **overrides) -> SFTConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.update({k: v for k, v in overrides.items() if v is not None})
    d["betas"] = tuple(d.get("betas", (0.9, 0.95)))
    for k in ("data_dir", "init_checkpoint"):
        if d.get(k):
            d[k] = os.path.expanduser(d[k])
    for k in ("lr", "warmup_tokens", "keep_fraction"):
        if isinstance(d.get(k), str):
            d[k] = float(d[k])
    return SFTConfig(**d)


def lr_at(step: int, total: int, warmup: int, peak: float, final_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def _forward(fwd, b, cfg: SFTConfig):
    """Compiled runs and the SDPA backend use the fixed-shape layout; flash uses cu_seqlens, which only it can use to skip work."""
    if cfg.attention_backend == "flash" and not cfg.compile:
        return fwd(b.x, b.y, cu_seqlens=b.cu_seqlens)
    return fwd(b.x, b.y, doc_pos=(b.doc, b.position_ids))


@torch.no_grad()
def heldout_loss(fwd, model, conv: S.Conversations, cfg: SFTConfig, device) -> dict:
    """Mean assistant-only loss per target token over the whole held-out split, in a fixed order."""
    model.eval()
    order = list(range(len(conv)))[:cfg.max_eval_conversations]
    rows = S.pack_rows(conv.lengths, cfg.row_len, order)
    total, targets, t0 = 0.0, 0, time.time()
    for s in range(0, len(rows), cfg.eval_micro_batch):
        b = S.make_batch(conv, rows[s:s + cfg.eval_micro_batch], cfg.row_len, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, loss = _forward(fwd, b, cfg)
        total += float(loss) * b.target_tokens
        targets += b.target_tokens
    model.train()
    return {"loss": total / max(targets, 1), "target_tokens": targets, "conversations": len(order), "seconds": time.time() - t0}


def _load_init(cfg: SFTConfig) -> tuple[LiteGPT, dict]:
    if cfg.init_checkpoint is None:
        from litegpt.config import get_config
        mc = get_config(cfg.model_preset, vocab_size=32768, block_size=max(cfg.row_len, 128), attention_backend=cfg.attention_backend)
        return LiteGPT(mc), {"init": "fresh", "preset": cfg.model_preset}
    state = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False, mmap=True)   # own training file
    mc = ModelConfig.from_dict({**state["model_config"], "attention_backend": cfg.attention_backend})
    model = LiteGPT(mc)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
    return model, {"init": cfg.init_checkpoint, "init_step": state["step"], "init_tokens": state["tokens"], "init_spec_sha256": state.get("spec_sha256")}


def find_run(root: Path, name: str) -> Path | None:
    """The newest run with exactly this configured name."""
    found = []
    for p in sorted(root.glob(f"*-train-{name}-*")):
        try:
            if json.loads((p / "run.json").read_text(encoding="utf-8"))["config"]["name"] == name:
                found.append(p)
        except (OSError, ValueError, KeyError):
            continue
    return found[-1] if found else None


def train_sft(cfg: SFTConfig, runs_root: str | Path | None = None, max_steps: int | None = None) -> dict:
    """Start, or continue the newest run with this name. A completed run returns its summary without training again. `max_steps` stops
    early (smoke runs); the schedule keeps its full length."""
    root = Path(runs_root) if runs_root else Path(__file__).resolve().parents[2] / "runs"
    run_dir = find_run(root, cfg.name)
    if run_dir and (run_dir / "train_summary.json").exists():
        done = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        if done["status"] == "completed":
            return {**done, "run": run_dir.name}
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model, init = _load_init(cfg)
    model = model.to(device)
    if run_dir:
        rec = RunRecord(run_dir, json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    else:
        rec = create_run("train", cfg.name, config=cfg.to_dict(), compile_enabled=cfg.compile, attention_backend=cfg.attention_backend, root=root, extra={"sft_init": init})
    data = S.read_manifest(cfg.data_dir)
    rec.update(sft_data_sha256=data.get("sha256"))
    train_conv = S.Conversations(cfg.data_dir, "train", keep_fraction=cfg.keep_fraction)
    held_conv = S.Conversations(cfg.data_dir, "heldout")
    opt = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas), device.type)
    ckpt = CheckpointManager(rec.path / "checkpoints", keep_last=cfg.keep_last, spec_sha256=spec_sha256())
    fwd = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
    mb = cfg.micro_batch
    if device.type == "cuda":
        probe_cfg = _ProbeConfig(mb, cfg.batch_rows // mb, cfg.row_len, model.config.vocab_size)
        mb, _, probe = fit_micro_batch(_ProbeFwd(fwd, cfg), model, probe_cfg, device)
        rec.update(micro_batch_probe=probe, micro_batch_used=mb)
    rows_per_epoch = [S.epoch_rows(train_conv.lengths, cfg.row_len, cfg.seed, e, cfg.batch_rows) for e in range(cfg.epochs)]
    steps_per_epoch = [len(r) // cfg.batch_rows for r in rows_per_epoch]
    total = sum(steps_per_epoch)
    warmup = max(1, round(cfg.warmup_tokens / (cfg.batch_rows * cfg.row_len)))
    step, slots, real, epoch, row_at, best, history = 0, 0, 0, 0, 0, None, []
    if ckpt.latest() is not None:
        state = ckpt.load(model=model, optimizer=opt, expected_config=model.config.to_dict())
        step, slots = state["step"], state["tokens"]
        pos = state["data_position"]
        epoch, row_at, real, best, history = pos["epoch"], pos["row"], pos["real_tokens"], pos["best"], pos["history"]
        rec.log("events", event="resumed", step=step, tokens=slots, epoch=epoch)
    filled = sum(int(train_conv.lengths[i]) for r in rows_per_epoch[0] for i in r) / max(1, len(rows_per_epoch[0]) * cfg.row_len)
    rec.log("events", event="start", step=step, total_steps=total, steps_per_epoch=steps_per_epoch, params=model.num_params(), row_fill_epoch0=filled, conversations=len(train_conv),
            heldout_conversations=len(held_conv), micro_batch=mb, warmup_steps=warmup)

    def save(milestone: str | None = None) -> None:
        ckpt.save(model=model, optimizer=opt, step=step, tokens=slots, model_config=model.config.to_dict(),
                  data_position={"epoch": epoch, "row": row_at, "real_tokens": real, "best": best, "history": history}, milestone=milestone)

    status, stop_at = "completed", min(total, step + max_steps) if max_steps else total
    last_ckpt, t_log, slots_log, real_log = time.time(), time.time(), slots, real
    skipped, stopped_early = 0, False
    model.train()
    with GracefulInterrupt() as interrupt:
        while step < stop_at and epoch < cfg.epochs and not stopped_early:
            rows = rows_per_epoch[epoch]
            while row_at < len(rows) and step < stop_at:
                lr = lr_at(step, total, warmup, cfg.lr, cfg.final_lr_frac)
                for g in opt.param_groups:
                    g["lr"] = lr
                loss_sum, targets = 0.0, 0
                micro = [rows[row_at + i:row_at + i + mb] for i in range(0, cfg.batch_rows, mb)]
                batches = [S.make_batch(train_conv, m, cfg.row_len, device) for m in micro]
                n_targets = sum(b.target_tokens for b in batches)
                for b in batches:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        _, loss = _forward(fwd, b, cfg)
                    (loss * b.target_tokens / max(n_targets, 1)).backward()     # every target token weighs the same across micro-batches
                    loss_sum += float(loss.detach()) * b.target_tokens
                    targets += b.target_tokens
                    real += b.real_tokens
                loss_v = loss_sum / max(targets, 1)
                if not math.isfinite(loss_v):
                    opt.zero_grad(set_to_none=True)
                    skipped += 1
                    rec.log("events", event="nonfinite_loss_skipped", step=step)
                    if skipped > 20:
                        status = "aborted_nonfinite"
                        break
                    gn = None
                else:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                row_at += cfg.batch_rows
                slots += cfg.batch_rows * cfg.row_len
                if step % cfg.log_every == 0 or step == stop_at:
                    now = time.time()
                    dt = max(now - t_log, 1e-9)
                    rec.log("metrics", step=step, tokens=slots, real_tokens=real, epoch=epoch, loss=loss_v, lr=lr, grad_norm=gn, tok_per_s=(slots - slots_log) / dt, real_tok_per_s=(real - real_log) / dt)
                    t_log, slots_log, real_log = now, slots, real
                if interrupt.requested or time.time() - last_ckpt > cfg.checkpoint_every_minutes * 60:
                    save()
                    last_ckpt = time.time()
                    if interrupt.requested:
                        status = "interrupted"
                        break
            if status != "completed" or (row_at < len(rows)):
                break
            ev = heldout_loss(fwd, model, held_conv, cfg, device)
            history.append({"epoch": epoch, "step": step, "tokens": slots, **ev})
            rec.log("eval", epoch=epoch, step=step, tokens=slots, heldout=ev)
            improved = best is None or ev["loss"] < best["loss"]
            if improved:
                best = {"epoch": epoch, "loss": ev["loss"], "step": step}
            elif cfg.epochs > 1:
                stopped_early = True
                rec.log("events", event="stopped_early", epoch=epoch, loss=ev["loss"], best=best)
            epoch, row_at = epoch + 1, 0
            save(f"epoch{epoch}")
            last_ckpt = time.time()
    if status == "completed" and (epoch >= cfg.epochs or stopped_early) and best is not None:
        chosen = ckpt.milestone_path(f"epoch{best['epoch'] + 1}")
        final = ckpt.milestone_path("final")
        if final.exists():
            final.unlink()
        os.link(chosen, final)                  # the chosen epoch under the name the guard and the evaluations read
        finished_total = step
    else:
        if status == "completed":
            status = "stopped_at_max_steps"
        save()
        finished_total = total
    metrics = rec.path / "metrics.jsonl"
    speeds = [json.loads(v)["tok_per_s"] for v in metrics.read_text(encoding="utf-8").splitlines()[1:]] if metrics.exists() else []
    summary = {"status": status, "step": step, "total_steps": finished_total, "tokens": slots, "real_tokens": real, "lr": cfg.lr, "best": best, "history": history, "stopped_early": stopped_early,
               "skipped_steps": skipped, "mean_tok_per_s": float(np.mean(speeds)) if speeds else None, "row_fill_epoch0": filled, "micro_batch_used": mb,
               "config_hash": config_hash(model.config.to_dict()), "run": rec.path.name}
    rec.write_json("train_summary.json", summary)
    gc.collect()
    return summary


@dataclass
class _ProbeConfig:
    micro_batch: int
    accum: int
    block_len: int
    vocab_size: int


class _ProbeFwd:
    """The memory probe's forward, with the mask path a real step uses: it needs more memory than a plain causal row."""

    def __init__(self, fwd, cfg: SFTConfig):
        self.fwd, self.cfg = fwd, cfg

    def __call__(self, x, y):
        B, T = x.shape
        doc = torch.zeros((B, T), dtype=torch.long, device=x.device)
        pos = torch.arange(T, device=x.device)[None].expand(B, T)
        return self.fwd(x, y, doc_pos=(doc, pos))


def choose_lr(root: Path, names: list[str]) -> dict:
    """The sweep's rule: lowest held-out assistant-only loss wins; a diverged point is out."""
    rows = []
    for name in names:
        run = find_run(root, name)
        s = json.loads((run / "train_summary.json").read_text(encoding="utf-8"))
        finite = s["status"] == "completed" and s["best"] is not None and math.isfinite(s["best"]["loss"])
        rows.append({"name": name, "lr": s["lr"], "loss": s["best"]["loss"] if finite else None, "status": s["status"]})
    alive = [r for r in rows if r["loss"] is not None]
    if not alive:
        raise SystemExit("every point of the sweep diverged: stop and report")
    return {"rows": rows, "chosen_lr": min(alive, key=lambda r: r["loss"])["lr"]}
