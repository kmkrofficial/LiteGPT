"""The training loop shared by the M2 proxy ladder and M3 pretraining.

AdamW (decay on 2-D weights only), gradient accumulation, bf16 autocast, optional torch.compile, gradient clipping, a WSD
schedule, non-finite-loss skipping and spike alerts, evaluation every N tokens (token-level validation loss / BPB and
tokenizer-independent BPB on raw `val_text`), atomic checkpoints with the loader position (exact resume), GPU telemetry
with a temperature abort, and every number written to the run record.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from litegpt import data as D
from litegpt.checkpoint import CheckpointManager, GracefulInterrupt
from litegpt.records import RunRecord, create_run, read_jsonl
from litegpt.spec import spec_sha256
from litegpt.training.bpb import TextEncoder, token_byte_lengths, token_val, val_text_bpb
from litegpt.training.config import TrainConfig, build_model
from litegpt.training.schedule import wsd_lr


def _loader(cfg: TrainConfig, index_name: str, batch_size: int) -> D.MixtureLoader:
    shards_dir = Path(cfg.data_run) / "shards"
    src, blk, names, meta = D.load_index(shards_dir / "indexes" / f"{index_name}.npz")
    part = meta["part"]
    shards = {}
    for n in names:
        bins = sorted((shards_dir / n / part).glob("*.bin"))
        shards[n] = D.ShardSet(bins)
    return D.MixtureLoader(src, blk, names, shards, batch_size=batch_size, block_len=cfg.block_len)


def _val_text(cfg: TrainConfig) -> dict:
    if not cfg.val_text_dir:
        return {}
    out = {}
    for dom in cfg.val_text_domains:
        p = Path(cfg.val_text_dir) / f"{dom}.jsonl"
        if p.exists():
            out[dom] = [r["text"] for r in read_jsonl(p)]
    return out


def evaluate(model, cfg: TrainConfig, val_loader, byte_lengths, encoder, val_docs, device) -> dict:
    model.eval()
    res = {"val": token_val(model, val_loader, max(1, cfg.eval_blocks // cfg.micro_batch), byte_lengths, device)}
    for dom, docs in val_docs.items():
        res[f"text_{dom}"] = val_text_bpb(model, encoder, docs, cfg.block_len, device)
    model.train()
    return res


def train(cfg: TrainConfig, resume: str | None = None, runs_root: str | Path | None = None, max_steps: int | None = None) -> dict:
    """Run (or resume) a training run. `max_steps` stops early (smoke tests); the schedule still uses the full length."""
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if resume:
        run_dir = Path(resume)
        rec = RunRecord(run_dir, json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    else:
        rec = create_run("train", cfg.name, config=cfg.to_dict(), compile_enabled=cfg.compile,
                         attention_backend=str(cfg.model.get("attention_backend", "sdpa")), root=runs_root)
    model = build_model(cfg).to(device)
    model_cfg = model.config.to_dict()
    opt = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas), device.type)
    train_loader = _loader(cfg, cfg.train_index, cfg.micro_batch)
    val_loader = _loader(cfg, cfg.val_index, cfg.micro_batch)
    ckpt = CheckpointManager(rec.path / "checkpoints", keep_last=cfg.keep_last, spec_sha256=spec_sha256())
    step, tokens, skipped, spikes = 0, 0, 0, 0
    if resume and ckpt.latest() is not None:
        state = ckpt.load(model=model, optimizer=opt, expected_config=model_cfg)
        step, tokens = state["step"], state["tokens"]
        train_loader.load_state_dict(state["data_position"])
        skipped, spikes = state["extra"].get("skipped", 0), state["extra"].get("spikes", 0)
        rec.log("events", event="resumed", step=step, tokens=tokens)
    total = cfg.total_steps
    warmup = max(1, round(total * cfg.warmup_frac))
    decay = round(total * cfg.decay_frac)
    if len(train_loader.src) < total * cfg.micro_batch * cfg.accum:
        raise ValueError(f"index {cfg.train_index} has {len(train_loader.src)} blocks; the run needs {total * cfg.micro_batch * cfg.accum}")
    fwd = torch.compile(model) if cfg.compile else model
    byte_lengths = token_byte_lengths(cfg.tokenizer, cfg.vocab_size)
    encoder = TextEncoder(cfg.tokenizer) if cfg.val_text_dir else None
    val_docs = _val_text(cfg)
    sampler = None
    if cfg.telemetry and device.type == "cuda":
        from benchmarks.telemetry import TelemetrySampler
        sampler = TelemetrySampler(1.0, rec).start()
    rec.log("events", event="start", step=step, total_steps=total, params=model.num_params(),
            params_non_embedding=model.num_params(non_embedding=True), tokens_per_step=cfg.tokens_per_step)
    next_eval = (tokens // int(cfg.eval_every_tokens) + 1) * int(cfg.eval_every_tokens)
    last_ckpt = time.time()
    run_min, ema = math.inf, None
    t_log, tok_log = time.time(), tokens
    status = "completed"
    stop_at = min(total, step + max_steps) if max_steps else total
    model.train()
    with GracefulInterrupt() as interrupt:
        while step < stop_at:
            lr = wsd_lr(step, total, cfg.lr, warmup, decay, cfg.final_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for _ in range(cfg.accum):
                x, y = train_loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = fwd(x, y)
                (loss / cfg.accum).backward()
                loss_sum += loss.detach() / cfg.accum
            loss_v = float(loss_sum)
            if not math.isfinite(loss_v):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                rec.log("events", event="nonfinite_loss_skipped", step=step)
                if skipped > 20:
                    status = "aborted_nonfinite"
                    break
            else:
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema = loss_v if ema is None else 0.98 * ema + 0.02 * loss_v
                run_min = min(run_min, ema)
                if step > warmup and ema > 1.10 * run_min:
                    spikes += 1
                    rec.log("events", event="loss_spike", step=step, ema=ema, running_min=run_min)
            step += 1
            tokens += cfg.tokens_per_step
            if step % 10 == 0 or step == stop_at:
                now = time.time()
                rec.log("metrics", step=step, tokens=tokens, loss=loss_v, lr=lr, grad_norm=gn if math.isfinite(loss_v) else None,
                        tok_per_s=(tokens - tok_log) / max(now - t_log, 1e-9))
                t_log, tok_log = now, tokens
            if sampler is not None and sampler.samples and sampler.samples[-1]["temp_c"] >= cfg.abort_temp_c:
                status = "aborted_temperature"
                rec.log("events", event="temperature_abort", step=step, temp_c=sampler.samples[-1]["temp_c"])
                break
            if tokens >= next_eval or step == total:
                ev = evaluate(model, cfg, val_loader, byte_lengths, encoder, val_docs, device)
                rec.log("eval", step=step, tokens=tokens, **{k: v for k, v in ev.items()})
                next_eval += int(cfg.eval_every_tokens)
            if interrupt.requested or time.time() - last_ckpt > cfg.checkpoint_every_minutes * 60:
                ckpt.save(model=model, optimizer=opt, step=step, tokens=tokens, model_config=model_cfg,
                          data_position=train_loader.state_dict(), extra={"skipped": skipped, "spikes": spikes})
                last_ckpt = time.time()
                if interrupt.requested:
                    status = "interrupted"
                    break
    ckpt.save(model=model, optimizer=opt, step=step, tokens=tokens, model_config=model_cfg, data_position=train_loader.state_dict(),
              extra={"skipped": skipped, "spikes": spikes}, milestone="final" if status == "completed" and step == total else None)
    tel = sampler.stop() if sampler is not None else {}
    metrics = read_jsonl(rec.path / "metrics.jsonl")
    evals = read_jsonl(rec.path / "eval.jsonl")
    summary = {"status": status, "step": step, "total_steps": total, "tokens": tokens, "skipped_steps": skipped, "loss_spikes": spikes,
               "final_eval": evals[-1] if evals else None, "mean_tok_per_s": float(np.mean([m["tok_per_s"] for m in metrics[1:]])) if len(metrics) > 1 else None,
               "params": model.num_params(), "params_non_embedding": model.num_params(non_embedding=True), "telemetry": tel}
    rec.write_json("train_summary.json", summary)
    return summary
