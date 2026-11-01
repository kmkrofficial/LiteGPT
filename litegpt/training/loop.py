"""The training loop shared by the M2 proxy ladder and M3 pretraining.

AdamW (decay on 2-D weights only), gradient accumulation, bf16 autocast, optional torch.compile, gradient clipping, a WSD
schedule, non-finite-loss skipping and spike alerts, evaluation every N tokens (token-level validation loss / BPB and
tokenizer-independent BPB on raw `val_text`), atomic checkpoints with the loader position (exact resume), GPU telemetry
with a temperature abort, and every number written to the run record. With `cfg.schedule` (M3): stable phase, the extension
gate, decay on its own index, the context extension and milestones (litegpt/training/m3schedule.py).
"""
from __future__ import annotations

import dataclasses
import gc
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
from litegpt.training.m3schedule import Schedule, decide_gate
from litegpt.training.schedule import wsd_lr


def _loader(cfg: TrainConfig, index_name: str, batch_size: int, seq_len: int | None = None) -> D.MixtureLoader:
    shards_dir = Path(cfg.data_run) / "shards"
    src, blk, names, meta = D.load_index(shards_dir / "indexes" / f"{index_name}.npz")
    part = meta["part"]
    shards = {}
    for n in names:
        bins = sorted((shards_dir / n / part).glob("*.bin"))
        shards[n] = D.ShardSet(bins)
    return D.MixtureLoader(src, blk, names, shards, batch_size=batch_size, block_len=cfg.block_len, seq_len=seq_len)


def _val_text(cfg: TrainConfig) -> dict:
    if not cfg.val_text_dir:
        return {}
    out = {}
    for dom in cfg.val_text_domains:
        p = Path(cfg.val_text_dir) / f"{dom}.jsonl"
        if p.exists():
            out[dom] = [r["text"] for r in read_jsonl(p)]
    return out


def evaluate(model, cfg: TrainConfig, val_loader, byte_lengths, encoder, val_docs, device, fwd=None) -> dict:
    """`fwd` (the compiled model) is used when cfg.eval_fast; the same blocks, windows and targets either way."""
    model.eval()
    net = fwd if cfg.eval_fast and fwd is not None else model
    res = {"val": token_val(net, val_loader, max(1, cfg.eval_blocks // val_loader.batch_size), byte_lengths, device)}
    for dom, docs in val_docs.items():
        res[f"text_{dom}"] = val_text_bpb(net, encoder, docs, cfg.block_len, device, batch_size=val_loader.batch_size if cfg.eval_fast else 0)
    model.train()
    return res


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def fit_micro_batch(fwd, model, cfg: TrainConfig, device, optimizer_resident: bool = False) -> tuple[int, int, dict]:
    """CUDA only. Cap the allocator so an overflow raises instead of spilling into shared memory (WDDM, also under WSL2), then
    probe one forward/backward at the configured micro-batch; on out-of-memory halve it and double `accum`. Tokens per step
    and the block order are unchanged (micro_batch * accum is constant; accumulation equals the large batch). The probe
    reserves room for the AdamW state (2 fp32 tensors per parameter) and leaves parameters, grads and RNG untouched."""
    mb, accum = cfg.micro_batch, cfg.accum
    if device.type != "cuda":
        return mb, accum, {}
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved()      # the cap covers this process's own allocations too (mid-run probes hold weights + optimizer)
    opt_bytes = 0 if optimizer_resident else 8 * sum(p.numel() for p in model.parameters())
    cap = max(0.1, (free - 2**30) / total)
    torch.cuda.set_per_process_memory_fraction(max(0.05, (free - 2**30 - opt_bytes) / total))
    gen = torch.Generator().manual_seed(0)
    tried = []
    while True:
        try:
            x = torch.randint(0, cfg.vocab_size, (mb, cfg.block_len), generator=gen).to(device)
            y = torch.randint(0, cfg.vocab_size, (mb, cfg.block_len), generator=gen).to(device)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            loss.backward()
            peak = torch.cuda.max_memory_allocated()
            tried.append({"micro_batch": mb, "fits": True, "peak_GiB": round(peak / 2**30, 2)})
            break
        except Exception as e:  # noqa: BLE001 - only out-of-memory is handled
            if not _is_oom(e) or mb == 1 or (mb // 2) * 2 != mb:
                raise
            tried.append({"micro_batch": mb, "fits": False})
            mb, accum = mb // 2, accum * 2
        finally:
            model.zero_grad(set_to_none=True)
            x = y = loss = None
            gc.collect()                       # a failed attempt's frames keep its tensors alive until collected
            torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(cap)
    return mb, accum, {"free_GiB": round(free / 2**30, 2), "cap_fraction": round(cap, 4), "optimizer_reserve_GiB": round(opt_bytes / 2**30, 2),
                       "tried": tried}


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
    sched = Schedule(cfg.schedule, cfg.tokens_per_step) if cfg.schedule else None
    gate_file = rec.path / "gate" / "extension_gate.json"
    if sched and gate_file.exists():            # the gate is decided once; a resume reads the record
        sched.apply_gate(json.loads(gate_file.read_text(encoding="utf-8")))
    opt = model.configure_optimizers(cfg.weight_decay, cfg.lr, tuple(cfg.betas), device.type)
    ckpt = CheckpointManager(rec.path / "checkpoints", keep_last=cfg.keep_last, spec_sha256=spec_sha256())
    if resume and sched and ckpt.latest() is not None:   # the checkpoint knows its context (a stop can land on a phase boundary)
        saved = torch.load(ckpt.latest(), map_location="cpu", weights_only=False, mmap=True)["model_config"]
        if saved["block_size"] != model.config.block_size:
            model.set_context(saved["block_size"], saved["rope_theta"])
        del saved
    fwd = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
    mb, accum, probe = fit_micro_batch(fwd, model, cfg, device)
    if probe:
        rec.update(micro_batch_probe=probe, micro_batch_used=mb, accum_used=accum)
        if (mb, accum) != (cfg.micro_batch, cfg.accum):
            rec.log("events", event="micro_batch_halved", configured=cfg.micro_batch, used=mb, accum_used=accum)
            cfg = dataclasses.replace(cfg, micro_batch=mb, accum=accum)
    train_loader = _loader(cfg, cfg.train_index, cfg.micro_batch)
    val_loader = _loader(cfg, cfg.val_index, cfg.eval_micro_batch or cfg.micro_batch)
    decay_loader = _loader(cfg, cfg.schedule["decay_index"], cfg.micro_batch) if sched else None
    step, tokens, skipped, spikes = 0, 0, 0, 0
    if resume and ckpt.latest() is not None:
        state = ckpt.load(model=model, optimizer=opt, expected_config=model.config.to_dict())
        step, tokens = state["step"], state["tokens"]
        if sched:
            train_loader.load_state_dict(state["data_position"]["stable"])
            decay_loader.load_state_dict(state["data_position"]["decay"])
        else:
            train_loader.load_state_dict(state["data_position"])
        skipped, spikes = state["extra"].get("skipped", 0), state["extra"].get("spikes", 0)
        rec.log("events", event="resumed", step=step, tokens=tokens)
    total = sched.total_steps if sched else cfg.total_steps
    warmup = sched.warmup_steps if sched else max(1, round(total * cfg.warmup_frac))
    decay = sched.decay_steps if sched else round(total * cfg.decay_frac)
    if sched:
        need = max(sched.steps(cfg.schedule["gate"]["max_stable_end_tokens"]), sched.stable_end_step) * cfg.micro_batch * cfg.accum
        if len(train_loader.src) < need:
            raise ValueError(f"index {cfg.train_index} has {len(train_loader.src)} blocks; the largest stable phase needs {need}")
    elif len(train_loader.src) < total * cfg.micro_batch * cfg.accum:
        raise ValueError(f"index {cfg.train_index} has {len(train_loader.src)} blocks; the run needs {total * cfg.micro_batch * cfg.accum}")
    loader, accum, phase = train_loader, cfg.accum, "stable"

    def sync_phase() -> None:
        """Move to the phase of the next step: its data, and at the extension its context, compiled graph and micro-batch."""
        nonlocal loader, accum, phase, fwd, decay_loader
        want = sched.phase(step)
        if want == phase:
            return
        if want == "extension":
            ext = cfg.schedule["extension"]
            if model.config.block_size != ext["seq_len"]:
                save("pre_extension")          # the last 2,048-context weights: the switch itself costs BPB (EXP-M3-OPT notes)
                model.set_context(ext["seq_len"], ext["rope_theta"])
            fwd = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
            ecfg = dataclasses.replace(cfg, block_len=ext["seq_len"], micro_batch=ext["micro_batch"], accum=ext["accum"])
            mb, accum, probe = fit_micro_batch(fwd, model, ecfg, device, optimizer_resident=True)
            if mb * accum * ext["seq_len"] != cfg.tokens_per_step:
                raise ValueError("the context extension must keep the tokens per step")
            position = decay_loader.position
            decay_loader = _loader(cfg, cfg.schedule["decay_index"], mb, seq_len=ext["seq_len"])
            decay_loader.position = position
            rec.log("events", event="context_extension", step=step, seq_len=ext["seq_len"], rope_theta=ext["rope_theta"],
                    micro_batch=mb, accum=accum, probe=probe)
        loader = train_loader if want == "stable" else decay_loader
        phase = want
        rec.log("events", event="phase", step=step, tokens=tokens, phase=want)

    def position() -> dict:
        return {"stable": train_loader.state_dict(), "decay": decay_loader.state_dict()} if sched else train_loader.state_dict()

    def save(milestone: str | None = None) -> None:
        ckpt.save(model=model, optimizer=opt, step=step, tokens=tokens, model_config=model.config.to_dict(), data_position=position(),
                  extra={"skipped": skipped, "spikes": spikes}, milestone=milestone)

    if sched:
        sync_phase()
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
    limit = step + max_steps if max_steps else None
    stop_at = min(total, limit) if limit else total
    model.train()
    with GracefulInterrupt() as interrupt:
        while step < stop_at:
            lr = sched.lr(step, cfg.lr, cfg.final_lr_frac) if sched else wsd_lr(step, total, cfg.lr, warmup, decay, cfg.final_lr_frac)
            for g in opt.param_groups:
                g["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for _ in range(accum):
                x, y = loader.next_batch()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _, loss = fwd(x, y)
                (loss / accum).backward()
                loss_sum += loss.detach() / accum
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
                ev = evaluate(model, cfg, val_loader, byte_lengths, encoder, val_docs, device, fwd=fwd)
                rec.log("eval", step=step, tokens=tokens, **{k: v for k, v in ev.items()})
                next_eval += int(cfg.eval_every_tokens)
            if sched:
                if not sched.decided and step == sched.gate_step:
                    record = decide_gate(sched, read_jsonl(rec.path / "eval.jsonl"), read_jsonl(rec.path / "metrics.jsonl"), time.time())
                    rec.write_json("gate/extension_gate.json", {**record, "spec_sha256": spec_sha256()})
                    sched.apply_gate(record)
                    total = sched.total_steps
                    stop_at = min(total, limit) if limit else total
                    rec.log("events", event="gate", step=step, decision=record["decision"], delta=record["delta"],
                            stable_end_tokens=record["stable_end_tokens"], reason=record["reason"])
                for t in cfg.schedule.get("milestone_tokens", []):
                    name = f"tokens-{t / 1e9:g}B"
                    if tokens >= t and not ckpt.milestone_path(name).exists():
                        save(name)
                if sched.decided and step == sched.stable_end_step:
                    save("stable_end")
                if step < total:
                    sync_phase()
            if interrupt.requested or time.time() - last_ckpt > cfg.checkpoint_every_minutes * 60:
                save()
                last_ckpt = time.time()
                if interrupt.requested:
                    status = "interrupted"
                    break
    save("final" if status == "completed" and step == total else None)
    tel = sampler.stop() if sampler is not None else {}
    metrics = read_jsonl(rec.path / "metrics.jsonl")
    evals = read_jsonl(rec.path / "eval.jsonl")
    summary = {"status": status, "step": step, "total_steps": total, "tokens": tokens, "skipped_steps": skipped, "loss_spikes": spikes,
               "final_eval": evals[-1] if evals else None, "mean_tok_per_s": float(np.mean([m["tok_per_s"] for m in metrics[1:]])) if len(metrics) > 1 else None,
               "params": model.num_params(), "params_non_embedding": model.num_params(non_embedding=True), "telemetry": tel,
               "micro_batch_used": cfg.micro_batch, "accum_used": cfg.accum}
    rec.write_json("train_summary.json", summary)
    return summary
