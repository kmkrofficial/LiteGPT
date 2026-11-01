"""EXP-M3-OPT eval_fast on the rented GPU: the current evaluation vs the fast path, same weights and data (freshly initialised
weights are enough: both arms must agree and the time is what is measured). Adopted only if the time drops by >= 25%.

    python -m benchmarks.eval_speed --config benchmarks/m3/m3.yaml [--eval-micro-batch 16] [--blocks 2048]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch

from litegpt.training import load_config
from litegpt.training.bpb import TextEncoder, token_byte_lengths
from litegpt.training.config import build_model
from litegpt.training.loop import _loader, _val_text, evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-micro-batch", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=None, help="the training micro-batch, which the current evaluation path uses")
    ap.add_argument("--blocks", type=int, default=2048, help="validation blocks per evaluation; the speed ratio does not need all 10,240")
    a = ap.parse_args()
    cfg = dataclasses.replace(load_config(a.config), eval_blocks=a.blocks)
    if a.micro_batch:
        cfg = dataclasses.replace(cfg, micro_batch=a.micro_batch, accum=524288 // (a.micro_batch * cfg.block_len))
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_model(cfg).to(dev)
    fwd = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
    lengths, enc, docs = token_byte_lengths(cfg.tokenizer, cfg.vocab_size), TextEncoder(cfg.tokenizer), _val_text(cfg)

    def run(c):
        loader = _loader(c, c.val_index, c.eval_micro_batch or c.micro_batch)
        t = time.time()
        ev = evaluate(model, c, loader, lengths, enc, docs, dev, fwd=fwd)
        torch.cuda.synchronize()
        return time.time() - t, {k: v["bpb"] for k, v in ev.items()}

    base_s, base = run(dataclasses.replace(cfg, eval_fast=False, eval_micro_batch=None))
    fast = dataclasses.replace(cfg, eval_fast=True, eval_micro_batch=a.eval_micro_batch)
    run(fast)                                  # first call compiles
    fast_s, got = run(fast)
    diff = max(abs(got[k] - base[k]) for k in base)
    drop = 1 - fast_s / base_s
    print(json.dumps({"current_s": round(base_s, 1), "fast_s": round(fast_s, 1), "time_drop": round(drop, 3), "max_abs_diff": diff,
                      "adopt": bool(drop >= 0.25 and diff <= 0.001)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
