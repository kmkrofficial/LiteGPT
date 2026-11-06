"""
Eager vs torch.compile equivalence check (EXP-M0-COMPILE correctness criterion).

  python -m benchmarks.compile_check [--config litegpt-v2-28l] [--micro-batch 3] [--seq-len 2048]

With identical weights and the same batch, compares forward logits, loss and gradients between the eager model and the
compiled model under bf16 autocast. Writes a run record (kind 'bench') with bench/compile_equivalence.json. Needs CUDA + Triton.
"""

from __future__ import annotations

import argparse
import json

import torch

from litegpt.config import get_config
from litegpt.model import LiteGPT
from litegpt.records import create_run


def check(config: str, micro_batch: int, seq_len: int, backend: str = "sdpa") -> dict:
    torch.manual_seed(0)
    cfg = get_config(config, attention_backend=backend, block_size=seq_len)
    m = LiteGPT(cfg).cuda()
    c = torch.compile(m)
    x = torch.randint(0, cfg.vocab_size, (micro_batch, seq_len), device="cuda")
    y = torch.roll(x, -1, 1)
    last = cfg.n_layer - 1

    def run(model):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in m.parameters() if p.grad is not None)).item()
        grads = {n: p.grad.detach().float().clone() for n, p in m.named_parameters()
                 if n.endswith(("q_proj.weight", "down_proj.weight")) and f".layers.{last}." in n}
        return logits.detach().float(), loss.item(), gn, grads

    le, loss_e, gn_e, ge = run(m)
    lc, loss_c, gn_c, gc = run(c)

    def rel(a, b):
        return ((a - b).norm() / a.norm()).item()

    return {
        "config": config, "micro_batch": micro_batch, "seq_len": seq_len, "backend": backend,
        "loss_eager": loss_e, "loss_compiled": loss_c, "loss_abs_diff": abs(loss_e - loss_c),
        "logits_rel_l2_diff": rel(le, lc), "logits_max_abs_diff": (le - lc).abs().max().item(),
        "grad_norm_eager": gn_e, "grad_norm_compiled": gn_c, "grad_norm_rel_diff": abs(gn_e - gn_c) / gn_e,
        "last_layer_grad_rel_l2_diff": {n: rel(ge[n], gc[n]) for n in ge},
        "all_finite": bool(torch.isfinite(lc).all().item()),
        "tolerance": {"logits_rel_l2_diff_max": 5e-2, "grad_norm_rel_diff_max": 5e-2},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="litegpt-v2-28l")
    ap.add_argument("--micro-batch", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--backend", default="sdpa", choices=["sdpa", "flash"])
    a = ap.parse_args(argv)
    res = check(a.config, a.micro_batch, a.seq_len, a.backend)
    tol = res["tolerance"]
    res["passed"] = bool(res["all_finite"] and res["logits_rel_l2_diff"] <= tol["logits_rel_l2_diff_max"]
                         and res["grad_norm_rel_diff"] <= tol["grad_norm_rel_diff_max"])
    rec = create_run("bench", f"compile-equivalence-{a.config}-{a.backend}", config=vars(a), attention_backend=a.backend,
                     compile_enabled=True)
    rec.write_json("bench/compile_equivalence.json", res)
    print(json.dumps(res, indent=2))
    print("run record:", rec.path)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
