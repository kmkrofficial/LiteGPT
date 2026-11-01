"""M3 schedule: phases, the pre-registered gate with its budget, context extension, and exact resume across every phase boundary."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import data as D  # noqa: E402
from litegpt.training import TrainConfig, train  # noqa: E402
from litegpt.training.m3schedule import Schedule, decide_gate  # noqa: E402

BL = 32
TPS = BL * 4        # tokens per step in the end-to-end tests: 4 x 32
G = 1_000_000_000


def spec(**kw):
    s = {"warmup_tokens": 0.1 * G, "stable_end_tokens": 9 * G, "decay_tokens": 1 * G, "decay_index": "decay",
         "extension": {"seq_len": 4096, "rope_theta": 100000.0, "tokens": 0.4 * G, "micro_batch": 2, "accum": 64},
         "gate": {"window_a": [6.5 * G, 6.75 * G, 7 * G], "window_b": [8.5 * G, 8.75 * G, 9 * G], "threshold_bpb": 0.010,
                  "max_stable_end_tokens": 14 * G, "step_tokens": 0.5 * G, "speed_window_tokens": 1 * G}}
    return {**s, **kw}


def evals(a=1.20, b=1.18):
    tps = 524288
    rows = []
    for v, ts in ((a, (6.5, 6.75, 7)), (b, (8.5, 8.75, 9))):
        for t in ts:
            rows.append({"tokens": int(np.ceil(t * G / tps)) * tps, "val": {"bpb": v}})
    return rows


def metrics(tok_per_s=95000.0, t_end=1e6, stable=True):
    tps = 524288
    gate = int(np.ceil(9 * G / tps)) * tps
    rows = [{"step": s, "tokens": s * tps, "t": t_end - (gate - s * tps) / tok_per_s, "grad_norm": 0.3} for s in range(1000, gate // tps + 1, 10)]
    if not stable:
        rows = [dict(m, grad_norm=50.0) if m["step"] > 9000 else m for m in rows]
    return rows


def test_schedule_steps_and_phases():
    s = Schedule(spec(), 524288)
    assert s.gate_step == 17167 and s.decay_steps == 1908 and s.extension_steps == 763 and s.total_steps == 19075
    assert (17167 * 524288) >= 9 * G > (17166 * 524288)
    assert s.phase(0) == s.phase(17166) == "stable"
    assert s.phase(17167) == "decay" and s.phase(s.extension_start_step - 1) == "decay"
    assert s.phase(s.extension_start_step) == s.phase(s.total_steps - 1) == "extension"


def test_lr_is_peak_until_the_decay_and_moves_with_the_gate():
    s = Schedule(spec(), 524288)
    assert s.lr(0, 2.4e-3, 0.1) < 2.4e-3 and s.lr(s.warmup_steps, 2.4e-3, 0.1) == 2.4e-3
    assert s.lr(s.gate_step - 1, 2.4e-3, 0.1) == 2.4e-3 and s.lr(s.gate_step, 2.4e-3, 0.1) < 2.4e-3
    assert s.lr(s.total_steps - 1, 2.4e-3, 0.1) == pytest.approx(2.4e-4)
    s.apply_gate({"stable_end_tokens": 14 * G})
    assert s.phase(s.gate_step) == "stable" and s.lr(s.gate_step, 2.4e-3, 0.1) == 2.4e-3 and s.total_steps > 19075


def test_gate_extends_at_the_threshold_and_stops_below():
    s = Schedule(spec(), 524288)
    up = decide_gate(s, evals(1.20, 1.19), metrics(), 1e6)
    assert up["decision"] == "extend" and up["stable_end_tokens"] == 14 * G and up["delta"] == pytest.approx(0.01)
    low = decide_gate(s, evals(1.20, 1.195), metrics(), 1e6)
    assert low["decision"] == "stop" and low["stable_end_tokens"] == 9 * G


def test_gate_invalidation_means_stop():
    s = Schedule(spec(), 524288)
    missing = decide_gate(s, evals()[:-1], metrics(), 1e6)
    assert missing["decision"] == "stop" and "missing" in missing["reason"]
    div = decide_gate(s, evals(), metrics(stable=False), 1e6)
    assert div["decision"] == "stop" and "divergence" in div["reason"]


def test_gate_uses_the_last_evaluation_after_a_resume():
    s = Schedule(spec(), 524288)
    e = evals(1.20, 1.18)
    last = [dict(r) for r in e if r["val"]["bpb"] == 1.18][-1]
    e.append({**last, "val": {"bpb": 1.21}})
    r = decide_gate(s, e, metrics(), 1e6)
    assert r["inputs"]["window_b"][str(9 * G)] == 1.21


def budget(rented_hours_ago, cap=2400.0):
    return {"price_usd_h": 0.60, "inr_per_usd": 98.40, "gate_budget_inr": cap, "rented_at_unix": 1e6 - rented_hours_ago * 3600}


def test_budget_without_a_rental_time_fails_at_the_start():
    bud = {"price_usd_h": 0.6, "inr_per_usd": 98.4, "gate_budget_inr": 2400.0}
    with pytest.raises(ValueError, match="rented"):
        Schedule(spec(budget=bud), 524288)


def test_real_m3_config_has_the_decided_schedule():
    from litegpt.training import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "benchmarks" / "m3" / "m3.yaml")
    assert cfg.tokens_per_step == 524288 and cfg.lr == 0.0024 and cfg.train_index == "stable_v2"
    ext = cfg.schedule["extension"]
    assert ext["seq_len"] == 4096 and ext["rope_theta"] == 10000.0 and ext["micro_batch"] * ext["accum"] * 4096 == 524288
    cfg.schedule["budget"]["rented_at_unix"] = 1.0
    s = Schedule(cfg.schedule, cfg.tokens_per_step)
    assert s.gate_step == 17167 and s.total_steps == 19075 and s.extension_steps == 763
    assert cfg.schedule["decay_index"] == "decay_v2" and "rope_theta" not in cfg.model


def test_m3_config_matches_the_spec():
    from litegpt.spec import load_spec
    from litegpt.training import load_config
    spec, cfg = load_spec(), load_config(Path(__file__).resolve().parents[1] / "benchmarks" / "m3" / "m3.yaml")
    t, s = spec["training"]["pretraining"], cfg.schedule
    assert cfg.lr == t["peak_lr"] and cfg.final_lr_frac == t["final_lr_fraction"] and s["warmup_tokens"] == t["warmup_tokens"]
    assert s["stable_end_tokens"] == t["stable_phase_end_tokens"] and s["decay_tokens"] == t["decay_tokens"]
    ext = t["context_extension"]
    assert (s["extension"]["seq_len"], s["extension"]["rope_theta"], s["extension"]["tokens"]) == (ext["seq_len"], ext["rope_theta"], ext["tokens"])
    assert spec["model"]["rope_theta"] == ext["rope_theta"], "the extension must not switch theta"
    g, sg = spec["training"]["extension_gate"], s["gate"]
    assert sg["threshold_bpb"] == g["threshold_bpb"] and sg["window_a"] == g["bpb_window_a"] and sg["window_b"] == g["bpb_window_b"]
    assert s["budget"]["gate_budget_inr"] == g["budget"]["gate_projection_inr"] and sg["step_tokens"] == g["budget"]["step_tokens"]
    idx = spec["data"]["mixture"]["indexes"]
    assert cfg.train_index == idx["stable"] and s["decay_index"] == idx["decay"] and cfg.eval_every_tokens == t["eval_interval_tokens"]


def test_budget_keeps_the_full_extension_when_it_fits():
    s = Schedule(spec(budget=budget(10)), 524288)
    r = decide_gate(s, evals(), metrics(), 1e6)
    assert r["decision"] == "extend" and r["stable_end_tokens"] == 14 * G


def test_budget_trims_to_the_largest_affordable_half_billion():
    s = Schedule(spec(budget=budget(28.6)), 524288)       # ~ the 95k tok/s case: spent ~INR 1,690
    r = decide_gate(s, evals(), metrics(95000.0), 1e6)
    rate = 0.60 * 98.40 / 3600
    assert r["decision"] == "extend" and r["stable_end_tokens"] % (0.5 * G) == 0 and 9.5 * G <= r["stable_end_tokens"] < 14 * G
    spent = 28.6 * 3600 * rate
    assert spent + (r["stable_end_tokens"] - 9 * G + G) / r["budget"]["measured_tok_per_s"] * rate <= 2400
    assert spent + (r["stable_end_tokens"] + 0.5 * G - 9 * G + G) / r["budget"]["measured_tok_per_s"] * rate > 2400
    assert "trimmed" in r["reason"] and r["budget"]["measured_tok_per_s"] == pytest.approx(95000, rel=1e-3)


def test_budget_exhausted_stops():
    r = decide_gate(Schedule(spec(budget=budget(40)), 524288), evals(), metrics(), 1e6)
    assert r["decision"] == "stop" and "budget" in r["reason"]


def test_set_context_changes_rope_only():
    from litegpt.config import get_config
    from litegpt.model import LiteGPT
    torch.manual_seed(0)
    m = LiteGPT(get_config("litegpt-tiny"))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randint(0, 512, (1, 16))
    a = m(x, logits_all=True)[0]
    m.set_context(256, 100000.0)
    assert m.config.block_size == 256 and m.config.rope_theta == 100000.0 and m.rope_cos.shape[0] == 256
    assert before.keys() == m.state_dict().keys() and all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert not torch.equal(a, m(x, logits_all=True)[0])
    assert m(torch.randint(0, 512, (1, 200)), logits_all=True)[0].shape[1] == 200


def test_extension_reads_are_contiguous_and_clamped_at_the_stream_end(tmp_path):
    w = D.ShardWriter(tmp_path / "s", shard_tokens=100)
    w.add(np.arange(330))
    w.close()
    ss = D.ShardSet(sorted((tmp_path / "s").glob("*.bin")))
    src, blk = np.zeros(3, dtype=np.uint8), np.array([0, 3, 9], dtype=np.uint32)
    long = D.MixtureLoader(src, blk, ["a"], {"a": ss}, batch_size=1, block_len=32, seq_len=64)
    x, y = long.next_batch()
    assert x[0].tolist() == list(range(64)) and y[0].tolist() == list(range(1, 65))
    assert long.next_batch()[0][0, 0] == 96 and long.position == 2
    assert long.next_batch()[1][0, -1] == 329                          # block 9 would read past the end: the read moves back
    short = D.MixtureLoader(src, blk, ["a"], {"a": ss}, batch_size=1, block_len=32)
    assert short.next_batch()[0].shape == (1, 32)


@pytest.fixture(scope="module")
def data_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("datarun")
    shards = root / "shards"
    rng = np.random.default_rng(0)
    for part, n in (("stable", BL * 800 + 1), ("decay", BL * 300 + 1), ("val", BL * 40 + 1)):
        w = D.ShardWriter(shards / "fineweb_edu" / part, shard_tokens=4000)
        w.add(rng.integers(40, 2000, size=n))
        w.close()
    (shards / "indexes").mkdir()
    for idx, part, total in (("stable", "stable", 800), ("decay", "decay", 290), ("val", "val", 40)):
        ss = D.ShardSet(sorted((shards / "fineweb_edu" / part).glob("*.bin")))
        src, blk, names = D.build_index({"fineweb_edu": D.block_count(len(ss), BL)}, {"fineweb_edu": 1.0}, total, seed=3)
        D.save_index(shards / "indexes" / f"{idx}.npz", src, blk, names, {"part": part})
    return root


def m3cfg(data_run, threshold=-1e9, **kw):
    """Stable 20 steps with the gate on two windows, up to 30 steps if extended; decay 10 steps, the last 4 at context 64."""
    sch = {"warmup_tokens": TPS * 2, "stable_end_tokens": TPS * 20, "decay_tokens": TPS * 10, "decay_index": "decay",
           "extension": {"seq_len": 64, "rope_theta": 100000.0, "tokens": TPS * 4, "micro_batch": 2, "accum": 1},
           "gate": {"window_a": [TPS * 10], "window_b": [TPS * 20], "threshold_bpb": threshold, "max_stable_end_tokens": TPS * 30,
                    "step_tokens": TPS * 10, "speed_window_tokens": TPS * 10},
           "milestone_tokens": [TPS * 10]}
    base = dict(name="m3", model_kind="litegpt", model={"preset": "litegpt-tiny", "attention_backend": "sdpa"}, tokenizer="litegpt",
                data_run=str(data_run), train_index="stable", val_index="val", tokens=TPS * 30, micro_batch=4, accum=1, lr=1e-3,
                eval_every_tokens=TPS * 10, eval_blocks=8, compile=False, device="cpu", telemetry=False, block_len=BL,
                checkpoint_every_minutes=1e9, seed=7, schedule=sch)
    base.update(kw)
    return TrainConfig(**base)


def only_run(root):
    return next(Path(root).iterdir())


def phases(run):
    return [(e["step"], e["phase"]) for e in map(json.loads, (run / "events.jsonl").read_text().splitlines()) if e["event"] == "phase"]


def test_end_to_end_stop(data_run, tmp_path):
    s = train(m3cfg(data_run, threshold=1e9), runs_root=tmp_path)
    run = only_run(tmp_path)
    assert s["status"] == "completed" and s["step"] == s["total_steps"] == 30
    assert phases(run) == [(20, "decay"), (26, "extension")]
    gate = json.loads((run / "gate" / "extension_gate.json").read_text())
    assert gate["decision"] == "stop" and gate["spec_sha256"]
    pre = torch.load(run / "checkpoints" / "milestone-pre_extension.pt", map_location="cpu", weights_only=False)
    assert pre["step"] == 26 and pre["model_config"]["block_size"] == BL and pre["model_config"]["rope_theta"] != 100000.0
    for m in ("stable_end", "final"):
        assert (run / "checkpoints" / f"milestone-{m}.pt").exists(), m
    assert len(list((run / "checkpoints").glob("milestone-tokens-*.pt"))) == 1
    state = torch.load(run / "checkpoints" / "milestone-final.pt", map_location="cpu", weights_only=False)
    assert state["model_config"]["block_size"] == 64 and state["model_config"]["rope_theta"] == 100000.0
    assert state["data_position"]["stable"]["position"] == 80
    assert state["data_position"]["decay"]["position"] == 6 * 4 + 4 * 2      # 6 steps x 4 blocks, 4 steps x 2 sequences


def test_end_to_end_extend(data_run, tmp_path):
    s = train(m3cfg(data_run), runs_root=tmp_path)
    run = only_run(tmp_path)
    assert s["status"] == "completed" and s["step"] == s["total_steps"] == 40
    assert phases(run) == [(30, "decay"), (36, "extension")]
    state = torch.load(run / "checkpoints" / "milestone-stable_end.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 30 and state["model_config"]["block_size"] == BL


@pytest.mark.parametrize("stop_after", [9, 20, 21, 26, 27])     # mid-stable, the gate step, inside decay, the extension boundary, inside it
def test_resume_is_bit_exact_across_phase_boundaries(data_run, tmp_path, stop_after):
    cfg = m3cfg(data_run, threshold=1e9)
    train(cfg, runs_root=tmp_path / "full")
    ref = torch.load(only_run(tmp_path / "full") / "checkpoints" / "milestone-final.pt", map_location="cpu", weights_only=False)
    part = train(cfg, runs_root=tmp_path / "part", max_steps=stop_after)
    assert part["step"] == stop_after
    run = only_run(tmp_path / "part")
    s = train(cfg, resume=str(run))
    assert s["status"] == "completed" and s["step"] == 30
    got = torch.load(run / "checkpoints" / "milestone-final.pt", map_location="cpu", weights_only=False)
    for k in ref["model"]:
        assert torch.equal(ref["model"][k], got["model"][k]), k
    assert got["data_position"] == ref["data_position"]
