"""Training loop: schedule shape, accumulation == large batch, bit-exact resume, end-to-end CPU smoke with evaluation."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import data as D  # noqa: E402
from litegpt.training import TrainConfig, train  # noqa: E402
from litegpt.training.schedule import wsd_lr  # noqa: E402

BL = 32


def test_wsd_schedule_shape():
    lrs = [wsd_lr(s, 100, 1.0, 10, 20, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(v == 1.0 for v in lrs[10:80])
    assert lrs[-1] == pytest.approx(0.1) and all(a >= b for a, b in zip(lrs[80:], lrs[81:]))
    assert wsd_lr(90, 100, 1.0, 10, 20, 0.1, shape="linear") < 1.0


@pytest.fixture(scope="module")
def data_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("datarun")
    shards = root / "shards"
    rng = np.random.default_rng(0)
    for name, part, n in (("fineweb_edu", "stable", BL * 400 + 1), ("fineweb_edu", "val", BL * 40 + 1)):
        w = D.ShardWriter(shards / name / part, shard_tokens=4000)
        w.add(rng.integers(40, 2000, size=n))
        w.close()
    (shards / "indexes").mkdir()
    for idx, part, total in (("train", "stable", 400), ("val", "val", 40)):
        ss = D.ShardSet(sorted((shards / "fineweb_edu" / part).glob("*.bin")))
        src, blk, names = D.build_index({"fineweb_edu": D.block_count(len(ss), BL)}, {"fineweb_edu": 1.0}, total, seed=3)
        D.save_index(shards / "indexes" / f"{idx}.npz", src, blk, names, {"part": part})
    vt = root / "val_text"
    vt.mkdir()
    (vt / "web.jsonl").write_text(json.dumps({"key": "a", "text": "The cat sat on the mat. " * 20}) + "\n", encoding="utf-8")
    return root


def cfg(data_run, **kw):
    base = dict(name="unit", model_kind="litegpt", model={"preset": "litegpt-tiny", "attention_backend": "sdpa"}, tokenizer="litegpt",
                data_run=str(data_run), train_index="train", val_index="val", tokens=BL * 4 * 12, micro_batch=4, accum=1, lr=1e-3,
                val_text_dir=str(data_run / "val_text"), val_text_domains=["web"], eval_every_tokens=BL * 4 * 6, eval_blocks=8,
                compile=False, device="cpu", telemetry=False, block_len=BL, checkpoint_every_minutes=1e9, seed=7)
    base.update(kw)
    return TrainConfig(**base)


def final_params(run_dir):
    state = torch.load(sorted((Path(run_dir) / "checkpoints").glob("*.pt"))[-1], map_location="cpu", weights_only=False)
    return state["model"], state["step"]


def test_end_to_end_smoke_with_evaluation(data_run, tmp_path):
    s = train(cfg(data_run), runs_root=tmp_path)
    assert s["status"] == "completed" and s["step"] == 12 and s["skipped_steps"] == 0
    ev = s["final_eval"]
    assert ev["val"]["bpb"] > 0 and ev["text_web"]["bpb"] > 0 and ev["text_web"]["docs"] == 1
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "metrics.jsonl").exists() and (run_dir / "checkpoints" / "milestone-final.pt").exists()


def test_accumulation_equals_large_batch(data_run, tmp_path):
    a = train(cfg(data_run, micro_batch=4, accum=1, tokens=BL * 4 * 3, val_text_dir=None), runs_root=tmp_path / "a")
    b = train(cfg(data_run, micro_batch=2, accum=2, tokens=BL * 4 * 3, val_text_dir=None), runs_root=tmp_path / "b")
    pa, _ = final_params(next((tmp_path / "a").iterdir()))
    pb, _ = final_params(next((tmp_path / "b").iterdir()))
    for k in pa:
        assert torch.allclose(pa[k], pb[k], atol=1e-5), k
    assert a["step"] == b["step"] == 3


def test_resume_is_bit_exact(data_run, tmp_path):
    full = cfg(data_run, val_text_dir=None)
    train(full, runs_root=tmp_path / "full")
    ref, _ = final_params(next((tmp_path / "full").iterdir()))
    part = train(full, runs_root=tmp_path / "part", max_steps=5)
    assert part["status"] == "completed" and part["step"] == 5
    run_dir = next((tmp_path / "part").iterdir())
    s = train(full, resume=str(run_dir))
    assert s["step"] == 12
    got, step = final_params(run_dir)
    assert step == 12
    for k in ref:
        assert torch.equal(ref[k], got[k]), k
