"""M4 chat SFT training on a tiny model and CPU: completion, exact resume, early stop, the learning-rate rule."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import chat_template as C  # noqa: E402
from litegpt import sft as S  # noqa: E402
from litegpt.tokenizer import LiteTokenizer  # noqa: E402
from litegpt.training import sft as T  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft_data")
    tok = LiteTokenizer.load()

    def enc(c):
        i, m = C.render(tok, c)
        return (np.asarray(i, dtype=np.uint16) | (np.asarray(m, dtype=np.uint16) << 15)).tobytes()

    convs = [[{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i % 5}"}] for i in range(60)]
    S.write_split(d, "train", [enc(c) for c in convs[:50]])
    S.write_split(d, "heldout", [enc(c) for c in convs[50:]])
    (d / "manifest.json").write_text(json.dumps({"sha256": {}}), encoding="utf-8")
    return d


def config(data_dir, name, **kw):
    base = dict(name=name, data_dir=str(data_dir), init_checkpoint=None, model_preset="litegpt-tiny", lr=1e-3, row_len=64, batch_rows=4, micro_batch=2, epochs=2, warmup_tokens=256.0,
                compile=False, device="cpu", log_every=1, eval_micro_batch=4)
    return T.SFTConfig(**{**base, **kw})


def weights(run: Path, name="milestone-final.pt"):
    return torch.load(run / "checkpoints" / name, map_location="cpu", weights_only=False)["model"]


def test_completes_with_the_chosen_epoch_as_final_and_does_not_train_twice(data_dir, tmp_path):
    cfg = config(data_dir, "t-a")
    s = T.train_sft(cfg, runs_root=tmp_path)
    run = tmp_path / s["run"]
    assert s["status"] == "completed" and s["step"] == s["total_steps"] > 0 and len(s["history"]) >= 1
    chosen = run / "checkpoints" / f"milestone-epoch{s['best']['epoch'] + 1}.pt"
    assert chosen.read_bytes() == (run / "checkpoints" / "milestone-final.pt").read_bytes()
    assert s["best"]["loss"] == min(h["loss"] for h in s["history"])
    assert T.train_sft(cfg, runs_root=tmp_path)["step"] == s["step"]
    assert len(list(tmp_path.glob("*-train-t-a-*"))) == 1


def test_a_stopped_run_resumes_to_the_same_weights(data_dir, tmp_path):
    a = T.train_sft(config(data_dir, "t-full"), runs_root=tmp_path)
    stopped = T.train_sft(config(data_dir, "t-res"), runs_root=tmp_path, max_steps=max(1, a["step"] // 2))
    assert stopped["status"] == "stopped_at_max_steps"
    b = T.train_sft(config(data_dir, "t-res"), runs_root=tmp_path)
    assert b["status"] == "completed" and b["step"] == a["step"] and b["best"]["loss"] == pytest.approx(a["best"]["loss"], rel=1e-6)
    wa, wb = weights(tmp_path / a["run"]), weights(tmp_path / b["run"])
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_stops_early_when_heldout_loss_rises(data_dir, tmp_path, monkeypatch):
    losses = iter([2.0, 3.0, 1.0])

    def fake(fwd, model, conv, cfg, device):
        return {"loss": next(losses), "target_tokens": 1, "conversations": 1, "seconds": 0.0}

    monkeypatch.setattr(T, "heldout_loss", fake)
    s = T.train_sft(config(data_dir, "t-early", epochs=3), runs_root=tmp_path)
    assert s["status"] == "completed" and s["stopped_early"] and len(s["history"]) == 2 and s["best"]["epoch"] == 0


def test_only_assistant_tokens_are_counted_in_the_loss(data_dir):
    conv = S.Conversations(data_dir, "heldout")
    b = S.make_batch(conv, [[0, 1]], 64, "cpu")
    assert 0 < b.target_tokens < b.real_tokens


def test_learning_rate_warms_up_then_falls_to_the_final_fraction():
    assert T.lr_at(0, 100, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert T.lr_at(9, 100, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert T.lr_at(10, 100, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert T.lr_at(100, 100, 10, 1.0, 0.1) == pytest.approx(0.1)


def test_sweep_picks_the_lowest_heldout_loss_and_drops_a_diverged_point(data_dir, tmp_path):
    for name, lr in (("s-a", 1e-3), ("s-b", 1e-4)):
        T.train_sft(config(data_dir, name, lr=lr, epochs=1), runs_root=tmp_path)
    res = T.choose_lr(tmp_path, ["s-a", "s-b"])
    assert res["chosen_lr"] == min(res["rows"], key=lambda r: r["loss"])["lr"]
    p = T.find_run(tmp_path, "s-a") / "train_summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps({**s, "status": "aborted_nonfinite"}), encoding="utf-8")
    assert T.choose_lr(tmp_path, ["s-a", "s-b"])["chosen_lr"] == 1e-4
