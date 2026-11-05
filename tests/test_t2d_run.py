"""M4 text-to-data runner on tiny LiteGPT models and CPU: every step of the protocol, in order, including the rules that stop misuse."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import t2d as X  # noqa: E402
from litegpt import t2d_run as R  # noqa: E402

CPU = torch.device("cpu")
GOLD = {"document_name": "Lease", "parties": ["Acme"], "agreement_date": None, "governing_law": None}


@pytest.fixture()
def cfg(tmp_path):
    items = tmp_path / "items"
    items.mkdir()
    rows = {"cuad": lambda i: {"id": f"c{i}", "input": f"contract number {i}", "gold": GOLD},
            "massive": lambda i: {"id": f"m{i}", "input": f"wake me at {i}", "gold": [{"type": "time", "value": str(i)}]},
            "banking77": lambda i: {"id": f"b{i}", "input": f"card {i} lost", "gold": "lost_card"},
            "billsum": lambda i: {"id": f"s{i}", "input": f"a bill about {i}", "gold": f"funds {i}"}}
    tasks = {}
    for task, make in rows.items():
        tasks[task] = {}
        for split, n in (("train", 6), ("dev", 3), ("test", 3)):
            p = items / f"{task}_{split}.jsonl"
            p.write_text("".join(json.dumps(make(i)) + "\n" for i in range(n)), encoding="utf-8", newline="\n")
            tasks[task][split] = {"items": n, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        tasks[task]["test"]["flagged_kept_ids"] = [make(0)["id"]]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    tiny = {"kind": "litegpt", "preset": "litegpt-tiny", "block_size": 512}
    return {"manifest": str(manifest), "items_dir": str(items), "results_dir": str(tmp_path / "results"),
            "recipe": {"epochs": 1, "max_examples": 4, "max_chars": 16000, "lr_grid": [1e-5, 1e-4], "seeds": [1, 2], "weight_decay": 0.01, "betas": [0.9, 0.999], "grad_clip": 1.0, "new_tokens": 8,
                       "micro_tokens": 256, "gen_tokens": 4096, "gen_items": 4},
            "arms": {"a-base": {**tiny, "layout": "plain"}, "a-chat": {**tiny, "layout": "chat"}, "b-base": {**tiny, "layout": "plain"}},
            "models": {"v2": ["a-base", "a-chat"], "rival": ["b-base"]}, "compare_against": ["rival"], "runs": str(tmp_path / "runs")}


def test_the_whole_protocol_in_order_and_test_only_once(cfg, tmp_path):
    runs, results = tmp_path / "runs", tmp_path / "results"
    runs.mkdir()
    results.mkdir()
    choices = {}
    for arm in cfg["arms"]:
        for lr in cfg["recipe"]["lr_grid"]:
            assert R.run_fit(cfg, arm, lr, 1, runs, CPU)["status"] == "completed"
        choices[arm] = R.choose_lr(cfg, arm, runs)
        assert R.run_fit(cfg, arm, choices[arm]["chosen_lr"], 2, runs, CPU)["status"] == "completed"
    routes = R.choose_routes(cfg, runs, choices)
    assert set(routes) == {"v2", "rival"} and routes["v2"]["chosen"]["mean"] == max(r["mean"] for r in routes["v2"]["routes"])
    res = R.score_test(cfg, routes, runs, results, CPU)
    assert res["comparisons"]["rival"]["score"]["verdict"] in ("win", "tie", "loss")
    assert res["test_items"] == {"cuad": 3, "massive": 3, "banking77": 3}
    assert "rival" in res["zero_match_comparisons"]
    with pytest.raises(SystemExit):
        R.score_test(cfg, routes, runs, results, CPU)
    b = R.run_billsum(cfg, "v2", routes["v2"]["chosen"], runs, results, CPU)
    assert b["status"] == "completed" and b["items"] == 3 and b["zero_match_items"] == 2
    assert R.run_billsum(cfg, "v2", routes["v2"]["chosen"], runs, results, CPU)["run"] == b["run"]


def test_finished_fit_is_not_repeated_and_records_everything(cfg, tmp_path):
    runs = tmp_path / "runs"
    a = R.run_fit(cfg, "a-base", 1e-4, 1, runs, CPU)
    b = R.run_fit(cfg, "a-base", 1e-4, 1, runs, CPU)
    assert a["run"] == b["run"] and len(list(runs.glob("*-train-*"))) == 1
    run = runs / a["run"]
    assert (run / "dev" / "predictions.jsonl").exists() and (run / "weights" / "weights.pt").exists()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))["config"]
    assert meta["instructions"] == X.INSTRUCTIONS and meta["lr"] == 1e-4


def test_saved_weights_are_the_ones_that_were_scored(cfg, tmp_path):
    runs = tmp_path / "runs"
    a = R.run_fit(cfg, "a-base", 1e-4, 1, runs, CPU)
    run = runs / a["run"]
    dev = R.build_examples(cfg["items_dir"], X.load_manifest(cfg["manifest"]), list(X.PRIMARY), "dev")
    obj = R.load_arm({**cfg["arms"]["a-base"], "name": "a-base"}, CPU, weights=run / "weights")
    obj.as_eval()
    outputs, _ = R.predict(obj, dev, cfg["recipe"])
    saved = [json.loads(x)["output"] for x in (run / "dev" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert outputs == saved


def test_a_route_missing_a_seed_is_out(cfg, tmp_path):
    runs = tmp_path / "runs"
    choices = {}
    for arm in ("a-base", "b-base"):
        for lr in cfg["recipe"]["lr_grid"]:
            R.run_fit(cfg, arm, lr, 1, runs, CPU)
        choices[arm] = R.choose_lr(cfg, arm, runs)
        R.run_fit(cfg, arm, choices[arm]["chosen_lr"], 2, runs, CPU)
    routes = R.choose_routes(cfg, runs, choices)            # a-chat never ran, as when G2 fails
    assert [r["arm"] for r in routes["v2"]["routes"]] == ["a-base"]


def test_a_prompt_too_long_for_the_context_stops_the_run(cfg):
    ex = [{"task": "banking77", "id": "x", "user": "word " * 2000, "target": "y", "gold": "y"}]
    obj = R.load_arm({**cfg["arms"]["a-base"], "name": "a-base"}, CPU)
    with pytest.raises(SystemExit):
        R.predict(obj, ex, cfg["recipe"])


def test_output_tokens_alone_carry_loss_and_line_up_with_the_previous_position():
    ids, mask, tgt = R.micro_tensors([([5, 6, 7], [8, 9]), ([5], [8])], CPU)
    assert ids.tolist() == [[5, 6, 7, 8, 9], [5, 8, 0, 0, 0]] and mask.sum(1).tolist() == [5, 2]
    assert tgt.tolist() == [[-1, -1, 8, 9, -1], [8, -1, -1, -1, -1]]


def test_micro_batches_respect_the_budget_and_keep_every_row():
    lengths = [10, 200, 30, 90, 90, 5]
    m = R.split_micro(lengths, 200)
    assert sorted(i for x in m for i in x) == list(range(6))
    assert all(len(x) * max(lengths[i] for i in x) <= 200 or len(x) == 1 for x in m)


def test_micro_batch_split_does_not_change_the_update(cfg):
    """The registry allows micro-batches to differ between models only because the update is the same."""
    manifest = X.load_manifest(cfg["manifest"])
    train = R.build_examples(cfg["items_dir"], manifest, list(X.PRIMARY), "train")
    finals = []
    for budget in (10000, 128):
        torch.manual_seed(0)
        obj = R.load_arm({**cfg["arms"]["a-base"], "name": "a-base"}, CPU)
        R.fine_tune(obj, train, 1e-3, 1, {**cfg["recipe"], "micro_tokens": budget}, None, CPU)
        finals.append(obj.model.state_dict())
    for k in finals[0]:
        assert torch.allclose(finals[0][k], finals[1][k], rtol=1e-3, atol=1e-5), k


def test_learning_rate_schedule_of_the_recipe():
    assert R.lr_at(0, 100, 1.0) == pytest.approx(0.2)
    assert R.lr_at(4, 100, 1.0) == pytest.approx(1.0)
    assert R.lr_at(100, 100, 1.0) == pytest.approx(0.1)
    assert np.all(np.diff([R.lr_at(s, 100, 1.0) for s in range(5, 100)]) <= 0)
