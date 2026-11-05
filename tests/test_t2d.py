"""M4 text-to-data scoring against hand-worked examples, and the decision rule. No model is involved."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from litegpt import t2d as X  # noqa: E402

GOLD = {"document_name": "Service Agreement", "parties": ["Acme Inc", "Beta LLC"], "agreement_date": None, "governing_law": ["New York"]}


def test_normalisation_lowercases_removes_punctuation_and_collapses_space():
    assert X.norm("  Acme,   INC.\n") == "acme inc"
    assert X.norm("O'Neil & Sons") == "o neil sons"


def test_cuad_perfect_output_after_normalisation():
    out = '{"document_name": "SERVICE AGREEMENT.", "parties": ["acme inc", "BETA, LLC"], "agreement_date": null, "governing_law": ["new york"]}'
    s = X.score_cuad(out, GOLD)
    assert (s["tp"], s["fp"], s["fn"], s["valid"]) == (4, 0, 0, True)


def test_cuad_hand_worked_partial_output():
    # document name right; one party right, one wrong, one missed; invented date; law wrong
    out = '{"document_name": "Service Agreement", "parties": ["Acme Inc", "Gamma"], "agreement_date": "1/1/2020", "governing_law": ["Texas"]}'
    s = X.score_cuad(out, GOLD)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 3, 2)
    assert X.f1(2, 3, 2) == pytest.approx(4 / 9)
    assert s["per_field"]["agreement_date"] == (0, 1, 0)


def test_cuad_invalid_json_misses_every_gold_value():
    for out in ("not json", '{"document_name": "Service', "[1, 2]", ""):
        s = X.score_cuad(out, GOLD)
        assert (s["tp"], s["fp"], s["fn"], s["valid"]) == (0, 0, 4, False)


def test_cuad_correct_null_earns_nothing_and_costs_nothing():
    gold = {"document_name": "A", "parties": [], "agreement_date": None, "governing_law": None}
    s = X.score_cuad('{"document_name": "A", "parties": [], "agreement_date": null, "governing_law": null}', gold)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)


def test_massive_scores_type_and_value_together_and_counts_repeats():
    gold = [{"type": "date", "value": "today"}, {"type": "person", "value": "Ann"}, {"type": "date", "value": "today"}]
    s = X.score_massive('[{"type": "date", "value": "Today"}, {"type": "place", "value": "Ann"}]', gold)
    assert (s["tp"], s["fp"], s["fn"], s["valid"]) == (1, 1, 2, True)
    assert X.score_massive("[]", [])["valid"] and (X.score_massive("[]", [])["tp"], X.score_massive("[]", [])["fp"]) == (0, 0)
    bad = X.score_massive('[{"type": "date"}]', gold)
    assert (bad["tp"], bad["fp"], bad["fn"], bad["valid"]) == (0, 0, 3, False)


def test_banking_accepts_only_the_exact_label():
    assert X.score_banking(" Card_Arrival\n", "card_arrival")["correct"] == 1
    assert X.score_banking("the label is card_arrival", "card_arrival")["correct"] == 0


def per_task(cuad_tp=4, massive_tp=1, correct=1, n=20):
    return {"cuad": [{"tp": cuad_tp, "fp": 4 - cuad_tp, "fn": 4 - cuad_tp, "valid": True, "per_field": {f: (1, 0, 0) for f in X.CUAD_FIELDS}} for _ in range(n)],
            "massive": [{"tp": massive_tp, "fp": 1 - massive_tp, "fn": 1 - massive_tp, "valid": True} for _ in range(n)],
            "banking77": [{"correct": correct} for _ in range(n)]}


def test_score_is_the_plain_mean_of_the_three_tasks():
    s = X.summarize(per_task(cuad_tp=4, massive_tp=0, correct=1))
    assert (s["cuad"], s["massive"], s["banking77"]) == (1.0, 0.0, 1.0)
    assert s["score"] == pytest.approx(2 / 3)


def mixed(rate, n=300, seed=0):
    import numpy as np
    r = np.random.default_rng(seed).random(n) < rate
    return [{"correct": int(v)} for v in r]


def seeded(rates):
    return {"cuad": [[{"tp": 1, "fp": 0, "fn": 0}] * 30], "massive": [[{"tp": 1, "fp": 0, "fn": 0}] * 30], "banking77": [mixed(r, seed=i) for i, r in enumerate(rates)]}


def test_clear_difference_wins_and_loses_and_noise_ties():
    high, low = seeded([0.95, 0.95]), seeded([0.05, 0.05])
    assert X.compare(high, low)["banking77"]["verdict"] == "win"
    assert X.compare(low, high)["banking77"]["verdict"] == "loss"
    same = seeded([0.5, 0.5])
    r = X.compare(same, same)
    assert r["score"]["diff"] == 0 and r["score"]["verdict"] == "tie"


def test_comparison_is_repeatable_and_antisymmetric():
    a, b = seeded([0.6, 0.62]), seeded([0.55, 0.5])
    r, r2 = X.compare(a, b), X.compare(a, b)
    assert r == r2
    assert X.compare(b, a)["score"]["diff"] == pytest.approx(-r["score"]["diff"])


def test_batches_are_the_same_for_every_model_and_respect_both_limits():
    examples = [{"user": "x" * (10 + 53 * i % 5000), "target": "y"} for i in range(500)]
    a = X.make_batches(examples, 1, 0)
    assert a == X.make_batches(examples, 1, 0) and a != X.make_batches(examples, 1, 1) and a != X.make_batches(examples, 2, 0)
    assert sorted(i for b in a for i in b) == list(range(500))
    for b in a:
        assert len(b) <= 32 and (len(b) == 1 or sum(len(examples[i]["user"]) + 1 for i in b) <= 16000)


def test_items_are_refused_if_they_do_not_match_the_hash(tmp_path):
    p = tmp_path / "cuad_test.jsonl"
    p.write_text('{"id": "a", "input": "x", "gold": {}}\n', encoding="utf-8")
    m = {"tasks": {"cuad": {"test": {"items": 1, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}}}}
    assert X.load_items(tmp_path, m, "cuad", "test")[0]["id"] == "a"
    p.write_text('{"id": "b", "input": "x", "gold": {}}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        X.load_items(tmp_path, m, "cuad", "test")


def test_committed_manifest_matches_the_registered_test_sets():
    m = json.loads((ROOT / "benchmarks" / "m4" / "t2d_manifest.json").read_text(encoding="utf-8"))
    sizes = {t: m["tasks"][t]["test"]["items"] for t in m["tasks"]}
    assert sizes == {"cuad": 98, "massive": 2973, "banking77": 3080, "billsum": 1000}
    assert m["tasks"]["cuad"]["test"]["zero_match_items"] == 42 and m["tasks"]["billsum"]["test"]["removed_for_context_ids"] == ["test1834"]
    rule = m["overlap_rule"]["remove_if"]
    assert rule == {"longest_run_words_at_least": 30, "coverage_at_least": 0.10}


def test_rouge_l_is_one_for_identical_text():
    assert X.rouge_l("the bill funds schools", "the bill funds schools") == pytest.approx(1.0)
    assert X.rouge_l("nothing alike", "the bill funds schools") == 0.0
