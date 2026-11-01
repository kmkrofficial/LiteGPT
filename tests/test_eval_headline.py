"""Headline metric selection: plain metric (first filter) and explicit metric,filter (GSM8K chat uses flexible-extract)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt.eval.run import headline, load_suite, model_label  # noqa: E402

RESULTS = {
    "gsm8k_cot_zeroshot": {"alias": "gsm8k_cot_zeroshot", "exact_match,strict-match": 0.0, "exact_match_stderr,strict-match": 0.0,
                           "exact_match,flexible-extract": 0.0637, "exact_match_stderr,flexible-extract": 0.0067},
    "hellaswag": {"acc,none": 0.3, "acc_stderr,none": 0.004, "acc_norm,none": 0.313, "acc_norm_stderr,none": 0.005},
}


def test_explicit_filter_and_plain_metric():
    h = headline(RESULTS, {"gsm8k_cot_zeroshot": "exact_match,flexible-extract", "hellaswag": "acc_norm"})
    assert h["gsm8k_cot_zeroshot"]["value"] == 0.0637 and h["gsm8k_cot_zeroshot"]["stderr"] == 0.0067
    assert h["hellaswag"]["value"] == 0.313 and h["hellaswag"]["stderr"] == 0.005 and h["hellaswag"]["key"] == "acc_norm,none"


def test_missing_filter_gives_no_headline_instead_of_another_filter():
    assert headline(RESULTS, {"gsm8k_cot_zeroshot": "exact_match,nonexistent"}) == {}


def test_suite_reads_gsm8k_chat_with_flexible_extract():
    head = load_suite()["headline_metrics"]
    assert head["gsm8k_cot_zeroshot"] == "exact_match,flexible-extract" and head["gsm8k"] == "exact_match,strict-match"


def test_model_label_adds_the_requested_ref():
    base = {"kind": "hf", "repo": "EleutherAI/pythia-160m", "revision": "50f5173d932e8e61f858120bcb800b97af589f46"}
    assert model_label(base) == "pythia-160m"
    assert model_label({**base, "ref": "step5000"}) == "pythia-160m-step5000"
