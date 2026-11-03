"""The text-to-data comparison (EXP-M4-T2D): items, prompts, batches and scores. Nothing here depends on which model is being scored.

Scores follow the registry. Text is compared after lowercasing, removing punctuation and collapsing whitespace. Invalid JSON means every
gold value of that output is missed, and validity is reported separately. An absent gold field counts only if the model invents a value.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

PRIMARY = ("cuad", "massive", "banking77")
CUAD_FIELDS = ("document_name", "parties", "agreement_date", "governing_law")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})
ANSWER_CUE = "\n\nAnswer:\n"      # base models have no chat template, so this follows the same user text; chat arms use their own template

INSTRUCTIONS = {
    "cuad": "Extract these fields from the contract as one JSON object with the keys document_name, parties, agreement_date and governing_law. parties and governing_law are lists of strings. Use null for a field the contract text does not state.",
    "massive": "Extract the slots from this request as a JSON list of objects with the keys type and value. Use an empty list if there are no slots.",
    "banking77": "Classify this banking customer query. Reply with the intent label only.",
    "billsum": "Summarize this bill of the US Congress.",
}


def load_manifest(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_items(items_dir, manifest: dict, task: str, split: str) -> list[dict]:
    """Items of one split, refused unless the bytes match the committed hash."""
    p = Path(items_dir) / f"{task}_{split}.jsonl"
    raw = p.read_bytes()
    want = manifest["tasks"][task][split]
    if hashlib.sha256(raw).hexdigest() != want["sha256"]:
        raise SystemExit(f"{p} does not match the committed hash")
    items = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    if len(items) != want["items"]:
        raise SystemExit(f"{p}: {len(items)} items, expected {want['items']}")
    return items


def target_text(task: str, gold) -> str:
    """The exact bytes every model learns to write."""
    if task == "cuad":
        return json.dumps({k: gold[k] for k in CUAD_FIELDS}, ensure_ascii=False)
    if task == "massive":
        return json.dumps(gold, ensure_ascii=False)
    return gold


def user_text(task: str, item: dict) -> str:
    return f"{INSTRUCTIONS[task]}\n\n{item['input']}"


def manifest_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_batches(examples: list[dict], seed: int, epoch: int, max_examples: int = 32, max_chars: int = 16000) -> list[list[int]]:
    """Batches of indices for one epoch. Limits are in characters, so every model gets the same batches whatever its tokenizer does."""
    order = list(range(len(examples)))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    out, cur, chars = [], [], 0
    for i in order:
        n = len(examples[i]["user"]) + len(examples[i]["target"])
        if cur and (len(cur) >= max_examples or chars + n > max_chars):
            out.append(cur)
            cur, chars = [], 0
        cur.append(i)
        chars += n
    if cur:
        out.append(cur)
    return out


def norm(text) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", t).strip()


def _as_values(v) -> list[str]:
    """A field's values, normalised. Null, empty and blank values are absent."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _as_values(e)]
    if isinstance(v, (dict, bool)):
        return []
    n = norm(v)
    return [n] if n else []


def _first_json(text: str):
    """The output parsed as JSON. Trailing text after a complete value is tolerated; anything else is invalid."""
    try:
        value, _ = json.JSONDecoder().raw_decode(text.strip())
        return value
    except ValueError:
        return None


def _prf_counts(pred: Counter, gold: Counter) -> tuple[int, int, int]:
    tp = sum((pred & gold).values())
    return tp, sum(pred.values()) - tp, sum(gold.values()) - tp


def score_cuad(output: str, gold: dict) -> dict:
    data = _first_json(output)
    valid = isinstance(data, dict)
    tp = fp = fn = 0
    per = {}
    for f in CUAD_FIELDS:
        p = Counter(set(_as_values(data.get(f)))) if valid else Counter()
        c = _prf_counts(p, Counter(set(_as_values(gold.get(f)))))
        per[f] = c
        tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
    return {"tp": tp, "fp": fp, "fn": fn, "valid": valid, "per_field": per}


def score_massive(output: str, gold: list) -> dict:
    data = _first_json(output)
    valid = isinstance(data, list) and all(isinstance(x, dict) and "type" in x and "value" in x for x in data)
    p = Counter((norm(x["type"]), norm(x["value"])) for x in data) if valid else Counter()
    tp, fp, fn = _prf_counts(p, Counter((norm(x["type"]), norm(x["value"])) for x in gold))
    return {"tp": tp, "fp": fp, "fn": fn, "valid": valid}


def score_banking(output: str, gold: str) -> dict:
    return {"correct": int(norm(output) == norm(gold))}


def score_item(task: str, output: str, gold) -> dict:
    return {"cuad": score_cuad, "massive": score_massive, "banking77": score_banking}[task](output, gold)


def f1(tp: float, fp: float, fn: float) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def task_arrays(task: str, scores: list[dict]) -> np.ndarray:
    """(items, 3) of tp, fp, fn for F1 tasks; (items, 1) of correct for accuracy."""
    if task == "banking77":
        return np.array([[s["correct"]] for s in scores], dtype=np.float64).reshape(-1, 1)
    return np.array([[s["tp"], s["fp"], s["fn"]] for s in scores], dtype=np.float64).reshape(-1, 3)


def task_value(a: np.ndarray) -> float:
    if a.shape[1] == 1:
        return float(a.mean()) if len(a) else 0.0
    return f1(*a.sum(axis=0))


def summarize(per_task: dict[str, list[dict]]) -> dict:
    """The registered score: plain mean of CUAD field micro-F1, MASSIVE slot micro-F1 and Banking77 accuracy."""
    out = {t: task_value(task_arrays(t, per_task[t])) for t in PRIMARY}
    out["score"] = sum(out[t] for t in PRIMARY) / len(PRIMARY)
    for t in ("cuad", "massive"):
        out[f"{t}_valid_json"] = float(np.mean([s["valid"] for s in per_task[t]])) if per_task[t] else 0.0
    out["cuad_per_field"] = {f: f1(*np.sum([s["per_field"][f] for s in per_task["cuad"]], axis=0)) for f in CUAD_FIELDS} if per_task["cuad"] else {}
    return out


def _bootstrap_values(arrays: dict[str, np.ndarray], picks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for t, a in arrays.items():
        s = a[picks[t]].sum(axis=1)                  # (rounds, columns)
        if a.shape[1] == 1:
            out[t] = s[:, 0] / a.shape[0]
        else:
            d = 2 * s[:, 0] + s[:, 1] + s[:, 2]
            out[t] = np.where(d > 0, 2 * s[:, 0] / np.maximum(d, 1e-12), 0.0)
    return out


def compare(a: dict[str, list[list[dict]]], b: dict[str, list[list[dict]]], rounds: int = 10000, seed: int = 20261103, margin_se: float = 2.0) -> dict:
    """A against B: seeds averaged item by item, then a paired bootstrap over test items within each task. Win or loss needs more than
    `margin_se` standard errors of the difference; otherwise tie. Also per task."""
    rng = np.random.default_rng(seed)
    aa = {t: np.mean([task_arrays(t, s) for s in a[t]], axis=0) for t in PRIMARY}
    bb = {t: np.mean([task_arrays(t, s) for s in b[t]], axis=0) for t in PRIMARY}
    picks = {t: rng.integers(0, len(aa[t]), size=(rounds, len(aa[t]))) for t in PRIMARY}
    va, vb = _bootstrap_values(aa, picks), _bootstrap_values(bb, picks)

    def verdict(diff: float, se: float) -> str:
        return "win" if diff > margin_se * se else "loss" if diff < -margin_se * se else "tie"

    res = {}
    for t in PRIMARY:
        d, se = task_value(aa[t]) - task_value(bb[t]), float(np.std(va[t] - vb[t], ddof=1))
        res[t] = {"a": task_value(aa[t]), "b": task_value(bb[t]), "diff": d, "se": se, "verdict": verdict(d, se)}
    sa, sb = sum(res[t]["a"] for t in PRIMARY) / 3, sum(res[t]["b"] for t in PRIMARY) / 3
    boot = sum(va[t] - vb[t] for t in PRIMARY) / 3
    se = float(np.std(boot, ddof=1))
    res["score"] = {"a": sa, "b": sb, "diff": sa - sb, "se": se, "verdict": verdict(sa - sb, se)}
    return {**res, "rounds": rounds, "seed": seed, "margin_se": margin_se}


def rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(reference, prediction)["rougeL"].fmeasure
