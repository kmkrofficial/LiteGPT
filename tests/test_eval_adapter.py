"""Native lm-eval adapter: log-likelihoods equal direct model log-probabilities (EXP-M1B-SUITE adapter rule)."""
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import tokenizer as T  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.eval.lm_adapter import rolling_loglikelihood, score_continuations  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    tok = T.LiteTokenizer(T.train_bpe(iter(["the cat sat on the mat. the dog ran far away from home."] * 50), vocab_size=300))
    torch.manual_seed(0)
    cfg = get_config("litegpt-tiny", attention_backend="sdpa", vocab_size=tok.vocab_size)
    return LiteGPT(cfg).float().eval(), tok


def direct(model, tok, context, continuation):
    ctx, cont = tok.encode_text(context), tok.encode_text(context + continuation)[len(tok.encode_text(context)):]
    seq = [T.BOS] + ctx + cont
    with torch.no_grad():
        logits, _ = model(torch.tensor([seq[:-1]]), logits_all=True)
    lp = F.log_softmax(logits.float(), -1)[0]
    n = len(cont)
    return float(sum(lp[len(seq) - 1 - n + k, seq[len(seq) - n + k]] for k in range(n)))


def test_batched_scores_equal_direct_logprobs(setup):
    model, tok = setup
    pairs = [("the cat sat", " on the mat."), ("the dog", " ran far away"), ("home", "."), ("the cat sat on the mat. the", " dog")]
    got = score_continuations(model, tok, pairs, batch_size=3)
    for (c, x), (lp, greedy) in zip(pairs, got):
        assert abs(lp - direct(model, tok, c, x)) <= 1e-4 and isinstance(greedy, bool)


def test_rolling_matches_single_window_sum(setup):
    model, tok = setup
    text = "the cat sat on the mat. the dog ran far away from home."
    ids = [T.BOS] + tok.encode_text(text)
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids[:-1]]), logits_all=True)
    lp = F.log_softmax(logits.float(), -1)[0]
    expected = float(sum(lp[i, ids[i + 1]] for i in range(len(ids) - 1)))
    assert math.isclose(rolling_loglikelihood(model, tok, text), expected, abs_tol=1e-4)


def test_rolling_handles_text_longer_than_context(setup):
    model, tok = setup
    text = " ".join(["the cat sat on the mat."] * 80)
    n = len(tok.encode_text(text)) + 1
    assert n > model.config.block_size
    assert rolling_loglikelihood(model, tok, text) < 0
