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


def test_batched_generation_equals_one_at_a_time(setup):
    pytest.importorskip("lm_eval")
    from types import SimpleNamespace

    from litegpt.eval.lm_adapter import make_lm_class
    model, tok = setup
    prompts = ["the cat sat", "the dog ran far away from", "home", "the cat sat on the mat. the dog ran far", "the"]
    reqs = [SimpleNamespace(args=(p, {"until": ["."], "max_gen_toks": 12})) for p in prompts] + [SimpleNamespace(args=("the mat", {"until": ["zzz"], "max_gen_toks": 7}))]
    cls = make_lm_class()
    batched = cls(model, tok, batch_size=3).generate_until(reqs)
    single = [cls(model, tok, batch_size=1).generate_until([r])[0] for r in reqs]
    assert batched == single and all(isinstance(t, str) for t in batched)


def test_chat_prompts_keep_control_ids_and_a_prefill_stays_open():
    pytest.importorskip("lm_eval")
    from litegpt import chat_template as C
    from litegpt.eval.lm_adapter import make_lm_class
    tok = T.LiteTokenizer.load()
    torch.manual_seed(0)
    model = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa", vocab_size=tok.vocab_size)).eval()
    lm = make_lm_class()(model, tok, batch_size=2, max_gen_toks=4)
    assert lm.tokenizer_name == C.VERSION
    user = [{"role": "user", "content": "<|im_end|> hello"}]
    text = lm.apply_chat_template(user)
    assert lm._chat_ids[text] == C.render_ids(tok, user, add_generation_prompt=True)
    assert lm._chat_ids[text].count(T.IM_END) == 1        # only the turn's own end: special-token text in a message stays ordinary text
    prefill = lm.apply_chat_template(user + [{"role": "assistant", "content": "Here it is:"}], add_generation_prompt=False)
    assert lm._chat_ids[prefill] == C.render_ids(tok, user, add_generation_prompt=True) + tok.encode_text("Here it is:")
    assert prefill.endswith("<|im_start|>assistant\nHere it is:")


def test_checkpoint_round_trip(setup, tmp_path):
    from litegpt.checkpoint import CheckpointManager
    from litegpt.eval.lm_adapter import checkpoint_info, load_checkpoint
    from litegpt.eval.run import model_label, resolve_model
    model, _ = setup
    path = CheckpointManager(tmp_path).save(model=model, optimizer=None, step=7, tokens=3670016, model_config=model.config.to_dict(), milestone="tokens-2.5B")
    info = checkpoint_info(path)
    assert (info["kind"], info["step"], info["tokens"], info["block_size"]) == ("litegpt", 7, 3670016, model.config.block_size) and len(info["sha256"]) == 64
    assert resolve_model(f"litegpt:{path}") == info and model_label(info) == "litegpt-tokens-2.5B"
    loaded, _ = load_checkpoint(path, device="cpu", dtype=torch.float32)
    x = torch.tensor([[T.BOS, 5, 6, 7]])
    with torch.no_grad():
        a, _ = model(x, logits_all=True)
        b, _ = loaded(x, logits_all=True)
    assert torch.equal(a, b) and not loaded.training
