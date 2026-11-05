import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt.config import get_config  # noqa: E402
from litegpt.inference import Engine, GenerationConfig  # noqa: E402
from litegpt.inference import sampling as S  # noqa: E402
from litegpt.inference.stopping import StreamDecoder, find_stop, holdback  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa")).float().eval()
    return m


def greedy_reference(model, prompt, n):
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(torch.tensor([ids]))
            ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt):]


def test_cached_greedy_equals_full_forward(model):
    eng = Engine(model)
    prompt = [7, 8, 9, 10, 11]
    res = eng.generate_ids([prompt], GenerationConfig(max_new_tokens=12, stop_token_ids=()))[0]
    assert res.ids == greedy_reference(model, prompt, 12) and res.finish_reason == "length"


def test_batched_left_padding_equals_unbatched(model):
    eng = Engine(model)
    prompts = [[5, 6, 7], [9, 10, 11, 12, 13, 14, 15], [20]]
    cfg = GenerationConfig(max_new_tokens=8, stop_token_ids=())
    batched = eng.generate_ids(prompts, cfg)
    for p, r in zip(prompts, batched):
        assert r.ids == eng.generate_ids([p], cfg)[0].ids


def test_stop_token_sets_finish_reason(model):
    eng = Engine(model)
    prompt = [5, 6, 7]
    first = greedy_reference(model, prompt, 1)[0]
    res = eng.generate_ids([prompt], GenerationConfig(max_new_tokens=5, stop_token_ids=(first,)))[0]
    assert res.ids == [] and res.finish_reason == "stop"


def test_sampling_is_seeded(model):
    eng = Engine(model)
    cfg = GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=50, seed=7, stop_token_ids=())
    a = eng.generate_ids([[3, 4, 5]], cfg)[0].ids
    b = eng.generate_ids([[3, 4, 5]], cfg)[0].ids
    assert a == b


def test_prompt_too_long_is_rejected(model):
    eng = Engine(model)
    with pytest.raises(ValueError):
        eng.generate_ids([[1] * model.config.block_size], GenerationConfig(max_new_tokens=1))


def test_processors():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.isinf(S.apply_top_k(logits, 2)[0, :2]).all() and not torch.isinf(S.apply_top_k(logits, 2)[0, 2:]).any()
    p = torch.softmax(logits, -1)[0]
    kept = ~torch.isinf(S.apply_top_p(logits, 0.7)[0])           # top token alone has 0.644 < 0.7
    assert kept.tolist() == [False, False, True, True] and float(p[2:].sum()) >= 0.7
    assert (~torch.isinf(S.apply_min_p(logits, 0.3)[0])).tolist() == [False, False, True, True]
    pen = S.apply_penalties(logits, [[3, 3]], repetition=2.0, frequency=0.5, presence=0.25)
    assert pen[0, 3] == pytest.approx(4.0 / 2 - 0.5 * 2 - 0.25) and pen[0, 0] == 1.0


def test_stop_string_helpers():
    assert find_stop("abc###def", ["###"]) == 3 and find_stop("abc", ["x"]) is None
    assert holdback("hello ##", ["###"]) == 2 and holdback("hello", ["###"]) == 0


def test_stream_decoder_handles_multibyte_and_stop_strings():
    text = "café \U0001f600 ok###tail"
    data = list(text.encode("utf-8"))

    def decode(ids):
        return bytes(ids).decode("utf-8", errors="replace")

    dec = StreamDecoder(decode, ["###"])
    out = ""
    for b in data:
        delta = dec.push(b)
        assert "�" not in delta
        out += delta
        if dec.stopped:
            break
    out += dec.flush()
    assert out == "café \U0001f600 ok"
