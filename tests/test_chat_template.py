import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import chat_template as C  # noqa: E402
from litegpt import tokenizer as T  # noqa: E402


@pytest.fixture(scope="module")
def lt():
    corpus = ["system user assistant hello world how are you fine thanks\n"] * 50
    return T.LiteTokenizer(T.train_bpe(iter(corpus), vocab_size=400))


MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]


def test_render_text_matches_literal_template():
    expected = ("<|bos|><|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n"
                "<|im_start|>assistant\nhi<|im_end|>\n")
    assert C.render_text(MSGS) == expected
    assert C.render_text(MSGS[:2], add_generation_prompt=True).endswith("<|im_start|>assistant\n")


def test_render_ids_structure(lt):
    ids = C.render_ids(lt, MSGS)
    assert ids[0] == T.BOS
    assert ids.count(T.IM_START) == 3 and ids.count(T.IM_END) == 3
    gen = C.render_ids(lt, MSGS[:2], add_generation_prompt=True)
    assert gen[-1] != T.IM_END and gen.count(T.IM_START) == 3


def test_decoded_ids_equal_text_rendering(lt):
    ids = C.render_ids(lt, MSGS)
    assert lt.decode(ids) == C.render_text(MSGS)


def test_reserved_and_unknown_roles_rejected(lt):
    with pytest.raises(ValueError):
        C.render_ids(lt, [{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError):
        C.render_ids(lt, [{"role": "robot", "content": "x"}])
    with pytest.raises(ValueError):
        C.render_ids(lt, [])
