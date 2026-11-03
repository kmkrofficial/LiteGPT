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


def test_loss_mask_covers_only_assistant_text_and_its_end(lt):
    ids, mask = C.render(lt, MSGS)
    assert ids == C.render_ids(lt, MSGS) and len(mask) == len(ids)
    assert [i for i, m in zip(ids, mask) if m] == lt.encode_text("hi") + [T.IM_END]
    multi = MSGS + [{"role": "user", "content": "how are you"}, {"role": "assistant", "content": "fine thanks"}]
    ids, mask = C.render(lt, multi)
    assert [i for i, m in zip(ids, mask) if m] == lt.encode_text("hi") + [T.IM_END] + lt.encode_text("fine thanks") + [T.IM_END]


def test_loss_mask_starts_right_after_the_generation_prompt(lt):
    prompt = C.render_ids(lt, MSGS[:2], add_generation_prompt=True)
    ids, mask = C.render(lt, MSGS)
    assert ids[:len(prompt)] == prompt and not any(mask[:len(prompt)]) and mask[len(prompt)] == 1


def test_assistant_text_starting_with_a_newline_is_refused():
    tok = T.LiteTokenizer.load()
    with pytest.raises(ValueError):
        C.render(tok, [{"role": "user", "content": "x"}, {"role": "assistant", "content": "\nhello"}])
    C.render(tok, [{"role": "user", "content": "\nx"}, {"role": "assistant", "content": " hello"}])


@pytest.mark.parametrize("gen", [False, True])
def test_jinja_twin_gives_identical_ids_on_the_real_tokenizer(gen):
    from jinja2 import Environment
    from tokenizers import Tokenizer
    lite = T.LiteTokenizer.load()
    plain = Tokenizer.from_file(str(lite.path))          # reads special-token strings as control tokens, as Hugging Face tools do
    template = Environment().from_string(C.JINJA_PATH.read_text(encoding="utf-8"))
    cases = [MSGS, MSGS[1:], [{"role": "user", "content": "Tab\there, été and 你好\n\n  spaced"}, {"role": "assistant", "content": '{"a": [1, null]}'},
                              {"role": "user", "content": "next"}, {"role": "assistant", "content": "done"}]]
    for msgs in cases:
        msgs = msgs[:-1] if gen else msgs
        text = template.render(messages=msgs, add_generation_prompt=gen)
        assert text == C.render_text(msgs, add_generation_prompt=gen)
        assert plain.encode(text, add_special_tokens=False).ids == C.render_ids(lite, msgs, add_generation_prompt=gen)
