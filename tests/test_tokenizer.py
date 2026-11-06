import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import tokenizer as T  # noqa: E402

CORPUS = ["def add(a, b):\n    return a + b  # 12345\n", "SELECT name FROM users WHERE id = 42;\n", "Hello, world! café 你好 \U0001f600",
          "The quick brown fox jumps over the lazy dog. 2024-10-11", "\tindented\tline\r\nwindows newline\r\n"] * 40


@pytest.fixture(scope="module")
def lt():
    return T.LiteTokenizer(T.train_bpe(iter(CORPUS), vocab_size=700))


def test_special_ids_are_fixed(lt):
    for i, name in enumerate(T.ALL_SPECIALS):
        assert lt.tok.token_to_id(name) == i
    assert (T.BOS, T.EOT, T.IM_START, T.IM_END, T.PAD) == (0, 1, 2, 3, 4)
    assert T.N_SPECIAL == 37


def test_round_trip_exact(lt):
    for text in CORPUS[:5] + ["", " ", "\n\n\n", "a" * 500, "é́ combining", "tabs\t\tand   spaces"]:
        ids = lt.encode_text(text)
        out = lt.decode(ids)
        assert out == text or out == __import__("unicodedata").normalize("NFC", text)


def test_byte_exact_round_trip_for_nfc_text(lt):
    text = "Mixed 你好 \U0001f600 é \x00 control"
    assert lt.decode(lt.encode_text(text)).encode("utf-8") == text.encode("utf-8")


def test_special_text_is_never_a_control_id(lt):
    ids = lt.encode_text("before <|endoftext|> <|im_start|> <|reserved_3|> after")
    assert not (set(ids) & set(range(T.N_SPECIAL)))
    assert lt.decode(ids) == "before <|endoftext|> <|im_start|> <|reserved_3|> after"


def test_document_has_bos_and_eot(lt):
    ids = lt.encode_document("hello")
    assert ids[0] == T.BOS and ids[-1] == T.EOT and T.BOS not in ids[1:-1]


def test_digits_are_single_tokens(lt):
    ids = lt.encode_text("2024")
    assert len(ids) == 4 and [lt.decode([i]) for i in ids] == ["2", "0", "2", "4"]


def test_token_byte_lengths_sum_to_utf8_length(lt):
    nb = lt.token_byte_lengths()
    assert nb[: T.N_SPECIAL].sum() == 0
    for text in CORPUS[:4]:
        ids = lt.encode_text(text)
        assert int(nb[np.array(ids)].sum()) == len(text.encode("utf-8"))


def test_save_load_hash_is_stable(lt, tmp_path):
    p = tmp_path / "tok.json"
    h = lt.save(p)
    again = T.LiteTokenizer.load(p)
    assert T.file_sha256(p) == h and again.encode_text(CORPUS[0]) == lt.encode_text(CORPUS[0])


def test_frozen_asset_matches_spec_hash():
    from litegpt.spec import load_spec
    spec = load_spec()["tokenizer"]
    assert T.DEFAULT_PATH.exists(), "frozen tokenizer asset missing"
    assert T.file_sha256(T.DEFAULT_PATH) == spec["sha256"]
    tok = T.LiteTokenizer.load()
    assert tok.vocab_size == spec["vocab_size"] == T.VOCAB_SIZE
    assert tok.decode(tok.encode_text("def f(x):\n    return x  # 2026\n")) == "def f(x):\n    return x  # 2026\n"
