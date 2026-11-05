"""M4 chat SFT data: partition, masks, packing and layout, without a GPU."""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import chat_template as C  # noqa: E402
from litegpt import data as D  # noqa: E402
from litegpt import sft as S  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.dataprep import content_hash  # noqa: E402
from litegpt.decontam import NgramIndex  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402
from litegpt.tokenizer import IM_END, PAD, LiteTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return LiteTokenizer.load()


def chat(user, reply):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


def write_parquet(path, conversations):
    pq.write_table(pa.table({"messages": conversations}), path)


def prepared(tmp_path, tok, conversations):
    data = [(np.asarray(i, dtype=np.uint16) | (np.asarray(m, dtype=np.uint16) << 15)).tobytes() for i, m in (C.render(tok, c) for c in conversations)]
    S.write_split(tmp_path, "train", data)
    return S.Conversations(tmp_path, "train")


def test_reserved_split_excludes_exactly_the_m1_decay_partition(tmp_path, tok):
    convs = [chat(f"question number {i}", f"answer number {i}") for i in range(200)]
    write_parquet(tmp_path / "a.parquet", convs)
    r = S.process_file(tmp_path / "a.parquet", "smol-rewrite", None, "salt", 0.85, tok)
    decay = sum(D.partition_u("salt", "instruction_style", content_hash(c)) < 0.85 for c in convs)
    assert r["counts"]["decay_partition"] == decay and r["counts"]["kept"] == 200 - decay
    r = S.process_file(tmp_path / "a.parquet", "smol-magpie-ultra", None, "salt", 0.85, tok)
    assert r["counts"]["decay_partition"] == 0 and r["counts"]["kept"] == 200


def test_every_drop_rule_is_counted(tmp_path, tok):
    long = " ".join(["word"] * 5000)
    bench = "the quick brown fox jumps over the lazy dog near the old river bank today"
    convs = [chat("fine", "kept"), chat("x", "\nstarts with a newline"), chat(long, "too long"), [{"role": "user", "content": "no reply"}],
             [{"role": "tool", "content": "t"}, {"role": "assistant", "content": "x"}], chat(bench, "contaminated")]
    write_parquet(tmp_path / "a.parquet", convs)
    ix = NgramIndex.from_texts({"bench": [bench]})
    r = S.process_file(tmp_path / "a.parquet", "smol-magpie-ultra", ix, "s", 0.85, tok)
    assert r["decontam"] == {"bench": 1}
    c = r["counts"]
    assert (c["read"], c["kept"], c["assistant_starts_with_newline"], c["over_max_tokens"], c["no_assistant"], c["bad_roles"], c["decontam"]) == (6, 1, 1, 1, 1, 1, 1)


def test_prepared_tokens_keep_ids_and_mask(tmp_path, tok):
    conv = prepared(tmp_path, tok, [chat("hello there", "general kenobi")])
    a = np.asarray(conv.get(0))
    ids, mask = C.render(tok, chat("hello there", "general kenobi"))
    assert (a & (S.MASK_BIT - 1)).tolist() == ids and (a >= S.MASK_BIT).astype(int).tolist() == mask


def test_rows_hold_whole_conversations_and_each_one_once():
    lengths = np.random.default_rng(0).integers(10, 300, 1000)
    rows = S.pack_rows(lengths, 512, list(range(1000)))
    assert sorted(i for r in rows for i in r) == list(range(1000))
    assert all(sum(lengths[i] for i in r) <= 512 for r in rows)
    assert lengths.sum() / (len(rows) * 512) > 0.9
    with pytest.raises(ValueError):
        S.pack_rows([513], 512, [0])


def test_epoch_order_depends_only_on_seed_and_epoch():
    lengths = np.random.default_rng(0).integers(10, 300, 500)
    a = S.epoch_rows(lengths, 512, 1, 0, 4)
    assert a == S.epoch_rows(lengths, 512, 1, 0, 4) and a != S.epoch_rows(lengths, 512, 1, 1, 4) and a != S.epoch_rows(lengths, 512, 2, 0, 4)
    assert len(a) % 4 == 0


def test_batch_targets_are_only_assistant_tokens_and_never_cross_a_conversation(tmp_path, tok):
    conv = prepared(tmp_path, tok, [chat("first question", "first reply"), chat("second one", "second reply"), chat("third", "last")])
    T = 128
    b = S.make_batch(conv, [[0, 1], [2]], T, "cpu")
    assert b.x.shape == (2, T) and int(b.cu_seqlens[-1]) == 2 * T and T in b.cu_seqlens.tolist()
    n0, n2 = int(conv.lengths[0]), int(conv.lengths[2])
    assert b.y[0, :n0][b.y[0, :n0] >= 0].tolist() == tok.encode_text("first reply") + [IM_END]
    assert b.y[0, n0 - 1] == -1                                  # the last token of a conversation predicts nothing
    assert int(b.position_ids[0, n0]) == 0 and int(b.doc[0, n0]) == int(b.doc[0, n0 - 1]) + 1
    assert (b.x[1, n2:] == PAD).all() and (b.y[1, n2:] == -1).all()
    assert b.real_tokens == int(conv.lengths.sum()) and b.target_tokens == int((b.y >= 0).sum())


@pytest.mark.parametrize("layout", ["cu_seqlens", "doc_pos"])
def test_a_conversation_scores_the_same_whatever_shares_its_row(tmp_path, tok, layout):
    torch.manual_seed(0)
    model = LiteGPT(get_config("litegpt-tiny", vocab_size=32768, block_size=128, attention_backend="sdpa")).eval()
    conv = prepared(tmp_path, tok, [chat("first question", "first reply"), chat("second one", "second reply")])

    def loss_sum(rows):
        b = S.make_batch(conv, rows, 96, "cpu")
        kw = {"cu_seqlens": b.cu_seqlens} if layout == "cu_seqlens" else {"doc_pos": (b.doc, b.position_ids)}
        with torch.no_grad():
            _, loss = model(b.x, b.y, **kw)
        return float(loss) * b.target_tokens, b.target_tokens

    both, n = loss_sum([[0, 1]])
    a, na = loss_sum([[0]])
    c, nc = loss_sum([[1]])
    assert n == na + nc and both == pytest.approx(a + c, rel=1e-4)
