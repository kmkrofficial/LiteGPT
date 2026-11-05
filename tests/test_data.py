import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import data as D  # noqa: E402


def make_source(tmp, name, n_tokens, base, shard_tokens=5000):
    w = D.ShardWriter(tmp / name, shard_tokens=shard_tokens)
    ids = (np.arange(n_tokens) + base) % 60000
    for i in range(0, n_tokens, 777):
        w.add(ids[i:i + 777])
    files = w.close()
    return D.ShardSet([tmp / name / f["file"] for f in files]), ids, files


def test_partition_is_deterministic_and_fractions_hold():
    us = [D.partition_u("salt", "src", f"doc{i}") for i in range(20000)]
    assert us == [D.partition_u("salt", "src", f"doc{i}") for i in range(20000)]
    parts = [D.assign_partition(u, 0.01, 0.2) for u in us]
    assert abs(parts.count("val") / len(parts) - 0.01) < 0.004
    assert abs(parts.count("decay") / len(parts) - 0.2) < 0.02
    assert D.assign_partition(0.1, 0.01, 0.2, decay_eligible=False) == "stable"
    assert D.partition_u("salt", "a", "k") != D.partition_u("salt", "b", "k")


def test_shardset_reads_across_file_boundaries(tmp_path):
    ss, ids, files = make_source(tmp_path, "a", 23456, 7)
    assert len(files) > 3 and len(ss) == 23456
    for start, n in [(0, 10), (4995, 20), (4990, 12000), (23000, 456)]:
        assert np.array_equal(ss.read(start, n), ids[start:start + n].astype(np.uint16))
    with pytest.raises(IndexError):
        ss.read(23400, 100)


def test_block_count():
    assert D.block_count(2049) == 1 and D.block_count(2048) == 0 and D.block_count(5000) == 2


def test_index_proportions_exact_at_every_prefix():
    weights = {"web": 0.6, "code": 0.3, "math": 0.1}
    src, blk, names = D.build_index({"web": 10000, "code": 10000, "math": 10000}, weights, 5000, seed=1)
    counts = np.zeros(3)
    for i, s in enumerate(src, start=1):
        counts[s] += 1
        if i % 250 == 0:
            for si, n in enumerate(names):
                assert abs(counts[si] / i - weights[n]) < 0.002 + 1.5 / i
    for si in range(3):
        sel = blk[src == si]
        assert len(set(sel.tolist())) == len(sel)          # no block used twice


def test_index_rejects_short_source():
    with pytest.raises(ValueError):
        D.build_index({"a": 10, "b": 1000}, {"a": 0.5, "b": 0.5}, 100, seed=0)


def test_index_is_deterministic_and_hash_changes_with_seed():
    a = D.build_index({"a": 500, "b": 500}, {"a": 0.5, "b": 0.5}, 400, seed=3)
    b = D.build_index({"a": 500, "b": 500}, {"a": 0.5, "b": 0.5}, 400, seed=3)
    c = D.build_index({"a": 500, "b": 500}, {"a": 0.5, "b": 0.5}, 400, seed=4)
    assert D.index_sha256(*a) == D.index_sha256(*b) != D.index_sha256(*c)


def test_index_save_load_roundtrip(tmp_path):
    src, blk, names = D.build_index({"a": 50, "b": 50}, {"a": 0.5, "b": 0.5}, 40, seed=2)
    h = D.save_index(tmp_path / "x.npz", src, blk, names, {"note": "t"})
    s2, b2, n2, meta = D.load_index(tmp_path / "x.npz")
    assert D.index_sha256(s2, b2, n2) == h and meta == {"note": "t"}


def test_loader_batches_and_exact_resume(tmp_path):
    bl = 64
    shards, ids = {}, {}
    for k, name in enumerate(["a", "b"]):
        shards[name], ids[name], _ = make_source(tmp_path, name, 64 * 300 + 1, k * 1000, shard_tokens=4000)
    n_blocks = {n: D.block_count(len(s), bl) for n, s in shards.items()}
    src, blk, names = D.build_index(n_blocks, {"a": 0.5, "b": 0.5}, 200, seed=9)
    ref = D.MixtureLoader(src, blk, names, shards, batch_size=4, block_len=bl)
    batches = [ref.next_batch() for _ in range(20)]
    x, y = batches[0]
    assert x.shape == (4, bl) and y.shape == (4, bl) and np.array_equal(x[:, 1:].numpy(), y[:, :-1].numpy())
    b0 = names[int(src[0])]
    start = int(blk[0]) * bl
    assert np.array_equal(x[0].numpy(), ids[b0][start:start + bl])
    for k in (0, 7, 13):
        res = D.MixtureLoader(src, blk, names, shards, batch_size=4, block_len=bl)
        res.load_state_dict({"position": k * 4, "block_len": bl, "batch_size": 4})
        for j in range(k, 20):
            xr, yr = res.next_batch()
            assert np.array_equal(xr.numpy(), batches[j][0].numpy()) and np.array_equal(yr.numpy(), batches[j][1].numpy())


def test_double_length_blocks_for_context_extension(tmp_path):
    shards, ids, _ = make_source(tmp_path, "a", 4096 * 5 + 1, 0, shard_tokens=9000)
    nb = {"a": D.block_count(len(shards), 4096)}
    src, blk, names = D.build_index(nb, {"a": 1.0}, 4, seed=0)
    ld = D.MixtureLoader(src, blk, names, {"a": shards}, batch_size=2, block_len=4096)
    x, y = ld.next_batch()
    assert x.shape == (2, 4096)
    s = int(blk[0]) * 4096
    assert np.array_equal(x[0].numpy(), ids[s:s + 4096])


def test_bpb_accounting():
    nb = np.array([0, 1, 2, 3])
    assert D.target_bytes(np.array([1, 2, 3, 0]), nb) == 6
    assert D.bits_per_byte(np.log(2) * 100, 100) == pytest.approx(1.0)
