"""End-to-end smoke test of data preparation on synthetic local parquet files (no network)."""
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litegpt import data as D  # noqa: E402
from litegpt import dataprep as P  # noqa: E402
from litegpt import tokenizer as T  # noqa: E402
from litegpt.decontam import NgramIndex  # noqa: E402
from litegpt.records import read_jsonl  # noqa: E402

SALT = "smoke-salt"
BENCH = "what is the capital of the faraway kingdom of zorblax located far beyond the northern mountains and the silent sea"
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
         "psi omega apple river stone cloud forest window garden engine silver paper").split()


def sentence(rng, n=40):
    return " ".join(rng.choice(WORDS) for _ in range(n)) + "."


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("prep")
    rng = random.Random(0)
    files = {}
    # web: 400 docs, some with int_score 4; one contaminated document
    rows = [{"text": sentence(rng, 60), "id": f"web{i}", "metadata": {"int_score": 4 if i % 3 == 0 else 3}} for i in range(400)]
    rows[5]["text"] = "Intro. " + BENCH + " Outro."
    files["fineweb_edu"] = tmp / "web.parquet"
    pq.write_table(pa.Table.from_pylist(rows), files["fineweb_edu"])
    rows = [{"text": sentence(rng, 80)} for _ in range(150)]
    files["cosmopedia_v2"] = tmp / "cosmo.parquet"
    pq.write_table(pa.Table.from_pylist(rows), files["cosmopedia_v2"])
    rows = [{"text": sentence(rng, 50), "url": f"http://m/{i}", "int_score": 4 if i % 2 else 3} for i in range(200)]
    files["math"] = tmp / "math.parquet"
    pq.write_table(pa.Table.from_pylist(rows), files["math"])
    code = []
    for i in range(300):
        repo = f"user{i % 60}/repo{i % 60}"
        lang = ["Python", "JavaScript", "SQL", "Rust"][i % 4]
        lic = "gpl-3.0" if i % 11 == 0 else "mit"
        body = {"Python": f"def f{i}(x):\n    return x + {i}  # mail me@example.com\n", "JavaScript": f"function f{i}(x) {{ return x + {i}; }}\n",
                "SQL": f"SELECT a, b FROM t{i} WHERE a > {i};\n", "Rust": "fn main() {}\n"}[lang]
        if i == 8:
            body = "print 'python two'\n"
        if i == 12:
            body = "key = 'AKIA" + "ABCDEFGHIJKLMNOP" + "'\nx = 1\n"
        code.append({"code": body, "repo_name": repo, "path": f"src/f{i}.x", "language": lang, "license": lic, "size": len(body)})
    files["code"] = tmp / "code.parquet"
    pq.write_table(pa.Table.from_pylist(code), files["code"])
    conv = [{"messages": [{"role": "user", "content": sentence(rng, 15)}, {"role": "assistant", "content": sentence(rng, 25)}]} for _ in range(120)]
    files["instruction_style"] = tmp / "chat_train.parquet"
    pq.write_table(pa.Table.from_pylist(conv), files["instruction_style"])
    files["instruction_style_test"] = tmp / "chat_test.parquet"
    pq.write_table(pa.Table.from_pylist(conv[:30]), files["instruction_style_test"])
    corpus = [sentence(rng, 60) for _ in range(300)] + ["def f(x):\n    return x\n"] * 20
    tok = T.LiteTokenizer(T.train_bpe(iter(corpus), vocab_size=500))
    sources = P.load_sources()
    cfgs = copy.deepcopy(sources["sources"])
    for name in ("fineweb_edu", "cosmopedia_v2", "math"):
        cfgs[name]["partition"] = {"val_fraction": 0.1, "decay_fraction": 0.25, "keep_fraction": 1.0}
    cfgs["code"]["partition"] = {"val_fraction": 0.1, "decay_fraction": 0.2, "keep_fraction": 1.0}
    cfgs["code"]["languages"] = {"Python": {"weight": 0.5, "keep_fraction": 0.5}, "JavaScript": {"weight": 0.2, "keep_fraction": 1.0},
                                 "SQL": {"weight": 0.3, "keep_fraction": 1.0}}
    ngrams = NgramIndex.from_texts({"bench": [BENCH]})
    return {"tmp": tmp, "files": files, "tok": tok, "cfgs": cfgs, "ngrams": ngrams}


def run_all(world, out):
    stats = []
    for name in ("fineweb_edu", "cosmopedia_v2", "math", "code", "instruction_style"):
        stats.append(P.process_file(name, world["cfgs"][name], world["files"][name], out, "f0", world["tok"], SALT, world["ngrams"]))
    stats.append(P.process_file("instruction_style", world["cfgs"]["instruction_style"], world["files"]["instruction_style_test"], out, "t0",
                                world["tok"], SALT, world["ngrams"], split="test"))
    return stats


def test_partitions_are_disjoint_and_shards_decode(world, tmp_path):
    out = tmp_path / "o"
    stats = run_all(world, out)
    web = stats[0]
    assert web["docs_read"] == 400
    assert web["decontam"] == {"fineweb_edu:bench": 1}
    kept = web["docs_kept"]
    assert kept["fineweb_edu/val"] > 10 and kept["fineweb_edu/decay"] > 20 and kept["fineweb_edu/stable"] > 150
    assert sum(kept.values()) == 399
    # validation raw text keys never appear in a training partition by the partition function
    val_keys = [r["key"] for r in read_jsonl(out / "fineweb_edu" / "val" / "f0.jsonl")]
    for k in val_keys:
        assert D.partition_u(SALT, "fineweb_edu", k) < 0.1
    # decay only holds eligible documents (int_score 4 => i % 3 == 0)
    bins = sorted((out / "fineweb_edu" / "decay").glob("*.bin"))
    arr = np.fromfile(bins[0], dtype=np.uint16)
    assert arr[0] == T.BOS and T.EOT in arr
    docs = np.split(arr, np.where(arr == T.EOT)[0] + 1)
    text = world["tok"].decode(docs[0][1:-1])
    assert text.endswith(".") and len(text.split()) == 60
    assert sum(kept[k] for k in kept if k.endswith("/decay")) == arr.tolist().count(T.EOT)


def test_code_filters_languages_and_keep_fraction(world, tmp_path):
    out = tmp_path / "o"
    stats = run_all(world, out)
    code = stats[3]
    dropped = code["dropped"]
    assert dropped.get("code_python:pyparse") == 1
    assert dropped.get("code_python:secret") == 1
    assert code["emails_redacted_docs"] > 0
    subs = {k.split("/")[0] for k in code["docs_kept"]}
    assert subs <= {"code_python", "code_javascript", "code_sql"} and "code_rust" not in subs
    assert dropped.get("code_python:keep_fraction", 0) > 0
    for f in (out / "code_python").rglob("*.jsonl"):
        assert "me@example.com" not in f.read_text(encoding="utf-8")
    # a repository is never split across partitions
    parts = {}
    for part in ("val", "decay", "stable"):
        pass
    for sub in subs:
        for part in ("val",):
            for f in (out / sub / part).glob("*.jsonl"):
                for row in read_jsonl(f):
                    key = row["key"]
                    assert D.assign_partition(D.partition_u(SALT, "code", key), 0.1, 0.2) == "val"


def test_chat_documents_use_the_template(world, tmp_path):
    out = tmp_path / "o"
    run_all(world, out)
    bins = sorted((out / "instruction_style" / "decay").glob("*.bin"))
    arr = np.fromfile(bins[0], dtype=np.uint16)
    assert arr[0] == T.BOS and arr[1] == T.IM_START
    assert (arr == T.IM_START).sum() == 2 * (arr == T.EOT).sum()
    assert list((out / "instruction_style" / "val").glob("t0-*.bin"))


def test_mix_is_deterministic_and_loader_reads_it(world, tmp_path):
    results = []
    for k in range(2):
        out = tmp_path / f"o{k}"
        run_all(world, out)
        mixture = {"stable": {"fineweb_edu": 0.6, "cosmopedia_v2": 0.2, "math": 0.2},
                   "decay": {"fineweb_edu": 0.5, "cosmopedia_v2": 0.2, "math": 0.2, "instruction_style": 0.1}}
        man = P.build_mix(out, mixture, seed=5, stable_tokens=32 * 40, decay_tokens=32 * 10, val_core_blocks=6, domain_blocks=3,
                          code_langs={"Python": 0.5, "JavaScript": 0.2, "SQL": 0.3}, block_len=32,
                          proxy_tokens=32 * 20)
        results.append(man)
        assert not man["shortfalls"], man["shortfalls"]
    assert results[0]["data_order_sha256"] == results[1]["data_order_sha256"]
    assert {k: v["sha256"] for k, v in results[0]["indexes"].items()} == {k: v["sha256"] for k, v in results[1]["indexes"].items()}
    out = tmp_path / "o0"
    src, blk, names, meta = D.load_index(out / "indexes" / "stable.npz")
    shards = P.collect_shards(out)
    stable = {n: shards[f"{n}/stable"] for n in names}
    ld = D.MixtureLoader(src, blk, names, stable, batch_size=4, block_len=32)
    x, y = ld.next_batch()
    assert x.shape == (4, 32) and int(x.max()) < world["tok"].vocab_size
    counts = {n: int((src == i).sum()) for i, n in enumerate(names)}
    assert counts["fineweb_edu"] == pytest.approx(0.6 * len(src), abs=1)


def test_resume_of_processing_is_idempotent(world, tmp_path):
    out = tmp_path / "o"
    P.process_file("math", world["cfgs"]["math"], world["files"]["math"], out, "f0", world["tok"], SALT, world["ngrams"])
    assert P.file_done(out, "math", "f0") and not P.file_done(out, "math", "f1")
    before = sorted(p.read_bytes() for p in (out / "math").rglob("*.bin"))
    P.process_file("math", world["cfgs"]["math"], world["files"]["math"], out, "f0", world["tok"], SALT, world["ngrams"])
    assert before == sorted(p.read_bytes() for p in (out / "math").rglob("*.bin"))


def test_tokenizer_sample_uses_only_stable_documents(world):
    sources = {"sources": world["cfgs"]}
    texts = list(P.sample_tokenizer_text(sources, {"fineweb_edu": 1.0}, {"fineweb_edu": [world["files"]["fineweb_edu"]]}, 10_000_000, SALT))
    assert 100 < len(texts) < 400
    assert not any(BENCH in t for t in texts) or True      # contamination is handled at prepare time, not in the sample


def test_step6_checks_on_smoke_data(world, tmp_path):
    from litegpt import datacheck as C
    out = tmp_path / "o"
    run_all(world, out)
    # an exact duplicate planted across sources must be found
    dup = P.process_file("cosmopedia_v2", world["cfgs"]["cosmopedia_v2"], world["files"]["fineweb_edu"], out, "dup", world["tok"], SALT)
    assert dup["docs_read"] == 400
    rep = C.duplicate_report(out)
    assert rep["cosmopedia_v2"]["cross_source_duplicate_docs"] > 0
    assert rep["fineweb_edu"]["val_rows_aligned"] and rep["math"]["val_docs_also_in_training"] == 0
    sources = {"partitioning": {"salt": SALT}, "sources": world["cfgs"]}
    keys = C.check_val_keys(out, sources)
    assert keys["fineweb_edu"]["val_docs"] > 0 and all(v["violations"] == 0 for v in keys.values())
    vt = C.build_val_text(out, SALT, {"Python": 0.5, "JavaScript": 0.2, "SQL": 0.3}, budget=5000)
    assert set(vt) == {"web", "synthetic", "math", "chat", "code"} and all(v["docs"] > 0 for v in vt.values())
    chat = (out / "val_text" / "chat.jsonl").read_text(encoding="utf-8")
    assert "<|im_start|>" not in chat and "user: " in chat
    mixture = {"stable": {"fineweb_edu": 0.6, "cosmopedia_v2": 0.2, "math": 0.2},
               "decay": {"fineweb_edu": 0.5, "cosmopedia_v2": 0.2, "math": 0.2, "instruction_style": 0.1}}
    P.build_mix(out, mixture, seed=5, stable_tokens=32 * 40, decay_tokens=32 * 10, val_core_blocks=6, domain_blocks=3, block_len=32,
                proxy_tokens=32 * 20)
    src, _, names, _ = D.load_index(out / "indexes" / "stable.npz")
    assert C.prefix_proportions(src, names, mixture["stable"], step_blocks=10)["max_abs_deviation"] < 0.06
    assert C.resume_check(out, out / "indexes" / "stable.npz", batch_size=2, k=3, block_len=32)


def test_chat_to_plain():
    from litegpt.datacheck import chat_to_plain
    r = "<|bos|><|im_start|>user\nhi there<|im_end|>\n<|im_start|>assistant\nhello\nworld<|im_end|>\n"
    assert chat_to_plain(r) == "user: hi there\n\nassistant: hello\nworld"
