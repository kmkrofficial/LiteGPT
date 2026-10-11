import copy
import json

import pytest

from litegpt import records
from litegpt.spec import spec_sha256


@pytest.fixture()
def run(tmp_path):
    return records.create_run("test", "unit", config={"a": 1}, root=tmp_path)


def test_create_run_writes_valid_record(run):
    rec = records.load_run(run.path)
    assert records.validate_run_record(rec) == []
    assert rec["spec_sha256"] == spec_sha256()
    env = rec["environment"]
    assert env["python"].startswith("3.12")
    assert env["uv_lock_sha256"] and len(env["uv_lock_sha256"]) == 64
    assert env["resolved_package_versions"]["torch"].startswith("2.14.1")
    assert env["torch_version"].startswith("2.14.1")
    assert rec["run_id"] == run.run_id and (run.path / "run.json").exists()


def test_gpu_fields_required_when_cuda_available(run):
    rec = records.load_run(run.path)
    if rec["environment"]["cuda_available"]:
        for k in ("gpu_name", "gpu_vram_gib", "cudnn_version", "nvidia_driver_version", "torch_cuda_version"):
            assert rec["environment"][k] is not None, k


@pytest.mark.parametrize("field", records.RUN_REQUIRED)
def test_missing_top_level_field_is_invalid(run, field):
    rec = copy.deepcopy(records.load_run(run.path))
    del rec[field]
    assert any(field in p for p in records.validate_run_record(rec))


@pytest.mark.parametrize("field", ["python", "uv_version", "uv_lock_sha256", "resolved_package_versions",
                                   "torch_version", "os", "os_version", "flash_attn_version_or_none", "cuda_available"])
def test_missing_environment_field_is_invalid(run, field):
    rec = copy.deepcopy(records.load_run(run.path))
    del rec["environment"][field]
    assert any(field in p for p in records.validate_run_record(rec))


def test_unexpected_null_is_invalid(run):
    rec = copy.deepcopy(records.load_run(run.path))
    rec["environment"]["uv_lock_sha256"] = None
    assert records.validate_run_record(rec)
    rec = copy.deepcopy(records.load_run(run.path))
    rec["environment"]["flash_attn_version_or_none"] = None  # explicitly allowed
    assert records.validate_run_record(rec) == []


def test_gpu_nulls_only_allowed_without_cuda(run):
    rec = copy.deepcopy(records.load_run(run.path))
    rec["environment"]["cuda_available"] = True
    rec["environment"]["gpu_name"] = None
    assert any("gpu_name" in p for p in records.validate_run_record(rec))
    rec["environment"]["cuda_available"] = False
    assert all("gpu_name" not in p for p in records.validate_run_record(rec))


def test_bad_values_are_invalid(run):
    rec = copy.deepcopy(records.load_run(run.path))
    rec["attention_backend"] = "auto"  # must be the *resolved* backend
    rec["spec_sha256"] = "xyz"
    rec["compile_enabled"] = "no"
    problems = records.validate_run_record(rec)
    assert len(problems) >= 3


def test_jsonl_and_atomic_json(run):
    run.log("metrics", step=1, loss=2.5)
    run.log("metrics", step=2, loss=2.25)
    rows = records.read_jsonl(run.path / "metrics.jsonl")
    assert [r["step"] for r in rows] == [1, 2] and all("t" in r for r in rows)
    p = run.write_json("bench/x.json", {"tok_per_s": 123.0})
    assert json.loads(p.read_text())["tok_per_s"] == 123.0 and not list(run.path.rglob("*.tmp"))
    run.update(status="done")
    assert records.load_run(run.path)["status"] == "done"


def test_create_run_rejects_unknown_kind(tmp_path):
    with pytest.raises(AssertionError):
        records.create_run("nonsense", "x", root=tmp_path)


def test_read_jsonl_keeps_unicode_line_separators(tmp_path):
    import json as _json
    from litegpt.records import read_jsonl
    rows = [{"text": "a" + chr(0x2028) + "b" + chr(0x85) + "c" + chr(0x2029) + "d"}, {"text": "plain"}]
    p = tmp_path / "x.jsonl"
    p.write_text("".join(_json.dumps(r, ensure_ascii=False) + chr(10) for r in rows), encoding="utf-8")
    assert read_jsonl(p) == rows
