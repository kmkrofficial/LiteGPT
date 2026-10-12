import json
from pathlib import Path

import pytest

from litegpt import records
from scripts import build_report as rb


def _fake_bench_run(runs_dir: Path) -> Path:
    rec = records.create_run("bench", "unit-train_mem", root=runs_dir)
    rec.write_json("bench/train_mem.json", {
        "required_tok_per_s": 16500, "matmul_ceiling": {"n": 8192, "dtype": "bfloat16", "tflops": 76.5},
        "results": [
            {"config": "litegpt-v2-28l", "micro_batch": 2, "seq_len": 2048, "backend": "sdpa", "compile": False,
             "status": "ok", "tok_per_s": 20000.0, "tflops_full": 34.7, "tflops_causal": 29.4,
             "mfu_full_vs_matmul_ceiling": 0.454, "peak_alloc_gib": 8.34, "meets_16_5k_tok_per_s": True},
            {"config": "litegpt-v2-28l", "micro_batch": 4, "seq_len": 2048, "backend": "sdpa", "compile": True,
             "status": "compile_error", "error": "RuntimeError: Cannot find a working triton installation",
             "peak_alloc_gib": 0.1, "meets_16_5k_tok_per_s": False},
        ]})
    return rec.path


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Point the builder at a temporary runs/ directory (inputs and the generated report both live there)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rb, "RUNS_DIR", runs)
    monkeypatch.setattr(rb, "OUTPUT_DIR", runs / "report-latest")
    return runs


def test_build_reads_runs_directly_and_lists_inputs(sandbox, tmp_path):
    run_dir = _fake_bench_run(sandbox)
    out = tmp_path / "out"
    rb.build(out)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "201.37M" in text and "176.21M" in text          # introspected from the model, not typed
    assert "20,000" in text and "compile_error" in text and "triton" in text
    assert f"`runs/{run_dir.name}/run.json`" in text and f"`runs/{run_dir.name}/bench/train_mem.json`" in text
    assert (out / "figures" / "params.svg").exists()


def test_build_is_deterministic(sandbox, tmp_path):
    _fake_bench_run(sandbox)
    a, b = tmp_path / "a", tmp_path / "b"
    rb.build(a)
    rb.build(b)
    assert rb._tree(a) == rb._tree(b), "report build must be bit-identical for identical inputs"


def test_invalid_run_record_is_rejected(sandbox, tmp_path):
    run_dir = _fake_bench_run(sandbox)
    run = json.loads((run_dir / "run.json").read_text())
    del run["environment"]["uv_lock_sha256"]
    (run_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="invalid run record"):
        rb.build(tmp_path / "out")


def test_generated_report_dir_is_not_mistaken_for_a_run(sandbox):
    _fake_bench_run(sandbox)
    assert rb.main([]) == 0                       # writes runs/report-latest/ inside the runs directory
    assert (sandbox / "report-latest" / "REPORT.md").exists()
    assert not (sandbox / "report-latest" / "run.json").exists()
    assert len(rb.load_bench_runs()) == 1         # the report directory is not picked up as an input run


def test_check_detects_stale_report(sandbox):
    _fake_bench_run(sandbox)
    assert rb.main(["--check"]) == 1          # nothing generated yet -> stale
    assert rb.main([]) == 0
    assert rb.main(["--check"]) == 0
    (sandbox / "report-latest" / "REPORT.md").write_text("tampered", encoding="utf-8")
    assert rb.main(["--check"]) == 1


def test_refuses_to_delete_a_non_report_directory(sandbox, monkeypatch):
    _fake_bench_run(sandbox)
    victim = sandbox / "important-run"
    victim.mkdir()
    monkeypatch.setattr(rb, "OUTPUT_DIR", victim)
    with pytest.raises(AssertionError, match="refusing to delete"):
        rb.main([])
    assert victim.exists()


def test_report_shows_ab_summary_and_compile_equivalence(sandbox, tmp_path):
    run_dir = _fake_bench_run(sandbox)
    bench = json.loads((run_dir / "bench" / "train_mem.json").read_text())
    bench["ab_summary"] = {
        "A:sdpa": {"n": 4, "tok_per_s_mean": 18560.6, "tok_per_s_min": 18467.1, "tok_per_s_max": 18753.8,
                   "peak_alloc_gib_max": 11.17, "warmup_s_mean": 4.1, "final_loss_mean": 7.05},
        "B:sdpa+compile": {"n": 4, "tok_per_s_mean": 22996.6, "tok_per_s_min": 22822.3, "tok_per_s_max": 23247.4,
                           "peak_alloc_gib_max": 8.28, "warmup_s_mean": 13.1, "final_loss_mean": 6.87},
        "speedup_B_over_A": 1.239, "final_loss_rel_diff_B_vs_A": 0.026}
    (run_dir / "bench" / "train_mem.json").write_text(json.dumps(bench))
    eq = records.create_run("bench", "compile-equivalence-unit", root=sandbox, compile_enabled=True)
    eq.write_json("bench/compile_equivalence.json", {
        "config": "litegpt-v2-28l", "micro_batch": 3, "seq_len": 2048, "loss_abs_diff": 0.0004,
        "logits_rel_l2_diff": 0.0088, "grad_norm_rel_diff": 0.00003, "passed": True})
    out = tmp_path / "out"
    rb.build(out)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "A:sdpa" in text and "B:sdpa+compile" in text and "Speedup B over A: 1.239x" in text
    assert "Compiled vs eager equivalence" in text and "0.88%" in text and "pass" in text
    assert f"`runs/{eq.path.name}/bench/compile_equivalence.json`" in text


def test_report_power_sweep_section_and_plots(sandbox, tmp_path):
    _fake_bench_run(sandbox)
    rec = records.create_run("bench", "power-sweep-screen", root=sandbox, compile_enabled=True)

    def pt(spec, tok, w, tmax, eligible, valid=True):
        return {"spec": spec, "status": "ok", "valid": valid, "eligible": eligible, "invalid_reasons": [],
                "ineligible_reasons": [] if eligible else ["thermal throttling 5.0%"],
                "metrics": {"tok_per_s": tok, "tok_per_s_per_w": tok / w, "power_w_mean": w, "power_w_peak": w + 5,
                            "power_limit_w_min": 155, "power_limit_w_max": 155, "sm_clock_mhz_mean": 1990,
                            "temp_c_mean": tmax - 3, "temp_c_max": tmax, "util_pct_mean": 99, "vram_used_mib_max": 12000,
                            "frac_thermal_throttled": 0.0, "frac_power_capped": 1.0, "measured_seconds": 240}}
    rec.write_json("bench/power_sweep.json", {
        "stage": "screen", "workload": {"config": "litegpt-v2-28l"}, "notes": "cooling pad", "thresholds": {},
        "points": [pt("observe", 30000.0, 150.0, 84, False), pt("lgc:1950", 29500.0, 128.0, 78, True),
                   pt("lgc:1800", 27000.0, 110.0, 72, True)],
        "recommended": "lgc:1950", "pareto_front": ["observe", "lgc:1950", "lgc:1800"]})
    out = tmp_path / "out"
    rb.build(out)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "GPU operating-point sweep" in text and "Recommended (pre-registered rule): **lgc:1950**" in text
    assert "30,000" in text and "cooling pad" in text and (out / "figures" / "power_sweep_0.svg").exists()
    assert f"`runs/{rec.path.name}/bench/power_sweep.json`" in text


def _fake_data_run(runs_dir: Path) -> Path:
    rec = records.create_run("data", "prepare", root=runs_dir, config={"sources_sha256": "ab" * 32})
    subs = ["fineweb_edu", "cosmopedia_v2", "math", "code_python", "code_javascript", "code_typescript", "code_sql", "code_shell",
            "instruction_style"]
    tokens = {f"{s}/{p}": 10**10 for s in subs for p in ("stable", "decay", "val")}
    rec.write_json("data_stats.json", {
        "files": 3, "docs_read": {"math": 10}, "docs_kept": {}, "tokens": tokens,
        "bytes": {"math/stable": 900, "math/decontam_dropped": 100}, "dropped": {"code_python:pyparse": 7, "code_sql:sqldump": 2},
        "decontam": {"math:humaneval": 3, "fineweb_edu:mmlu": 5}, "emails_redacted_docs": 4})
    rec.write_json("mix_manifest.json", {
        "val_core": {"sha256": "c" * 64, "tokens": 20971520, "blocks": 10240}, "data_order_sha256": "d" * 64, "shortfalls": [],
        "duplicates": {"math": {"train_docs": 100, "within_rate": 0.001, "cross_rate": 0.0, "val_docs": 5, "val_docs_also_in_training": 0}},
        "val_key_check": {"math": {"val_docs": 5, "violations": 0}}, "prefix_proportions_stable": {"max_abs_deviation": 0.0004},
        "resume_check_stable": True, "val_text": {"web": {"docs": 3, "bytes": 2_000_100, "sha256": "e" * 64}}})
    return rec.path


def test_data_section_renders_from_records(sandbox, tmp_path):
    _fake_data_run(sandbox)
    out = tmp_path / "out"
    rb.build(out)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## 3c. Data and tokenizer (M1)" in text and "Tokens prepared vs plan" in text
    assert "c" * 64 in text and "d" * 64 in text and "pyparse" in text and "10.00%" in text
    assert (out / "figures" / "data_mixture.svg").exists()
    assert "mix_manifest.json" in text                       # listed as an input with its sha256


def test_data_section_without_data_run(sandbox, tmp_path):
    out = tmp_path / "out"
    rb.build(out)
    assert "No mixed data run yet" in (out / "REPORT.md").read_text(encoding="utf-8")


def _fake_eval(runs_dir: Path, name: str, config: dict, path: str, result: dict) -> Path:
    rec = records.create_run("eval", name, root=runs_dir, config=config)
    rec.write_json(path, result)
    return rec.path


def _group_results(values: dict) -> dict:
    """lm-eval style results for each task, keyed by the suite's headline metric (and filter)."""
    from litegpt.eval.run import load_suite
    head = load_suite()["headline_metrics"]
    out = {}
    for t, v in values.items():
        m = head[t]
        name, filt = (m.split(",") + ["none"])[:2]
        out[t] = {f"{name},{filt}": v, f"{name}_stderr,{filt}": 0.01}
    return {"results": out}


def _fake_suite_run(runs_dir, name, config, values_by_group):
    rec = records.create_run("eval", name, root=runs_dir, config=config)
    for g, values in values_by_group.items():
        rec.write_json(f"eval/{g}.json", _group_results(values))
    rec.write_json("eval/summary.json", {"headline": {}})
    return rec.path


def test_baselines_section_renders_from_records(sandbox, tmp_path):
    from scripts.build_report import ZERO_SHOT
    gpt2 = {"repo": "openai-community/gpt2", "revision": "a" * 40, "license": "mit"}
    v1 = {"repo": "koganrath/LiteGPT-Instruct", "revision": "b" * 40, "license": "apache-2.0"}
    groups = ["zero_shot_lm", "triviaqa", "gsm8k", "code", "mbpp"]
    zs = dict({t: 0.25 for t in ZERO_SHOT}, hellaswag=0.313)
    _fake_suite_run(sandbox, "base-gpt2", {"suite": "base", "model": gpt2, "comparable": True, "groups": groups},
                    {"zero_shot_lm": zs, "triviaqa": {"triviaqa": 0.017}, "gsm8k": {"gsm8k": 0.006}, "code": {"humaneval": 0.0},
                     "mbpp": {"mbpp": 0.0}})
    _fake_suite_run(sandbox, "base-gpt2-smoke", {"suite": "base", "model": gpt2, "comparable": False, "groups": groups},  # ignored
                    {"zero_shot_lm": dict(zs, hellaswag=0.999)})
    inst = {"suite": "instruct", "model": v1, "comparable": True}
    _fake_suite_run(sandbox, "instruct-LiteGPT-Instruct", dict(inst, groups=["ifeval", "gsm8k_chat", "code_chat", "xstest"]),
                    {"ifeval": {"ifeval": 0.083}, "gsm8k_chat": {"gsm8k_cot_zeroshot": 0.021},
                     "code_chat": {"humaneval_instruct_litegpt": 0.777, "mbpp_instruct_litegpt": 0.0}})
    _fake_suite_run(sandbox, "instruct-LiteGPT-Instruct-rerun", dict(inst, groups=["code_chat"]),   # later re-run of one group
                    {"code_chat": {"humaneval_instruct_litegpt": 0.0123, "mbpp_instruct_litegpt": 0.0456}})
    _fake_eval(sandbox, "efficiency-gpt2", {"task": "efficiency", "model": gpt2}, "eval/efficiency.json",
               {"rows": [{"prompt_tokens": 512, "batch": b, "prefill_tok_s": 9000.0, "ttft_s": 0.02, "decode_tok_s": 150.0 * b,
                          "decode_bytes_s": 600.0, "peak_vram_GiB": 0.7} for b in (1, 8)], "gpu_power_limit_w": 175.0,
                "params": {"total": 124439808, "embedding": 39383808, "non_embedding": 85056000, "tied_output": True, "vocab_size": 50257}})
    _fake_eval(sandbox, "bpb-gpt2", {"task": "bpb", "model": gpt2}, "eval/bpb.json",
               {"window": 1024, "stride": 512, "domains": {"web": {"bpb": 1.111}, "code": {"bpb": 2.222}}})
    _fake_eval(sandbox, "xstest-LiteGPT-Instruct", {"task": "xstest", "model": v1}, "eval/xstest.json",
               {"unsafe_refusal_rate": 0.0, "safe_over_refusal_rate": 0.04})
    out = tmp_path / "out"
    rb.build(out)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "## 3d. Baselines (M1b)" in text
    assert "0.313" in text and "0.999" not in text                  # latest COMPLETE, COMPARABLE record only
    assert "0.083" in text and "0.012" in text and "0.046" in text and "0.777" not in text   # group re-run supersedes
    assert "124.4M" in text and "85.1M" in text and "50,257" in text  # counted parameters from the efficiency record
    assert "not published" in text and "300B" not in text            # GPT-2 tokens: published context, not invented
    assert "1.111" in text and "2.222" in text and "4.0%" in text and "1,200" in text
    assert (out / "figures" / "baselines_capability.svg").exists() and (out / "figures" / "baselines_bpb.svg").exists()
    assert "eval/efficiency.json" in text                             # listed as a report input


def test_baselines_section_without_records(sandbox, tmp_path):
    out = tmp_path / "out"
    rb.build(out)
    assert "No baseline records yet" in (out / "REPORT.md").read_text(encoding="utf-8")
