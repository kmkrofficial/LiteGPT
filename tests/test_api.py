import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import litegpt  # noqa: E402
from litegpt import api, chat_template, cli  # noqa: E402
from litegpt import t2d as X  # noqa: E402
from litegpt.config import get_config  # noqa: E402
from litegpt.inference import GenerationConfig  # noqa: E402
from litegpt.model import LiteGPT  # noqa: E402
from litegpt.tokenizer import BOS  # noqa: E402

CHEAP = {"max_new_tokens": 4}


def write_checkpoint(path, arm=None, backend="sdpa", block_size=256):
    torch.manual_seed(0)
    cfg = get_config("litegpt-tiny", vocab_size=32768, block_size=block_size, attention_backend="sdpa")
    state = {"format": 1, "model": LiteGPT(cfg).state_dict(), "model_config": {**cfg.to_dict(), "attention_backend": backend}, "step": 7, "tokens": 70, "spec_sha256": None}
    if arm:
        state["t2d"] = {"arm": arm, "lr": 1e-4, "seed": 1}
    torch.save(state, path)
    return path


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return write_checkpoint(tmp_path_factory.mktemp("api") / "m.pt")


@pytest.fixture(scope="module")
def m(ckpt):
    return api.load(ckpt, device="cpu", dtype=torch.float32)


def capture_prompt(monkeypatch):
    seen = {}

    def fake(model, prompt, options):
        seen["prompt"] = prompt
        return seen.get("reply", "x"), "stop"

    monkeypatch.setattr(api, "_complete", fake)
    return seen


def test_package_exposes_the_api_lazily():
    assert litegpt.load is api.load and litegpt.chat_stream is api.chat_stream
    with pytest.raises(AttributeError):
        litegpt.nothing_here


def test_load_reads_a_flash_checkpoint_on_sdpa(tmp_path):
    loaded = api.load(write_checkpoint(tmp_path / "f.pt", backend="flash"), device="cpu", dtype=torch.float32)
    assert loaded.model.config.attention_backend == "sdpa" and loaded.step == 7 and loaded.layout is None


def test_load_refuses_a_file_that_would_run_code(tmp_path):
    class Boom:
        def __reduce__(self):
            return (print, ("ran",))

    p = tmp_path / "bad.pt"
    torch.save({"model": Boom()}, p)
    with pytest.raises(Exception, match="(?i)weights_only|unsupported|global"):
        api.load(p, device="cpu")


def test_load_reads_a_training_checkpoint_with_its_rng_state(tmp_path):
    p = write_checkpoint(tmp_path / "t.pt")
    state = torch.load(p, weights_only=False)
    state["rng"] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    torch.save(state, p)
    assert api.load(p, device="cpu", dtype=torch.float32).step == 7


def test_cli_refuses_a_file_that_would_run_code(tmp_path, capsys):
    class Boom:
        def __reduce__(self):
            return (print, ("ran",))

    p = tmp_path / "bad.pt"
    torch.save({"model": Boom()}, p)
    assert cli.main(["info", "--model", str(p), "--device", "cpu"]) == 1
    captured = capsys.readouterr()
    assert "not loaded" in captured.err and "ran" not in captured.out


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.load(tmp_path / "none.pt", device="cpu")


def test_generate_starts_at_bos_and_equals_the_engine(m):
    prompt = [BOS] + m.tok.encode_text("Hello")
    want = m.engine.generate_ids([prompt], GenerationConfig(**CHEAP))[0].text
    assert api.generate(m, "Hello", **CHEAP) == want


def test_chat_uses_the_registered_template(m, monkeypatch):
    seen = capture_prompt(monkeypatch)
    api.chat(m, "Hi", system="S", **CHEAP)
    assert seen["prompt"] == chat_template.render_ids(m.tok, [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}], add_generation_prompt=True)


def test_stream_joins_to_the_same_text_as_chat(m):
    assert "".join(api.chat_stream(m, "Hi", max_new_tokens=8)) == api.chat(m, "Hi", max_new_tokens=8)


@pytest.mark.parametrize("call", [lambda m, **o: api.generate(m, "word " * 400, **o), lambda m, **o: api.chat(m, "word " * 400, **o), lambda m, **o: list(api.chat_stream(m, "word " * 400, **o))])
def test_too_long_is_an_error_not_a_silent_cut(m, call):
    with pytest.raises(ValueError, match="exceeds the context"):
        call(m, **CHEAP)


def test_plain_checkpoint_refuses_chat(tmp_path):
    plain = api.load(write_checkpoint(tmp_path / "p.pt", arm="v2-base"), device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="plain prompts"):
        api.chat(plain, "Hi", **CHEAP)


def test_extract_refuses_a_checkpoint_not_fine_tuned(m):
    with pytest.raises(ValueError, match="not fine-tuned"):
        api.extract(m, "banking77", "card")


def test_extract_refuses_an_unknown_task(tmp_path):
    plain = api.load(write_checkpoint(tmp_path / "p.pt", arm="v2-base"), device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown task"):
        api.extract(plain, "billsum", "text")


@pytest.mark.parametrize("arm,chat", [("v2-base", False), ("v2-chat", True)])
def test_extract_prompt_is_the_one_scored(tmp_path, monkeypatch, arm, chat):
    loaded = api.load(write_checkpoint(tmp_path / f"{arm}.pt", arm=arm), device="cpu", dtype=torch.float32)
    seen = capture_prompt(monkeypatch)
    api.extract(loaded, "massive", "wake me at nine")
    user = X.user_text("massive", {"input": "wake me at nine"})
    want = chat_template.render_ids(loaded.tok, [{"role": "user", "content": user}], add_generation_prompt=True) if chat else [BOS] + loaded.tok.encode_text(user + X.ANSWER_CUE)
    assert seen["prompt"] == want


def test_extract_parses_and_never_repairs(tmp_path, monkeypatch):
    plain = api.load(write_checkpoint(tmp_path / "p.pt", arm="v2-base"), device="cpu", dtype=torch.float32)
    seen = capture_prompt(monkeypatch)
    seen["reply"] = '[{"type": "time", "value": "nine"}]'
    r = api.extract(plain, "massive", "x")
    assert r["valid"] and r["data"] == [{"type": "time", "value": "nine"}]
    seen["reply"] = '[{"type": "time"'
    r = api.extract(plain, "massive", "x")
    assert not r["valid"] and r["data"] is None and r["output"] == seen["reply"]
    seen["reply"] = " lost_card\n"
    r = api.extract(plain, "banking77", "x")
    assert r["valid"] and r["data"] == "lost_card"


def test_info(m):
    i = api.info(m)
    assert i["parameters"] == sum(p.numel() for p in m.model.parameters()) and i["context"] == 256 and i["step"] == 7 and i["dtype"] == "float32"


def test_cli_info_generate_and_errors(ckpt, capsys, tmp_path):
    assert cli.main(["info", "--model", str(ckpt), "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == 70
    assert cli.main(["generate", "--model", str(ckpt), "--device", "cpu", "--max-new-tokens", "3", "Hello"]) == 0
    capsys.readouterr()
    assert cli.main(["info", "--model", str(tmp_path / "none.pt"), "--device", "cpu"]) == 1
    assert "no checkpoint" in capsys.readouterr().err
    assert cli.main(["generate", "--model", str(ckpt), "--device", "cpu", "--max-new-tokens", "9999", "Hello"]) == 1
    assert "exceeds the context" in capsys.readouterr().err


def test_cli_chat_keeps_history_and_resets(ckpt, monkeypatch, capsys):
    import io
    seen = []
    real = api.chat_stream

    def spy(m, history, **kw):
        seen.append([x["content"] for x in history])
        return real(m, history, **kw)

    monkeypatch.setattr(api, "chat_stream", spy)
    monkeypatch.setattr(sys, "stdin", io.StringIO("one\ntwo\n/reset\nthree\n/exit\nnever\n"))
    assert cli.main(["chat", "--model", str(ckpt), "--device", "cpu", "--max-new-tokens", "2"]) == 0
    assert len(seen) == 3 and seen[0] == ["one"] and seen[1][0] == "one" and seen[1][-1] == "two" and seen[2] == ["three"]


def test_cli_extract_from_a_file(tmp_path, monkeypatch, capsys):
    ck = write_checkpoint(tmp_path / "p.pt", arm="v2-base")
    capture = {}

    def fake(model, prompt, options):
        capture["n"] = options["max_new_tokens"]
        return "lost_card", "stop"

    monkeypatch.setattr(api, "_complete", fake)
    doc = tmp_path / "q.txt"
    doc.write_text("My card is gone", encoding="utf-8")
    assert cli.main(["extract", "--model", str(ck), "--device", "cpu", "--task", "banking77", "--input", str(doc)]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == "lost_card" and capture["n"] == 256


def test_cli_help_does_not_import_torch():
    code = "import sys; from litegpt import cli; \ntry:\n cli.main(['--help'])\nexcept SystemExit:\n pass\nassert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, cwd=Path(__file__).resolve().parents[1])
