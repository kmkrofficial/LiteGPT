"""Corrected chat code tasks (suite 0.1.1): code extraction on the failure cases found in the 0.1.0 records."""
import importlib.util
from pathlib import Path

import yaml

TASKS = Path(__file__).resolve().parents[1] / "litegpt" / "eval" / "tasks" / "code_chat"
spec = importlib.util.spec_from_file_location("code_chat_utils", TASKS / "utils.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

PROMPT = 'from typing import List\n\n\ndef f(xs: List[int]) -> int:\n    """Sum."""\n'


def test_mbpp_keeps_a_leading_def():
    # lm-eval 0.4.13 read "def" as the fence's language tag and returned " remove_Occ(...)"
    resp = 'def remove_Occ(s, c):\n    return s.replace(c, "")\n```\n\nThis function works by ...'
    assert U.build_predictions_mbpp([[resp]], [{}]) == [['def remove_Occ(s, c):\n    return s.replace(c, "")\n']]


def test_mbpp_reopened_fence_and_unclosed_block():
    assert U.code_before_fence("\n```python\ndef g():\n    return 1\n```\ntext") == "def g():\n    return 1\n"
    assert U.code_before_fence("def g():\n    return 1") == "def g():\n    return 1"


def test_humaneval_body_continuation_and_restated_function_both_compile_and_run():
    body = "    return sum(xs)\n```\nExplanation."
    restated = "\ndef f(xs: List[int]) -> int:\n    return sum(xs)\n```\n\nThis function works by ..."
    for resp in (body, restated):
        (code,), = U.build_predictions_humaneval([[resp]], [{"prompt": PROMPT}])
        ns = {}
        exec(compile(code, "<answer>", "exec"), ns)  # test-only: fixed, trusted strings
        assert ns["f"]([1, 2, 3]) == 6


def test_task_files_keep_lm_eval_prompts_and_drop_only_the_stops():
    he = yaml.safe_load((TASKS / "humaneval_instruct_litegpt.yaml").read_text(encoding="utf-8").replace("!function ", ""))
    mb = yaml.safe_load((TASKS / "mbpp_instruct_litegpt.yaml").read_text(encoding="utf-8").replace("!function ", ""))
    assert he["generation_kwargs"]["until"] == [] and he["gen_prefix"].startswith("Here is the completed function:")
    assert mb["gen_prefix"] == "\n```python\n" and mb["num_fewshot"] == 0
