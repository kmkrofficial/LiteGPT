"""Hooks for the corrected chat code tasks (EXP-M1B-SUITE correction 2026-10-12, suite 0.1.1).

lm-eval 0.4.13's humaneval_instruct stops generation at "\\ndef", so a model that restates the whole function (a correct
answer) is cut to an empty string; its mbpp_instruct extractor puts "```" back in front of the answer and reads a leading
"def" as the code block's language tag, deleting it. Both tasks here keep lm-eval's prompts, prefixes, datasets and metrics;
only generation stops and code extraction differ. Metrics import lm-eval lazily so the extraction is testable without it.
"""
import re

_REOPENED_FENCE = re.compile(r"\A\s*```[\w+-]*[ \t]*\n")


def code_before_fence(response: str) -> str:
    """The answer's code. The prompt already opened the ```python block (gen_prefix), so the code is everything before the
    closing fence; a model that opens a new block anyway has that opening line removed first."""
    text = _REOPENED_FENCE.sub("", response, count=1)
    end = text.find("```")
    return text if end == -1 else text[:end]


def build_predictions_humaneval(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Prompt (imports, signature, docstring) + the answer's code. A restated full function simply redefines the stub."""
    return [[doc["prompt"] + code_before_fence(r) for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_mbpp(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[code_before_fence(r) for r in resp] for resp in resps]


def pass_at_k(references, predictions, k=None):
    from lm_eval.tasks.humaneval.utils import pass_at_k as metric
    return metric(references, predictions, k)


def pass_at_1(references, predictions):
    from lm_eval.tasks.mbpp.utils import pass_at_1 as metric
    return metric(references, predictions)
