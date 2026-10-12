"""Native evaluations: parameter counting on tiny randomly initialised HF models (tied and untied output heads)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
transformers = pytest.importorskip("transformers")
from litegpt.eval.native import param_counts  # noqa: E402


def test_tied_gpt2_counts_token_and_position_tables_once():
    m = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=100, n_positions=64))
    c = param_counts(m)
    assert c["tied_output"] and c["vocab_size"] == 100
    assert c["embedding"] == 100 * 32 + 64 * 32
    assert c["total"] == sum(p.numel() for p in m.parameters()) and c["non_embedding"] == c["total"] - c["embedding"]


def test_untied_head_counts_as_embedding():
    cfg = transformers.GPTNeoXConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, intermediate_size=64,
                                     vocab_size=100, tie_word_embeddings=False)
    c = param_counts(transformers.GPTNeoXForCausalLM(cfg))
    assert not c["tied_output"] and c["embedding"] == 2 * 100 * 32
