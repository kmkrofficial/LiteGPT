# Inference engine

**Status: built.** Engine in M1b; Python API and CLI in M4 (end of this page).

`litegpt/inference/` works on token ids; text handling is limited to what stop strings and streaming need.

| Module | What it does |
|---|---|
| `engine.py` | `Engine.generate_ids(prompts, GenerationConfig)`: prefill + incremental decode on the model's preallocated `KVCache`; batched prompts are left-padded and the model's `key_padding_mask` keeps padded keys out of attention (each position may always attend to itself, so no attention row is empty). `Engine.stream(prompt, cfg)` yields text deltas for one prompt. |
| `sampling.py` | Logits processors: repetition / frequency / presence penalties, temperature, top-k, top-p, min-p; greedy when temperature is 0; seeded sampling. |
| `stopping.py` | Stop strings (earliest match wins), stream hold-back of text that could start a stop string, UTF-8-safe incremental decoding (nothing ending in an incomplete multi-byte character is emitted). |

`GenerationConfig` fields: `max_new_tokens`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`frequency_penalty`, `presence_penalty`, `stop_token_ids` (default `<|endoftext|>`, `<|im_end|>`), `stop_strings`, `seed`.
Results carry the generated ids, the decoded text (with a tokenizer), `finish_reason` (`stop` or `length`) and token usage.

Guarantees, each covered by `tests/test_inference.py`:

- Greedy cached decoding equals greedy decoding by repeated full forward passes.
- A left-padded batch produces exactly the same tokens as running each prompt alone.
- Seeded sampling is reproducible.
- Prompts longer than the context are rejected.
- Streaming never emits a partial UTF-8 character or a partial stop string.

## Example (runs in the test suite)

```python
# doc-test
import torch
from litegpt.config import get_config
from litegpt.inference import Engine, GenerationConfig
from litegpt.model import LiteGPT

torch.manual_seed(0)
model = LiteGPT(get_config("litegpt-tiny", attention_backend="sdpa")).float()
engine = Engine(model)
results = engine.generate_ids([[5, 6, 7], [9, 10]], GenerationConfig(max_new_tokens=8, stop_token_ids=()))
assert [len(r.ids) for r in results] == [8, 8] and results[0].finish_reason == "length"
```

Not yet: continuous batching, paged KV cache, speculative decoding, context management (dropping old turns), which come with
the API in M4 or later.

## Python API and CLI

```text
import litegpt
m = litegpt.load("weights.pt")                                   # GPU if there is one, bfloat16
litegpt.generate(m, "Once upon a time", max_new_tokens=32)
litegpt.chat(m, "What is a contract?", system="Be brief.")
for piece in litegpt.chat_stream(m, "Name three colours."): ...
litegpt.extract(m, "banking77", "My card has not arrived.")      # a text-to-data fine-tune only
```

```text
litegpt info --model weights.pt
litegpt generate --model weights.pt "Once upon a time"
litegpt chat --model milestone-final.pt
litegpt extract --model weights.pt --task cuad --input contract.txt
```

Everything runs offline. Options are the fields of `GenerationConfig`; greedy by default. A prompt that does not fit the context is an error, never cut silently. `load` reads a checkpoint with `weights_only`, allowing only the numpy types of a training checkpoint's RNG state, so a file that would run code is refused.

`extract` takes `cuad`, `massive` or `banking77` and uses the exact prompt and greedy decoding of EXP-M4-T2D. It refuses a checkpoint that was not fine-tuned for these tasks and never repairs invalid JSON: `valid` is false and `output` keeps the text. Exit code 2 from the CLI means invalid output. It runs one document at a time, where scoring ran batches: on real V2 weights it reproduced 209 of 210 saved dev outputs. For this job, see the result in the README first.

## Through Transformers

An exported model loads with `AutoModelForCausalLM` and needs no remote code. In float32, logits and greedy output are identical to LiteGPT's on the registered prompts (EXP-M5-EXPORT). Benchmarks were scored through LiteGPT's engine, and at bfloat16 Transformers can differ from it slightly.

`litegpt.load` and the `litegpt` command also read an exported folder.
