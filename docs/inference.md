# Inference engine

**Status: core built (M1b).** The Python API (`load` / `generate` / `chat` / `chat_stream`) and the CLI build on it in M4.

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
