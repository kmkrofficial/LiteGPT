# LiteGPT

A small language model trained from scratch, with its whole build on record: every dataset revision, every threshold set before the run, and every result, including the comparison it lost.

**LiteGPT V2** has 201M parameters. It is a new implementation with its own trained tokenizer and weights, designed on one laptop and trained on one rented GPU. It is a research model. **No V2 weights are public yet**: the release is being prepared.

## Should you use it?

| You want | Use |
|---|---|
| To extract data from text, offline | SmolLM2-135M or Gemma 3 270M, or Qwen3-0.6B if you have the memory. V2 lost to all three |
| A good small chat model | Gemma 3 270M-it or Qwen3-0.6B. Our chat model scores well below them and refuses nothing |
| To see how a small model gets built and tested, failures included | **LiteGPT V2** |

## Results

![V2 scored lowest of four models in the text-to-data comparison](docs/figures/t2d_scores.svg)

| Text-to-data, after fine-tuning | V2 | SmolLM2-135M | Gemma 3 270M | Qwen3-0.6B |
|---|---:|---:|---:|---:|
| Parameters | 201M | 135M | 268M | 596M |
| Score (higher is better) | 0.800 | 0.817 | 0.822 | 0.851 |
| V2's verdict | | Loss | Loss | Loss |
| Peak RAM on CPU | 1.66 GiB | 1.61 GiB | 2.62 GiB | 4.27 GiB |

| As a base model | V2 | V1 | SmolLM2-135M | Gemma 3 270M |
|---|---:|---:|---:|---:|
| Zero-shot mean, 10 tasks | 0.383 | 0.371 | 0.403 | 0.393 |
| TriviaQA | 0.054 | 0.029 | 0.134 | 0.133 |
| Decode speed on a laptop GPU, tok/s | 86 | 213 | 46 | 46 |
| GPU memory at batch 8, GiB | 0.59 | 0.78 | 0.73 | 2.59 |

![V2 against V1 on every benchmark](docs/figures/v1_vs_v2.svg)

V2 is clearly better than V1 on five tasks and clearly worse on none. Its speed and memory figures use its own engine and the others use Hugging Face `generate`, so they aren't like for like. It can't yet do math or write working code. **[All benchmarks and limits](docs/results.md)**, and [every chart](docs/figures/README.md).

## Use it

Install from this repository (Python 3.12, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --locked
```

```bash
uv run litegpt info --model weights.pt
```

```bash
uv run litegpt generate --model weights.pt "The capital of France is"
```

```bash
uv run litegpt chat --model instruct.pt
```

```python
import litegpt

m = litegpt.load("instruct.pt")
print(litegpt.chat(m, "Name three colours.", max_new_tokens=48))
```

Everything runs offline. More in [docs/inference.md](docs/inference.md).

## What's here

| Folder | What it holds |
|---|---|
| [`litegpt/`](litegpt/README.md) | The package: model, tokenizer, inference, training, evaluation, API and CLI |
| [`docs/`](docs/README.md) | Results, history, architecture, data, licences, decisions and the experiment registry |
| [`benchmarks/`](benchmarks/README.md) | Run configs and benchmark tools |
| [`scripts/`](scripts/README.md) | Data preparation, training, scoring and rented-server tools |
| [`tests/`](tests/README.md) | The test suite |
| [`runs/`](runs/README.md) | Run records behind every reported number |

## How it was built

Every experiment was registered before it ran. Design choices were tested one at a time on small models. Pretraining took 37 hours on one rented A100 and cost $16.64, and the cloud total including chat fine-tuning and the comparison was $24.73. [The history](docs/history.md) covers the decisions, the deviations and what went wrong.

No one has rebuilt the model independently. The recipe is on record and re-runnable: [docs/reproduce.md](docs/reproduce.md).

## V1

V1 is GPT-2 Small (124M), trained on FineWeb-Edu and fine-tuned on Alpaca, published as `koganrath/LiteGPT-Base` and `koganrath/LiteGPT-Instruct`. Its code is preserved in [`scripts/legacy_v1/`](scripts/legacy_v1/README.md) as the baseline.

## Licence and attribution

Code is MIT licensed ([`LICENSE`](LICENSE)).

The licence of the V2 weights isn't final: Apache-2.0 is proposed in [ADR-011](docs/decisions/ADR-011-model-licence.md), and no weights are published before it is accepted. Datasets and their terms are in [docs/data_licenses.md](docs/data_licenses.md).

## Acknowledgements

- [PyTorch](https://pytorch.org), [flash-attn](https://github.com/Dao-AILab/flash-attention), Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers) and [`transformers`](https://github.com/huggingface/transformers), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), which V2 is built and scored with
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, the starting point for V1's initial design
- [SmolLM corpus, FineMath and SmolTalk](https://huggingface.co/HuggingFaceTB) by Hugging Face
- [github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean) by CodeParrot
- [CUAD](https://huggingface.co/datasets/theatticusproject/cuad), [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), [Banking77](https://huggingface.co/datasets/PolyAI/banking77) and [BillSum](https://huggingface.co/datasets/FiscalNote/billsum), used for the comparison
