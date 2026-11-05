# Documentation

**Start here:** [results.md](results.md) for every benchmark, [figures](figures/README.md) for every chart, and [history.md](history.md) for how V2 was built.

| Page | What it covers |
|---|---|
| [results.md](results.md) | Benchmarks and comparisons, where V2 leads and where it loses |
| [figures/](figures/README.md) | All charts, generated from the run records |
| [history.md](history.md) | Milestones, decisions, deviations, failures and cost |
| [architecture.md](architecture.md) | The model |
| [inference.md](inference.md) | Engine, Python API and CLI |
| [chat_template.md](chat_template.md) | The chat format |
| [training.md](training.md) | Pretraining, chat fine-tuning and text-to-data fine-tuning |
| [data.md](data.md) | Data sources, mixture and preparation |
| [data_licenses.md](data_licenses.md) | Licence and provenance register |
| [evaluation.md](evaluation.md) | Evaluation suite and scoring |
| [benchmarking.md](benchmarking.md) | Run records and generated reports |
| [reproduce.md](reproduce.md) | Setting up the environment |
| [distribution.md](distribution.md) | Release state and plan |
| [repository_structure.md](repository_structure.md) | What goes where |
| [m0_benchmark_findings.md](m0_benchmark_findings.md) | Early GPU findings, kept for history |

**Binding records:**
- [spec/v2.yaml](spec/v2.yaml): the specification.
- [spec/registry.yaml](spec/registry.yaml): every experiment, with thresholds set before it ran, and its conclusion.
- [spec/data_sources.yaml](spec/data_sources.yaml): exact dataset revisions.
- [decisions/](decisions/README.md): architecture decision records.

**Release drafts:** [Base model card](release/model_card_base.md), [Instruct model card](release/model_card_instruct.md), [Base notice](release/NOTICE.txt), [Instruct notice](release/NOTICE_instruct.txt).
