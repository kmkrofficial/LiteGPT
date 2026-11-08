# Figures

Generated from the run records by `python -m scripts.build_figures`. Don't edit by hand. V2 is always blue and V1 always orange. The tables behind most charts are in [results.md](../results.md).

## Text-to-data comparison

![V2 scored lowest of the four, on every seed](t2d_scores.svg)

*Mean of CUAD, MASSIVE and Banking77. Fine-tuned by one recipe, test scored once.* Source: `runs/m4_t2d/test_results.json`.

![No win: every point sits left of zero](t2d_gap.svg)

*A loss is more than 2 standard errors behind. Paired bootstrap over test items, 10,000 rounds.* Source: `runs/m4_t2d/test_results.json`.

![Close on short text, furthest behind on contracts](t2d_tasks.svg)

*Mean and each of 3 seeds. Each axis is zoomed separately and does not start at zero.* Source: `runs/m4_t2d/test_results.json`.

![Parties are the hardest contract field for all four models](t2d_cuad_fields.svg)

*Mean of 3 seeds, 98 contracts.* Source: `runs/m4_t2d/test_results.json`.

![On CPU V2 is mid-pack: light, but not the fastest](t2d_cpu.svg)

*8 threads, one run, small samples. Peak RAM, GiB: SmolLM2 1.61, V2 1.66, Gemma 2.62, Qwen3 4.27.* Source: `runs/m4_t2d/cpu-*.json`.

![More size bought more score, and V2 sits below the trend](t2d_size.svg)

*SmolLM2-135M is smaller, uses about the same RAM, and scores higher.* Source: `runs/m4_t2d/test_results.json, cpu-*.json`.

![Dev scores predicted the order, and test confirmed it](t2d_dev_vs_test.svg)

*Routes and learning rates were chosen on dev only, before test was touched.* Source: `runs/m4_t2d/routes.json, test_results.json`.

![V2 and SmolLM2 chose the top of the grid, Gemma and Qwen3 the bottom](t2d_lr_grid.svg)

*Each could have scored more just past the fixed grid. The ring marks the choice.* Source: `runs/m4_t2d/lr_choices.json`.

![Fine-tuning runs were stable for every model](t2d_finetune_loss.svg)

*Chosen route, all 3 seeds. Loss is on each model's own tokens, so levels aren't comparable across models.* Source: `runs/*-train-m4-t2d-*/metrics.jsonl`.

![BillSum: V2 wrote its summaries fastest, and they scored lowest](t2d_billsum.svg)

*Reported only, not part of the score. Chosen route, seed 1, 1,000 items, on the A100.* Source: `runs/m4_t2d/billsum-*.json`.

## Base models

![As a base model V2 ranks 6th of 11](base_zero_shot_mean.svg)

*Ahead of V1, behind SmolLM2-135M and Gemma 3 270M, the two nearest its size.* Source: `runs/*-eval-base-*/eval/summary.json`.

![V2 is above chance on some tasks and close to it on others](base_zero_shot_tasks.svg)

*Each dot is one model. Chance lines are approximate. TruthfulQA has no single chance level.* Source: `runs/*-eval-base-*/eval/summary.json`.

![Math, code and fact recall are where V2 is weakest](base_few_shot_code.svg)

*Higher is better. TriviaQA and GSM8K few-shot; HumanEval and MBPP pass rate.* Source: `runs/*-eval-base-*/eval/summary.json`.

![V2 sits below the line modern small models set](base_size_vs_quality.svg)

*SmolLM2-135M beats it with fewer parameters.* Source: `runs/*-eval-base-*, runs/*-eval-efficiency-*`.

![V2 predicts code text well for its size, and web text less well](base_bits_per_byte.svg)

*Bits per byte, lower is better, best on top. Chat is left out: V2's chat text overlaps its training mix.* Source: `runs/*-eval-bpb-*/eval/bpb.json`.

![V2 uses the least GPU memory, and older, shallower models decode faster](base_speed.svg)

*Laptop GPU, prompt 512, 128 new tokens. V2 uses its own engine, the others HF generate: not like for like.* Source: `runs/*-eval-efficiency-*/eval/efficiency.json`.

![No model here is both fastest and best](base_speed_vs_quality.svg)

*V2 is the fastest of the modern models and sits mid-table on quality. Engines differ.* Source: `runs/*-eval-efficiency-*, runs/*-eval-base-*`.

![V2 against V1: much better on code text, smaller gains elsewhere](v1_vs_v2.svg)

*V2 is also 1.6 times larger, on different data. This shows the whole model changing, not code alone.* Source: `runs/*-eval-base-*, runs/*-eval-bpb-*`.

![The last billion tokens moved the most](m3_checkpoints.svg)

*Learning-rate decay, data mix and context all changed there, so one run can't separate the causes.* Source: `runs/*-eval-base-litegpt-*, runs/*-eval-bpb-litegpt-*`.

## CPU and Raspberry Pi

![On CPU, 4 to 8 threads is the sweet spot for decode](cpu_threads.svg)

*Laptop CPU (i9-14900HX), prompt 128, 64 new tokens, median of 3. Past 8 threads the slower E-cores join.* Source: `runs/*-bench-cpu-*/bench/cpu.json`.

![On CPU, V2 needs under 1.7 GiB and keeps pace with models of its size](cpu_ram_speed.svg)

*float32, prompt 128. The Pi 5 figure is an estimate from laptop runs and published Pi bandwidth, not a measurement.* Source: `runs/*-bench-cpu-*/bench/`.

## Chat models

![V2-Instruct beats V1-Instruct and trails the rest on instructions](chat_tasks.svg)

*Higher is better. IFEval is the registered gate for chat.* Source: `runs/*-eval-instruct-*/eval/summary.json`.

![V2-Instruct refuses nothing, so it has no safety behaviour](chat_xstest.svg)

*XSTest, 200 unsafe and 250 safe prompts. Refusals are detected by string match.* Source: `runs/*-eval-xstest-*/eval/xstest.json`.

## Training

![Pretraining ran stably for 37 hours, with no restarts or skipped steps](m3_loss.svg)

*Dotted lines: decay and new data mix from 9.0B, context 4,096 from 9.6B. The first steps are above the axis.* Source: `20261101T142254Z-train-m3-673cd85/metrics.jsonl`.

![Learning rate, gradients and speed across pretraining](m3_signals.svg)

*Speed drops at 9.6B, when the context doubles. The early gradient norms are above the axis.* Source: `20261101T142254Z-train-m3-673cd85/metrics.jsonl`.

![Every domain improved, most of all in the final billion tokens](m3_bits_per_byte.svg)

*Chat's drop partly reflects overlap with the new data mix, so it isn't a fair measure of chat ability.* Source: `20261101T142254Z-train-m3-673cd85/eval.jsonl`.

![The registered gate said stop: 0.00698 gained, 0.010 needed](m3_gate.svg)

*Extending past 10B needed the window means to differ by the registered threshold. Decided by script.* Source: `20261101T142254Z-train-m3-673cd85/gate/extension_gate.json`.

![Peak learning rate 2.4e-3 was chosen from a short sweep](m3_lr_sweep.svg)

*200M tokens each, with a short warm-up. Chosen before pretraining began.* Source: `runs/*-train-m3lr-*/eval.jsonl`.

![Architecture steps gained a little; the data mix transformed code](ladder_steps.svg)

*One change at a time on a small proxy model, 1B tokens each. Bits per byte, lower is better.* Source: `runs/*-train-ladder-*/eval.jsonl`.

![One baseline seed diverged late, so a stability rule was added](ladder_seed_divergence.svg)

*Taken as noise, seed 2 would have set a threshold no step could pass. Recorded before use.* Source: `runs/*-train-ladder-s1-*`.

![The ladder's learning rate came from a small registered sweep](ladder_lr_sweep.svg)

*The proxy model, 0.25B tokens each. The lowest web bits per byte won.* Source: `runs/*-train-ladder-lrsweep-*`.

![Chat fine-tuning ran 3 epochs with no skipped steps](sft_training.svg)

*Peak learning rate 6e-4, on flash attention at micro-batch 8.* Source: `20261103T144735Z-train-m4-sft-0034095/metrics.jsonl`.

![Held-out loss fell every epoch, and the sweep chose the top of its grid](sft_heldout.svg)

*Lower is better. This chooses the epoch and says nothing about quality against other models.* Source: `20261103T144735Z-train-m4-sft-0034095/train_summary.json`.

![On the rented A100, the two attention backends were nearly equal](a100_tuning.svg)

*Compiled, fixed-length rows, synthetic data, 2,048 context. Real conversations ran slower.* Source: `20261101T140558Z-bench-tune-throughput-5f34d96`.
