---
title: LiteGPT V2 Chat
emoji: 💬
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: Chat with LiteGPT V2 in your browser, built from scratch
models:
- kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# LiteGPT V2 Chat

Try [LiteGPT-Llama-V2-Instruct](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct) in your browser: a 201M-parameter language model built from scratch, with every step on record.

The page loads the 127 MB [Q4_K_M GGUF file](https://huggingface.co/kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF) into your browser and runs it there with [wllama](https://github.com/ngxson/wllama), llama.cpp compiled to WebAssembly. There is no server: nothing you type leaves the page, and the file is cached after the first visit.

It is a small model with no safety tuning: answers can be wrong, and it is meant for trying the model out, not for advice.

Run it yourself with llama.cpp:

```bash
llama-server -hf kmkrworks/LiteGPT-Llama-V2-Instruct-GGUF:Q8_0 --jinja
```

Code, results and run records: [github.com/kmkrofficial/LiteGPT](https://github.com/kmkrofficial/LiteGPT). This Space's source is `deploy/space/` there.

**Built with Llama. Built with Qwen.** Use of this model should follow the [Llama Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy).
