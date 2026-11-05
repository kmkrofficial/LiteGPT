# Chat template

`litegpt-chat-v1`, ChatML-style. Code: [`litegpt/chat_template.py`](../litegpt/chat_template.py). Jinja twin for Hugging Face tools: [`litegpt-chat-v1.jinja`](../litegpt/assets/tokenizer/litegpt-chat-v1.jinja).

```text
<|bos|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n
```

- `render_ids` inserts the special ids itself and encodes message text as ordinary text, so a message containing `<|im_end|>` cannot act as a control token. The Jinja twin cannot stop this, so LiteGPT's own code always uses `render_ids`.
- `render` also returns the assistant-only loss mask: 1 on the assistant's text and its `<|im_end|>`.
- `<|im_end|>` ends a turn and is a stop token. The `tool` role is reserved and refused.
- `tests/test_chat_template.py` checks the Python and Jinja ids are identical.
