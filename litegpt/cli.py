"""`litegpt` command line interface: chat, generate, info and extract, all offline.

    litegpt info --model weights.pt
    litegpt generate --model weights.pt "Once upon a time"
    litegpt chat --model milestone-final.pt
    litegpt extract --model weights.pt --task cuad --input contract.txt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def _options(a) -> dict:
    return {"max_new_tokens": a.max_new_tokens, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed}


def _chat(m, a, api) -> int:
    history: list[dict] = []
    print("/reset clears the conversation, /exit leaves.", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if line == "/exit":
            break
        if line == "/reset":
            history = []
            continue
        if not line:
            continue
        history.append({"role": "user", "content": line})
        reply = []
        try:
            for piece in api.chat_stream(m, history, system=a.system, **_options(a)):
                reply.append(piece)
                print(piece, end="", flush=True)
        except ValueError as e:
            print(f"error: {e}. Use /reset to start again.", file=sys.stderr)
            history.pop()
            continue
        print(flush=True)
        history.append({"role": "assistant", "content": "".join(reply)})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="litegpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--model", required=True, help="an exported model folder, or a LiteGPT checkpoint file")
    base.add_argument("--device", default=None, help="default: the GPU if there is one")
    sample = argparse.ArgumentParser(add_help=False)
    sample.add_argument("--max-new-tokens", type=int, default=256)
    sample.add_argument("--temperature", type=float, default=0.0, help="0 is greedy")
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--seed", type=int, default=None)
    sub.add_parser("info", parents=[base], help="what a checkpoint is")
    p = sub.add_parser("generate", parents=[base, sample], help="continue plain text")
    p.add_argument("prompt", help="the text, or - to read it from standard input")
    p = sub.add_parser("chat", parents=[base, sample], help="talk to a chat-trained checkpoint; one message per line")
    p.add_argument("--system", default=None)
    p = sub.add_parser("extract", parents=[base], help="one document through a fine-tuned text-to-data task; prints JSON")
    p.add_argument("--task", required=True, choices=["cuad", "massive", "banking77"], help="contract fields, request slots or banking intent")
    p.add_argument("--input", required=True, help="a text file, or - for standard input")
    a = ap.parse_args(argv)

    from litegpt import api                     # torch is slow to import, so --help does not wait for it
    try:
        m = api.load(a.model, device=a.device)
        if a.command == "info":
            print(api.dumps(api.info(m)))
        elif a.command == "generate":
            print(api.generate(m, sys.stdin.read() if a.prompt == "-" else a.prompt, **_options(a)))
        elif a.command == "chat":
            return _chat(m, a, api)
        else:
            text = sys.stdin.read() if a.input == "-" else Path(a.input).read_text(encoding="utf-8")
            res = api.extract(m, a.task, text)
            print(api.dumps(res))
            return 0 if res["valid"] else 2
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except pickle.UnpicklingError:
        print(f"error: {a.model} holds more than a LiteGPT checkpoint should, so it was not loaded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
