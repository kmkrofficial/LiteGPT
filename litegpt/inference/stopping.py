"""Stop conditions and UTF-8-safe incremental detokenisation."""
from __future__ import annotations

REPLACEMENT = "�"


def find_stop(text: str, stop_strings) -> int | None:
    """Index where the earliest stop string starts in `text`, or None."""
    hits = [text.find(s) for s in stop_strings if s and s in text]
    return min(hits) if hits else None


def holdback(text: str, stop_strings) -> int:
    """Number of trailing characters that could be the start of a stop string (must not be streamed yet)."""
    best = 0
    for s in stop_strings:
        for n in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:n]):
                best = max(best, n)
                break
    return best


class StreamDecoder:
    """Turns a growing list of token ids into text deltas. A delta is only emitted when the decoded text does not end in an
    incomplete UTF-8 sequence and cannot be the beginning of a stop string."""

    def __init__(self, decode, stop_strings=()):
        self.decode, self.stop_strings = decode, tuple(stop_strings)
        self.ids: list[int] = []
        self.emitted = 0
        self.stopped = False

    def push(self, token_id: int) -> str:
        if self.stopped:
            return ""
        self.ids.append(int(token_id))
        text = self.decode(self.ids)
        cut = find_stop(text, self.stop_strings)
        if cut is not None:
            self.stopped = True
            delta = text[self.emitted:cut]
            self.emitted = cut
            return delta
        if text.endswith(REPLACEMENT):                  # an unfinished multi-byte character
            return ""
        safe = len(text) - holdback(text, self.stop_strings)
        delta = text[self.emitted:safe] if safe > self.emitted else ""
        self.emitted = max(self.emitted, safe)
        return delta

    def flush(self) -> str:
        """Remaining text at the end of generation (held-back characters that turned out not to be a stop string)."""
        if self.stopped:
            return ""
        text = self.decode(self.ids)
        delta = text[self.emitted:]
        self.emitted = len(text)
        return delta
