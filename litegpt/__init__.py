"""LiteGPT V2."""

__version__ = "2.0.0"

_API = ("load", "generate", "chat", "chat_stream", "extract", "info")


def __getattr__(name):              # the API needs torch, which `litegpt --help` and the data tools should not wait for
    if name in _API:
        from litegpt import api
        return getattr(api, name)
    raise AttributeError(f"module 'litegpt' has no attribute {name!r}")
