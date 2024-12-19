import threading
from typing import Callable, Literal

from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from .local_callback_handler import LocalCallbackHandler

_callback_handler = None
_lock = threading.Lock()


def get_callback_handler():
    global _callback_handler
    with _lock:
        return _callback_handler


def set_callback_handler(handler: Callable):
    global _callback_handler
    with _lock:
        _callback_handler = handler()


def init_tracking(name: Literal["local", "langfuse"]):
    if name == "local":
        set_callback_handler(LocalCallbackHandler)
    elif name == "langfuse":
        set_callback_handler(LangfuseCallbackHandler)
    else:
        raise ValueError(f"Invalid tracker name: {name}")


__all__ = [
    "LocalCallbackHandler",
    "init_tracking",
    "get_callback_handler",
    "set_callback_handler",
]
