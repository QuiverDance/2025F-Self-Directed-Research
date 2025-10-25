# vllm/_debug.py
from dataclasses import dataclass
from typing import Any
import inspect
import os
import sys

@dataclass
class DebugFlags:
    # Add future flags here (all booleans only)
    path: bool = False

FLAGS = DebugFlags()

def set_flags(**flags: bool) -> None:
    """Set one or more boolean debug flags. e.g., set_flags(path=True)"""
    for k, v in flags.items():
        if not isinstance(v, bool):
            raise TypeError(f"Debug flag '{k}' must be bool, got {type(v)}")
        if not hasattr(FLAGS, k):
            raise AttributeError(f"Unknown debug flag '{k}'")
        setattr(FLAGS, k, v)

def dprint(flag: str, msg: str, **kv: Any) -> None:
    """Conditional print controlled by boolean flags in FLAGS."""
    if getattr(FLAGS, flag, False):
        frame = inspect.stack()[1]
        loc = f"{os.path.basename(frame.filename)}:{frame.lineno}"
        extra = f" {kv}" if kv else ""
        print(f"[Path Check] {loc} - {msg}{extra}", flush=True, file=sys.stdout)

