"""Forward historical entry points without duplicating their implementation."""

import importlib
import runpy
import sys


def expose(name: str, target: str) -> None:
    if name == "__main__":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        sys.modules[name] = importlib.import_module(target)
