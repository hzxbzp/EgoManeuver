"""Locate a script of the package by file name, so that scripts in data/, model/ and agent/ can load each other."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def script_path(name):
    for part in ("data", "model", "agent"):
        p = os.path.join(ROOT, part, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)
