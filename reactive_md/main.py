"""Unified entry point for reactive_md simulations."""

from __future__ import annotations

import sys

# Preserve the existing Python API.
from .apps.lipf6 import main


def cli(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "lipf6":
        from .apps.lipf6 import cli as lipf6_cli
        return lipf6_cli(argv[1:])

    if argv and argv[0] == "a2":
       from .apps.a2 import cli as a2_cli
       return a2_cli(argv[1:])

    # Backward compatibility with the original LiPF6 CLI.
    from .apps.lipf6 import cli as lipf6_cli
    return lipf6_cli(argv)


if __name__ == "__main__":
    cli()
