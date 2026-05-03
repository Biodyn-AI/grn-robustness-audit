"""``python -m fragility <command>`` entrypoint.

Delegates to :mod:`fragility.cli.__main__`.
"""

from .cli.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
