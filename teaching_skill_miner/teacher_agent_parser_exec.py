"""Exec one untrusted local document parser with TCP networking denied."""

from __future__ import annotations

import os
import sys

from .teacher_agent_worker_isolation import (
    WorkerIsolationError,
    install_parser_network_isolation,
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "--" or len(args) < 2:
        return 64
    try:
        install_parser_network_isolation()
    except WorkerIsolationError:
        return 70
    os.execv(args[1], args[1:])
    return 70  # pragma: no cover - exec never returns.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
