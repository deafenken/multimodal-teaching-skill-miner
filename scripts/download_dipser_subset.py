#!/usr/bin/env python3
"""List or range-download an explicit subset of one DIPSER subject ZIP.

Examples:
    python scripts/download_dipser_subset.py --url URL --list
    python scripts/download_dipser_subset.py --url URL --output data/dipser \
        --member labels/labeler_01.json --member labels/labeler_02.json

The command has no "download all" mode.  This is intentional: a subject archive
is roughly a gigabyte and most bytes are RGB frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from teaching_skill_miner.recognition.dipser import (
    HttpRangeSource,
    read_zip_directory,
    write_selected_members,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="ScienceDB direct archive URL")
    parser.add_argument(
        "--size",
        type=int,
        help="Known archive byte size (avoids a HEAD/probe request)",
    )
    parser.add_argument("--list", action="store_true", help="List central-directory members")
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        help="Exact member path to fetch; repeat for multiple files",
    )
    parser.add_argument("--output", type=Path, help="Output root for selected members")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.list and not args.member:
        parser.error("choose --list and/or at least one explicit --member")
    if args.member and args.output is None:
        parser.error("--output is required when --member is used")

    source = HttpRangeSource(
        args.url,
        size=args.size,
        timeout_seconds=args.timeout,
    )
    if args.list:
        directory = read_zip_directory(source)
        print(
            json.dumps(
                [
                    {
                        "name": member.name,
                        "compression": member.compression,
                        "compressed_size": member.compressed_size,
                        "uncompressed_size": member.uncompressed_size,
                        "crc32": f"{member.crc32:08x}",
                    }
                    for member in directory
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    if args.member:
        written = write_selected_members(source, args.output, names=args.member)
        print(
            json.dumps(
                {"written": [str(path.resolve()) for path in written]},
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
