#!/usr/bin/env python3
"""Compare saved benchmark runs.

    python compare.py results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys

from canit.comparison import build_comparison
from canit.report.compare_text import render_comparison
from canit.results import SchemaError, load_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare.py",
        description="Compare benchmark result files and recommend a model.",
    )
    parser.add_argument("files", nargs="+", help="Results JSON files")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Label to measure regressions against. Defaults to the first file given.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the comparison as JSON")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any candidate regressed against the baseline",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    documents = []
    for path in args.files:
        try:
            documents.append(load_results(path))
        except SchemaError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
        except (OSError, ValueError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)

    if not documents:
        print("no readable result files", file=sys.stderr)
        return 2

    comparison = build_comparison(documents, baseline_label=args.baseline)

    if args.json:
        print(json.dumps(comparison, indent=2, default=str))
    else:
        print(render_comparison(comparison))

    if args.fail_on_regression:
        regressed = any(
            block["metrics"] or block["categories"] or block["scenarios"]
            for block in comparison["regressions"]
        )
        if regressed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
