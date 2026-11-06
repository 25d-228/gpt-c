#!/usr/bin/env python3
"""Summarize C logs and escaped bytes; never execute a model or infer a missing pass."""
from __future__ import annotations

from pathlib import Path
import re
import sys


def same_files(a: Path, b: Path) -> bool:
    with a.open("rb") as left, b.open("rb") as right:
        while True:
            x, y = left.read(65536), right.read(65536)
            if x != y:
                return False
            if not x:
                return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 scripts/summarize_demo.py DEMO_DIRECTORY")
    root = Path(sys.argv[1])
    for strategy in ("full", "tiled"):
        if not same_files(root / f"{strategy}-whole.ckpt", root / f"{strategy}-split.ckpt"):
            raise ValueError(f"{strategy}: final checkpoints differ")
        lines = (root / f"{strategy}-whole.log").read_text().splitlines()
        before = next(x for x in lines if x.startswith("validation before"))
        after = [x for x in lines if x.startswith("validation update=")][-1]
        print(strategy, before)
        print(strategy, after)
        seen = 0
        with (root / f"{strategy}-loss.tsv").open("x", encoding="ascii") as output:
            output.write("update\tprocessed_targets\ttrain_loss\tlr\n")
            for line in lines:
                if not line.startswith("update="):
                    continue
                fields = dict(x.split("=", 1) for x in line.split())
                seen += int(fields["targets"])
                output.write(f'{fields["update"]}\t{seen}\t{fields["train_loss"]}\t{fields["lr"]}\n')
        print(f"{strategy} processed_targets={seen} checkpoint_resume=byte_equal")
    for sampling in ("greedy", "sample"):
        paths = [root / f"{sampling}-{mode}.bin" for mode in ("full", "cache", "tiled")]
        payloads = [p.read_bytes() for p in paths]
        if len(set(payloads)) != 1:
            raise ValueError("generation strategies differ: inspect logits and greedy margins")
        print(f"{sampling} transcript={payloads[0]!r}")
    for path in sorted(root.glob("*-measure.txt")):
        print(path.name)
        for line in path.read_text().splitlines():
            if line.startswith(("payload ", "phase=", "peak_rss_", "note=")):
                print("  " + line)
    checks = sorted(root.glob("*-check.txt"))
    if not checks:
        raise ValueError("no comparison files")
    for path in checks:
        lines = path.read_text().splitlines()
        if "check_only=no_timing_or_rss_reference_buffers_are_live" not in lines:
            raise ValueError(f"incomplete check: {path}")
        errors = [float(x) for line in lines for x in re.findall(r"max_abs=([\deE+.-]+)", line)]
        if not errors:
            raise ValueError(f"no error bound: {path}")
    print(f"completed_comparison_processes={len(checks)}")


if __name__ == "__main__":
    main()
