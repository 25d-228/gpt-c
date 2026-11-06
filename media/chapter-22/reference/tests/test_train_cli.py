#!/usr/bin/env python3
"""Exercise the C executable; Python never evaluates a model or gradient."""
from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/byte-check.conf"
    with tempfile.TemporaryDirectory(prefix="dl-train-") as directory:
        work = Path(directory)
        text = work / "text"
        subprocess.run([sys.executable, str(root / "scripts/make_training_fixture.py"),
                        str(text)], check=True, capture_output=True, text=True)
        split, whole = work / "split.bin", work / "whole.bin"

        def run(mode: str, path: Path, stop: int, *, fail: bool = False,
                preexec_fn=None) -> subprocess.CompletedProcess[str]:
            command = [str(binary), mode, str(config), str(text / "train.bin"),
                       str(text / "valid.bin"), str(path), "12", str(stop),
                       "2", "random", "blocked"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    preexec_fn=preexec_fn, timeout=60)
            if (result.returncode != 0) != fail:
                raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
            return result

        run("new", split, 5)
        old = split.read_bytes()
        refused = run("new", split, 5, fail=True)
        if "refuses an existing checkpoint" not in refused.stderr or split.read_bytes() != old:
            raise AssertionError("new replaced an existing checkpoint")
        print("cli: new refuses existing checkpoint")

        # POSIX/macOS/Linux only. SIGXFSZ must be ignored so stdio can report EFBIG.
        try:
            import resource
        except ImportError:
            print("cli: OS short-write injection skipped (resource unavailable)")
        else:
            def limited_file() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

            failed = run("resume", split, 6, fail=True, preexec_fn=limited_file)
            if "dl_checkpoint_save" not in failed.stderr:
                raise AssertionError("failure did not reach checkpoint saving")
            if split.read_bytes() != old or list(work.glob("split.bin.tmp.*")):
                raise AssertionError("short write damaged old file or leaked temporary")
            print("cli: real OS file-size write failure preserves old checkpoint")

        continued = run("resume", split, 12)
        uninterrupted = run("new", whole, 12)
        if split.read_bytes() != whole.read_bytes():
            raise AssertionError("resume and uninterrupted final state differ")
        # Update timing is observational and is not continuation state.
        def updates(text_output: str) -> list[str]:
            return [line.split(" seconds=", 1)[0] for line in text_output.splitlines()
                    if line.startswith("update=")]
        if updates(continued.stdout) != updates(uninterrupted.stdout)[5:]:
            raise AssertionError("resumed losses or optimizer statistics differ")
        print("cli: seven resumed update logs and complete checkpoint bytes match")

        damaged = bytearray((text / "train.bin").read_bytes())
        damaged[0] ^= 1
        (text / "train.bin").write_bytes(damaged)
        previous = split.read_bytes()
        run("resume", split, 12, fail=True)
        if split.read_bytes() != previous:
            raise AssertionError("incompatible data changed checkpoint")
        print("cli: changed same-length corpus rejected")
    print("all training CLI checks passed")


if __name__ == "__main__":
    main()
