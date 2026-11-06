#!/usr/bin/env python3
"""Train/resume/load C executables. Python only edits file fixtures and compares bytes."""
from __future__ import annotations

import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

def main() -> None:
    train = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    generate = Path(sys.argv[2] if len(sys.argv) > 2 else "build/release/generate").resolve()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/byte-check.conf"
    with tempfile.TemporaryDirectory(prefix="dl-tiled-") as directory:
        work = Path(directory)
        text = work / "text"
        subprocess.run([sys.executable, str(root / "scripts/make_training_fixture.py"), str(text)],
                       check=True, capture_output=True, text=True)
        prompt = work / "prompt.bin"
        prompt.write_bytes(b"A")

        def execute(command: list[str], *, fail: bool = False, preexec_fn=None):
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=90, preexec_fn=preexec_fn)
            if (result.returncode != 0) != fail:
                raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
            return result

        def run(mode: str, path: Path, stop: int, *, engine: str = "tiled",
                fail: bool = False, preexec_fn=None):
            return execute([str(train), mode, str(config), str(text / "train.bin"),
                            str(text / "valid.bin"), str(path), "12", str(stop),
                            "2", "random", "blocked", engine],
                           fail=fail, preexec_fn=preexec_fn)

        def infer(path: Path, output: Path, engine: str, *, fail: bool = False):
            return execute([str(generate), str(config), str(path), str(prompt), str(output),
                            "16", "15", "sample", "0.8", "42", "blocked", engine], fail=fail)

        split, whole = work / "split.bin", work / "whole.bin"
        run("new", split, 5)
        original = split.read_bytes()
        if original[:8] != b"DLCKPT02" or original[-8:] != b"ENDCKP02" or \
                struct.unpack_from("<QQQ", original, 256) != (1, 16, 32):
            raise AssertionError("wrong v2 format/policy extension")
        run("resume", split, 6, engine="full", fail=True)
        run("new", split, 6, fail=True)
        if split.read_bytes() != original:
            raise AssertionError("rejected mode/new call changed the checkpoint")
        try:
            import resource
        except ImportError:
            print("tiled cli: OS write-failure injection unavailable")
        else:
            def limited_file() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            failed = run("resume", split, 6, fail=True, preexec_fn=limited_file)
            if "dl_checkpoint_save" not in failed.stderr or split.read_bytes() != original or \
                    list(work.glob("split.bin.tmp.*")):
                raise AssertionError("failed write did not preserve original file/cleanup")
            print("tiled cli: actual OS short write preserved v2 checkpoint")

        continued = run("resume", split, 12)
        uninterrupted = run("new", whole, 12)
        def updates(output: str) -> list[str]:
            return [line.split(" seconds=", 1)[0] for line in output.splitlines()
                    if line.startswith("update=")]
        if split.read_bytes() != whole.read_bytes() or \
                updates(continued.stdout) != updates(uninterrupted.stdout)[5:]:
            raise AssertionError("v2 resume did not reproduce future state")
        print("tiled cli: seven resumed logs and final checkpoint bytes match")
        transcripts = []
        for engine in ("full", "cache", "tiled"):
            output = work / f"{engine}.bin"
            infer(whole, output, engine)
            transcripts.append(output.read_bytes())
        if len(set(transcripts)) != 1:
            raise AssertionError("v2 weight-only loading strategies disagree")
        print("tiled cli: v2 weights generate identical full/cache/tiled fixture bytes")

        damaged = work / "damaged.bin"
        rejected = work / "rejected.bin"
        bad_cases: list[bytes] = []
        for offset, value in ((8, 3), (256, 2), (264, 0), (272, 0)):
            data = bytearray(original)
            struct.pack_into("<Q", data, offset, value)
            bad_cases.append(bytes(data))
        bad_cases.extend(original[:cut] for cut in (256, 260, 264, 272, 279, 280))
        bad_cases.append(original[:-8] + b"ENDCKP01")
        bad_cases.append(original + b"x")
        for content in bad_cases:
            damaged.write_bytes(content)
            run("resume", damaged, 6, fail=True)
            infer(damaged, rejected, "full", fail=True)
            if damaged.read_bytes() != content or rejected.exists():
                raise AssertionError("failed loader changed file or produced output")
        # A different positive tile policy is legal metadata for inference, but
        # does not describe the same continuation arithmetic for training.
        data = bytearray(original)
        struct.pack_into("<Q", data, 264, 8)
        damaged.write_bytes(data)
        run("resume", damaged, 6, fail=True)
        infer(damaged, work / "other-policy.bin", "full")
        print(f"tiled cli: {len(bad_cases)} damaged/truncated files rejected by both loaders; policy mismatch rejected")
    print("all tiled training CLI checks passed")

if __name__ == "__main__":
    main()
