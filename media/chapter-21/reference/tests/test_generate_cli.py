#!/usr/bin/env python3
"""Exercise C generation and byte I/O; Python does not evaluate any model."""
from __future__ import annotations

import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def zero_weight_payload(checkpoint: bytes) -> bytes:
    """Create a diagnostic asset with zero weights, retaining format metadata."""
    data = bytearray(checkpoint)
    parameters = struct.unpack_from("<Q", data, 72)[0]
    offset = 256
    for _ in range(parameters):
        rows, cols = struct.unpack_from("<QQ", data, offset + 64)
        offset += 88
        count = rows * cols * 4
        if offset + count > len(data):
            raise AssertionError("invalid fixture directory")
        data[offset:offset + count] = bytes(count)
        offset += count
    return bytes(data)

def main() -> None:
    train = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    generate = Path(sys.argv[2] if len(sys.argv) > 2 else "build/release/generate").resolve()
    engine = sys.argv[3] if len(sys.argv) > 3 else "full"
    if engine not in ("full", "cache", "tiled"):
        raise ValueError("engine must be full, cache or tiled")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/byte-check.conf"
    with tempfile.TemporaryDirectory(prefix="dl-generate-") as directory:
        work = Path(directory)
        text = work / "text"
        subprocess.run([sys.executable, str(root / "scripts/make_training_fixture.py"), str(text)],
                       check=True, capture_output=True, text=True)
        checkpoint = work / "model.bin"
        subprocess.run([str(train), "new", str(config), str(text / "train.bin"),
                        str(text / "valid.bin"), str(checkpoint), "12", "12", "2",
                        "random", "blocked"], check=True, capture_output=True, text=True, timeout=60)
        original = checkpoint.read_bytes()
        shutil.rmtree(text)  # Generation MUST NOT need the original training files.
        prompt = work / "prompt.bin"
        prompt.write_bytes(b"A")

        def run(output: Path, *, new: str = "15", capacity: str = "16", mode: str = "sample",
                temperature: str = "0.8", seed: str = "42", backend: str = "blocked",
                source: Path = checkpoint, fail: bool = False, preexec_fn=None,
                inference: str = engine):
            command = [str(generate), str(config), str(source), str(prompt), str(output),
                       capacity, new, mode, temperature, seed, backend]
            if inference != "legacy":
                command.append(inference)
            result = subprocess.run(command, capture_output=True, text=True, timeout=60,
                                    preexec_fn=preexec_fn)
            if (result.returncode != 0) != fail:
                raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
            return result

        a, b, c = work / "a.bin", work / "b.bin", work / "c.bin"
        log_a = run(a)
        run(b)
        run(c, backend="ref")
        if a.read_bytes() != b.read_bytes() or a.read_bytes() != c.read_bytes():
            raise AssertionError("same-seed or backend sequence disagreement")
        alternate, legacy = work / "alternate.bin", work / "legacy.bin"
        run(alternate, inference="cache" if engine == "full" else "full")
        run(legacy, inference="legacy")
        if a.read_bytes() != alternate.read_bytes() or a.read_bytes() != legacy.read_bytes():
            raise AssertionError("inference strategy/legacy output disagreement")
        if f"inference={engine} consumed=15 " not in log_a.stdout:
            raise AssertionError("selected final token was incorrectly counted as consumed")
        if len(a.read_bytes()) != 16 or not a.read_bytes().startswith(b"A"):
            raise AssertionError("wrong prompt/count")
        prefix = work / "prefix.bin"
        run(prefix, new="7")
        if prefix.read_bytes() != a.read_bytes()[:8]:
            raise AssertionError("longer request changed earlier sampling draws")
        print(f"generation cli: {engine}, selected/full/cache/legacy equality, consumed count, seed/backend replay; no datasets")

        greedy1, greedy2 = work / "greedy1.bin", work / "greedy2.bin"
        log1 = run(greedy1, mode="greedy", seed="42")
        log2 = run(greedy2, mode="greedy", seed="999")
        if greedy1.read_bytes() != greedy2.read_bytes() or \
                "sampling_rng=42 " not in log1.stdout or "sampling_rng=999 " not in log2.stdout:
            raise AssertionError("greedy consumed RNG or depended on seed")
        old = a.read_bytes()
        run(a, fail=True)
        if a.read_bytes() != old:
            raise AssertionError("existing output was truncated")
        run(checkpoint, fail=True)
        run(work / "unknown.bin", inference="unknown", fail=True)
        if checkpoint.read_bytes() != original:
            raise AssertionError("checkpoint overwritten")

        rejected = work / "rejected.bin"
        for temperature in ("0", "-1", "nan", "inf"):
            run(rejected, temperature=temperature, fail=True)
        for arguments in ({"new": "-1"}, {"capacity": "0"}, {"new": "4097"},
                          {"seed": "18446744073709551616"}, {"backend": "unknown"}):
            run(rejected, fail=True, **arguments)
        prompt.write_bytes(b"")
        run(rejected, fail=True)
        prompt.write_bytes(b"abcd")
        run(rejected, capacity="3", new="0", fail=True)
        run(rejected, capacity="4", new="1", fail=True)
        if rejected.exists():
            raise AssertionError("preflight failure created output")
        no_new = work / "no-new.bin"
        log = run(no_new, capacity="4", new="0")
        if no_new.read_bytes() != b"abcd" or "sampling_rng=42 " not in log.stdout or "consumed=0 " not in log.stdout:
            raise AssertionError("zero-token request changed bytes or RNG")
        print("generation cli: greedy RNG isolation, zero-new/exact capacity, malformed inputs, no overwrite")

        zeros = work / "zero-weights.bin"
        zeros.write_bytes(zero_weight_payload(original))
        raw = b"\x00\x80\xffA"
        prompt.write_bytes(raw)
        raw_output = work / "raw-output.bin"
        log = run(raw_output, source=zeros, capacity="7", new="3", mode="greedy")
        if raw_output.read_bytes() != raw + b"\x00\x00\x00":
            raise AssertionError("NUL or high-bit bytes lost or changed")
        if "\\x00" not in log.stdout or "\\x80" not in log.stdout or not log.stdout.isascii():
            raise AssertionError("terminal display was not separately escaped")
        damaged = work / "damaged.bin"
        damaged.write_bytes(original[:-1])
        run(rejected, source=damaged, capacity="7", new="3", fail=True)
        if rejected.exists():
            raise AssertionError("failed checkpoint load created output")
        print("generation cli: binary prompt/output preserved; generated NULs are bytes, not terminators")

        try:
            import resource
        except ImportError:
            print("generation cli: OS write-failure injection skipped (resource unavailable)")
        else:
            def limit_output() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (3, 3))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

            failed = run(rejected, source=zeros, capacity="7", new="3", mode="greedy",
                         fail=True, preexec_fn=limit_output)
            if "write_new_file" not in failed.stderr or rejected.exists():
                raise AssertionError("failed write was not reported/removed")
            print("generation cli: actual OS write failure removes only the new incomplete output")
        if checkpoint.read_bytes() != original:
            raise AssertionError("generation mutated the checkpoint")
    print("all generation CLI checks passed")

if __name__ == "__main__":
    main()
