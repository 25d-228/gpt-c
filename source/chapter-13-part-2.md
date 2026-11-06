# Chapter 13 · Part 2 — Bounded preparation and end-to-end checks

A byte cap must constrain bytes actually written, not the number of characters or records inspected. A validation split must come from its intended source, not the remainder of a training batch. This part implements those preparation decisions, then follows the files through the C loader, batch builder, and complete decoder.

**Prerequisites:** [Part 1](https://25d-228.github.io/gpt-c/chapter-13.html) and the unchanged Chapter 12 model/operator sources. This is the second half of the same **2.00-hour** chapter, not additional scope. Concatenate filename-labeled blocks with one blank line between blocks. No external dataset is needed for the offline checks.

## Fix the byte-stream format and the split policy

The [TinyStories dataset card][tinystories] describes synthetic short stories and identifies the dataset license as `cdla-sharing-1.0`. Its repository contains separate [training and validation Parquet files][datafiles]. We use those named splits, not a random split of already overlapping byte windows. The script records the resolved source revision, selected file paths, license identifier, and tool versions with the output. The corpus itself is not part of the book or companion source snapshot.

For each record, read the string field `text`, encode it as strict UTF-8 **without changing it**, and append exactly two newline bytes, `0A 0A`. A record that already ends in newlines keeps them; there is no `strip`, newline conversion, or replacement of encoding errors. Empty strings are valid records and contribute the two separator bytes. Malformed non-string fields and unencodable surrogate data fail preparation rather than being silently dropped.

These two newlines are ordinary byte tokens, not an EOS marker. A training window may cross them from one story into another. The model has no document-boundary attention mask and does not reset its positions at each separator inside a window. Stopping preparation at record boundaries prevents a truncated final story; it does **not** imply that training samples respect story boundaries. This is an explicitly concatenated byte-stream objective, not packed independent documents.

Separate splits prevent this preparer from placing a training record into validation by its own splitting procedure. They do not prove that the source corpus contains no duplicates, similar stories, or shared phrases across splits. We are preserving source split membership, not making a deduplication claim.

## Bound the output, the inspected prefix, and one record

The defaults cap training at 16 MiB and validation at 1 MiB, including separators. They also inspect at most 50,000 training records and 5,000 validation records, respectively. A single encoded record including its separator is limited to 64 KiB. All are preparation limits, not instructions to pad output to an exact size.

When a complete next record does not fit the remaining byte budget, stop. Do not truncate that record, skip it to search for a shorter later one, or exceed the cap by one record. This produces a deterministic prefix of the selected split. With `é` followed by `éé` and an eight-byte cap, the first encoded record occupies four bytes including its separator; the next needs six and does not fit. The correct result has four bytes, not eight, and not ten.

Hugging Face [streaming][streaming] yields an iterable dataset rather than materializing the entire corpus. It does not make a 16 MiB output cap a 16 MiB network or process-memory cap. Parquet decoding, range requests, caching, row groups, and library buffers can read or retain more. The script does not claim a strict bound on those internals. It consumes one record at a time at the application boundary and bounds its own encoded payload; even the character precheck occurs after the source library has produced the Python string.

Create `scripts/prepare_data.py`. Its standard-library imports support the offline fixture and tests. The optional Hub and datasets imports are delayed until the remote path is requested. The annotations describe expected values; they do not replace the explicit runtime checks.


```python title="scripts/prepare_data.py"
#!/usr/bin/env python3
"""Prepare bounded byte streams; never compute model outputs or gradients."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any

DATASET = "roneneldan/TinyStories"
SPLITS = ("train", "validation")
CAPS = {"train": 16 * 1024 * 1024, "validation": 1024 * 1024}
SEPARATOR = b"\n\n"
RECORD_CAP = 64 * 1024
```

`write_split` receives any iterable of mapping-like records. Turning it into an iterator does not materialize it. `next(iterator)` requests one record, and `StopIteration` marks normal source exhaustion. The bounded `range(record_limit)` loop deliberately does not fetch a record beyond the count cap.

The character precheck limits how much new UTF-8 storage this function can create, but the **encoded** length is authoritative. A code point can require multiple bytes. Binary mode `xb` refuses an existing file and performs no encoding or newline conversion; see Python's [binary-I/O documentation][python-io]. Closing the `with` block flushes the file, and an exception from writing or closing propagates to the outer preparation operation.


```python title="scripts/prepare_data.py"
def write_split(path: Path, rows: Iterable[Mapping[str, Any]], *,
                byte_limit: int, record_limit: int, min_bytes: int) -> dict:
    """Write a prefix of WHOLE records. A non-fitting record ends the prefix."""
    if not (0 < min_bytes <= byte_limit and record_limit > 0):
        raise ValueError("invalid byte/record limits")
    written = inspected = total = 0
    stop = "record_limit"
    iterator = iter(rows)
    with path.open("xb") as stream:
        for _ in range(record_limit):
            try:
                row = next(iterator)
            except StopIteration:
                stop = "source_exhausted"
                break
            inspected += 1
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("record must have a string text field")
            # Bound encoding expansion, too; the source library already owns text.
            if len(text) > RECORD_CAP - len(SEPARATOR):
                raise ValueError("record exceeds the character precheck")
            payload = text.encode("utf-8", errors="strict") + SEPARATOR
            if len(payload) > RECORD_CAP:
                raise ValueError("record exceeds the UTF-8 byte cap")
            if len(payload) > byte_limit - total:
                stop = "byte_limit"
                break
            if stream.write(payload) != len(payload):
                raise OSError("short output write")
            total += len(payload)
            written += 1
            if total == byte_limit:
                stop = "byte_limit"
                break
    if total < min_bytes:
        raise ValueError(f"{path.name}: {total} bytes, need at least {min_bytes}")
    return {"bytes": total, "records": written, "inspected_records": inspected,
            "byte_limit": byte_limit, "record_limit": record_limit, "stop": stop}
```

`inspected_records` includes the first non-fitting record; `records` counts only complete written records. If the byte cap is reached exactly, the code stops immediately instead of fetching one more record merely to discover that no room remains. At source exhaustion the output can be much smaller than either limit. The minimum-size check happens after successful close, but still inside the outer temporary-directory transaction.

The helper itself can leave a partial file if called directly and an input fails. The public `prepare` operation below confines that partial file to a temporary directory. This distinction is intentional: one function writes a bounded stream; its caller owns publication of the complete train/validation set.

## Publish two complete files together, or neither

The destination must be new. A rerun does not append, overwrite, mix old validation with new training, or pretend to resume an interrupted source iterator. The script stages both files and metadata under a temporary directory in the destination's parent. Only after every file closes successfully does it rename the prepared subdirectory into place.

[`TemporaryDirectory`][tempfile] removes the staging directory on ordinary exception unwinding, including an input failure while reading validation after training has already finished. The [rename operation][rename] is within one parent filesystem. This is a single-writer local preparation operation, not a concurrent no-replace protocol, a crash-recovery system, or a guarantee of persistence after power loss. There is no `fsync` here. Do not have another process create the same destination concurrently.


```python title="scripts/prepare_data.py"
def prepare(output: Path, streams: Mapping[str, Iterable[Mapping[str, Any]]],
            source: dict, byte_limits: Mapping[str, int],
            record_limits: Mapping[str, int], min_bytes: int) -> dict:
    """Single-writer publication to a NEW local directory; no overwrite/resume."""
    if set(streams) != set(SPLITS):
        raise ValueError("require separate train and validation streams")
    for split in SPLITS:
        if not (0 < min_bytes <= byte_limits[split] <= CAPS[split]):
            raise ValueError(f"invalid {split} byte limit")
        if record_limits[split] <= 0:
            raise ValueError(f"invalid {split} record limit")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staging = Path(tmp) / "prepared"
        staging.mkdir()
        result = {"format": "decoder-lab-byte-v1", "source": source,
                  "encoding": "utf-8", "separator_hex": SEPARATOR.hex(),
                  "record_byte_cap": RECORD_CAP, "min_bytes": min_bytes,
                  "transform": "text encoded strictly, unchanged, then two LF bytes",
                  "splits": {}}
        for split in SPLITS:
            result["splits"][split] = write_split(
                staging / f"{split}.bin", streams[split],
                byte_limit=byte_limits[split], record_limit=record_limits[split],
                min_bytes=min_bytes)
        with (staging / "metadata.json").open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
            f.write("\n")
        # Files have closed successfully. Same-parent rename exposes the set.
        # Single writer: this is not a concurrent no-replace or fsync protocol.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"destination appeared: {output}")
        staging.rename(output)
    return result
```

The metadata is data provenance, not model computation. It records the format, exact separator, limits, final counts, stop reasons, and source information. It contains no model weights, gradients, credentials, or generated token IDs. `ensure_ascii=True` keeps JSON presentation unambiguous without changing the separate binary files.

A symlink destination is refused as well as an existing regular directory. The second existence check catches ordinary interference before rename, but is not a synchronization primitive; the single-writer contract still matters. On normal success, the renamed directory is outside the temporary directory, so leaving the `with` block removes only the empty staging parent.

## Resolve a mutable revision once, then pin both splits

The command may initially request `main`, a tag, or a commit. [`HfApi.dataset_info`][hub-api] returns the resolved commit ID, card metadata, and repository filenames. Resolve it once, require an exact commit ID, select sorted Parquet paths by their **separate** split prefixes, and give that same commit to both streaming loads. The [loading documentation][loading] defines the `revision`, `data_files`, and `split` arguments.

The code deliberately excludes the repository's other raw-text archives instead of relying on automatic format selection. Missing split files or missing license metadata is a preparation error, not a reason to substitute training data for validation. All Hub calls explicitly disable token authentication; the public data path does not require access to the learner's repository or Git credentials.


```python title="scripts/prepare_data.py"
def hub_source(revision: str) -> tuple[dict, dict]:
    # Optional imports: --fixture and unit tests need only the standard library.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi(token=False).dataset_info(DATASET, revision=revision)
    resolved = info.sha
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("Hub did not return an exact commit revision")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    license_id = card.get("license")
    if not isinstance(license_id, str) or not license_id:
        raise ValueError("dataset license metadata is missing; inspect its card")
    files = {split: sorted(s.rfilename for s in (info.siblings or [])
                           if s.rfilename.startswith(f"data/{split}-") and
                           s.rfilename.endswith(".parquet")) for split in SPLITS}
    if not all(files.values()):
        raise ValueError("expected distinct train/validation Parquet files")
    streams = {split: load_dataset(DATASET, data_files=files, split=split,
                                    revision=resolved, streaming=True, token=False)
               for split in SPLITS}
    versions = {"python": sys.version.split()[0]}
    for name in ("datasets", "huggingface_hub", "pyarrow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    source = {"kind": "huggingface-dataset", "dataset": DATASET,
              "requested_revision": revision, "resolved_revision": resolved,
              "url": f"https://huggingface.co/datasets/{DATASET}/tree/{resolved}",
              "license": license_id, "files": files, "versions": versions}
    return source, streams
```

Sorting the filenames fixes the requested file list, and recording it makes that selection visible. Tool versions also matter because a fixed corpus revision does not by itself specify every library's iteration behavior. Repeating a preparation means using the same revision, limits, transformation, file list, and compatible behavior—not merely reusing the word `main` months later.

The two `load_dataset` calls create separate iterable views. Each selects its own named split while sharing the same explicit file mapping and resolved revision. Python never asks for model logits or calls a tensor library here. The source library supplies transport and Parquet decoding only.

The command-line entry point supplies either that remote source or four original ASCII fixture strings. The fixture is deliberately labeled `original-fixture`; it must never be mistaken for downloaded TinyStories data. It contains only 12 bytes per split after separators, so its command specifies a smaller minimum than the real-data default of 129 bytes, which supports $T=128$.


```python title="scripts/prepare_data.py"
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--train-bytes", type=int, default=CAPS["train"])
    parser.add_argument("--validation-bytes", type=int, default=CAPS["validation"])
    parser.add_argument("--train-records", type=int, default=50000)
    parser.add_argument("--validation-records", type=int, default=5000)
    parser.add_argument("--min-bytes", type=int, default=129)
    args = parser.parse_args(argv)
    limits = {"train": args.train_bytes, "validation": args.validation_bytes}
    counts = {"train": args.train_records, "validation": args.validation_records}
    # Reject obvious mistakes before contacting the Hub.
    for split in SPLITS:
        if not (0 < args.min_bytes <= limits[split] <= CAPS[split] and counts[split] > 0):
            parser.error(f"invalid {split} limits")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new directory")
    if args.fixture:
        source = {"kind": "original-fixture", "revision": "fixture-v1",
                  "license": "CC0-1.0", "description": "Original four ASCII strings"}
        streams = {"train": [{"text": "ABCD"}, {"text": "EFGH"}],
                   "validation": [{"text": "wxyz"}, {"text": "mnop"}]}
    else:
        source, streams = hub_source(args.revision)
    result = prepare(args.output, streams, source, limits, counts, args.min_bytes)
    for split in SPLITS:
        info = result["splits"][split]
        print(f"{split}: {info['bytes']} bytes, {info['records']} records, {info['stop']}")
    print(f"metadata: {args.output / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"prepare_data: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
```

The top-level handler reports a failure and exits nonzero; it does not fall back to a fixture after a network error. `KeyboardInterrupt` and `SystemExit` are not caught by this `Exception` handler, while context managers still perform ordinary cleanup during unwinding. A process killed without unwinding can leave a temporary directory; it is not a complete published dataset and should not be used as one.

## Inspect one training sample and one validation batch

The small C application below loads **both** file owners, constructs a $B=2,T=4$ batch, samples training using its explicit seed, and starts validation at cursor zero. It prints escaped source windows rather than dumping raw terminal bytes. This is an inspection tool, not the training application, and its small fixed dimensions keep its output readable.

Create `app/inspect_data.c`. `show` does not calculate targets; it displays the source intervals recorded in a successfully filled batch. The executable tests below separately check that the copied integer arrays agree with those intervals.


```c title="app/inspect_data.c"
#include "dl/data.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int show(const char *label, const DlByteData *data, const DlByteBatch *batch)
{
    for (size_t row = 0; row < batch->batch; ++row) {
        size_t start = batch->starts[row];
        if (printf("%s row %zu start=%zu x=", label, row, start) < 0 ||
            dl_bytes_fprint(stdout, data->bytes + start, batch->time) != DL_OK ||
            printf(" y=") < 0 ||
            dl_bytes_fprint(stdout, data->bytes + start + 1, batch->time) != DL_OK ||
            putchar('\n') == EOF) return 0;
    }
    return 1;
}
```

The cleanup label destroys the batch and both file owners regardless of which construction or output operation failed. `TRY` is a local convenience for checking status-returning calls; it never treats a failed load as an empty valid file. The program also rejects a validation file that cannot produce one full eight-target batch at its chosen dimensions.


```c title="app/inspect_data.c"
int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s TRAIN.bin VALIDATION.bin\n", argv[0]);
        return EXIT_FAILURE;
    }
    DlByteData train = {0}, validation = {0};
    DlByteBatch batch = {0};
    DlRng rng = {42};
    DlStatus status = DL_OK;
    int result = EXIT_FAILURE, has_batch = 0;
    size_t cursor = 0;
#define TRY(call) do { status = (call); if (status != DL_OK) goto done; } while (0)
    TRY(dl_bytes_load(&train, argv[1], 16 * 1024 * 1024));
    TRY(dl_bytes_load(&validation, argv[2], 1024 * 1024));
    TRY(dl_byte_batch_create(&batch, 2, 4));
    TRY(dl_byte_batch_sample(&train, &batch, &rng));
    if (!show("train", &train, &batch)) goto done;
    TRY(dl_byte_batch_next(&validation, &batch, &cursor, &has_batch));
    if (!has_batch) {
        fprintf(stderr, "validation has no full batch of eight targets\n");
        goto done;
    }
    if (!show("validation", &validation, &batch)) goto done;
    if (printf("train RNG=%" PRIu64 "; validation cursor=%zu\n", rng.state, cursor) < 0 ||
        fflush(stdout) == EOF) goto done;
    result = EXIT_SUCCESS;
done:
    if (status != DL_OK) fprintf(stderr, "inspect_data: %s\n", dl_status_name(status));
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&train);
    return result;
#undef TRY
}
```

## Test the byte boundaries before using the model

Create `tests/test_data.c`. The helpers below use explicit failures rather than assertions that might disappear in a release build. `tmpfile` creates a temporary binary stream. `fseek(...,0,SEEK_SET)` switches it from writing to reading at the beginning and is checked; no platform-specific fixed test filename is required.


```c title="tests/test_data.c"
#include "dl/data.h"
#include "dl/model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int require(int yes, const char *where)
{
    if (!yes) fprintf(stderr, "%s\n", where);
    return yes;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "%s: %s; expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

#define CHECK(yes, where) do { if (!require((yes), (where))) goto done; } while (0)
#define STATUS(call, want) do { if (!status_is((call), (want), #call)) goto done; } while (0)
#define OK(call) STATUS((call), DL_OK)

/* Real binary stdio with no persistent test files or platform-specific paths. */
static DlStatus memory_file(DlByteData *out, const unsigned char *bytes,
                             size_t count, size_t limit)
{
    FILE *stream = tmpfile();
    if (stream == NULL) return DL_STATE;
    DlStatus status = DL_STATE;
    if ((count == 0 || fwrite(bytes, 1, count, stream) == count) &&
        fseek(stream, 0, SEEK_SET) == 0)
        status = dl_bytes_read(out, stream, limit);
    if (fclose(stream) != 0 && status == DL_OK) {
        dl_bytes_destroy(out);
        status = DL_STATE;
    }
    return status;
}
```

The first test includes zero, high-bit bytes, and both line-ending bytes. It distinguishes an exact-size file from one exceeding its cap and rejects empty or one-byte content. It also checks that loading into a live owner does not replace the existing allocation. Batch-size overflow is tested before allocation rather than by attempting to exhaust machine memory.


```c title="tests/test_data.c"
static int file_checks(void)
{
    int ok = 0;
    DlByteData data = {0}, rejected = {0};
    DlByteBatch bad = {0};
    const unsigned char raw[] = {0, 127, 128, 255, 0, 10, 13};
    OK(memory_file(&data, raw, sizeof raw, sizeof raw));
    CHECK(data.count == sizeof raw && data.capacity == sizeof raw &&
          memcmp(data.bytes, raw, sizeof raw) == 0, "all raw bytes preserved");
    unsigned char *owner = data.bytes;
    STATUS(memory_file(&data, raw, sizeof raw, sizeof raw), DL_STATE);
    CHECK(data.bytes == owner && data.count == sizeof raw, "live owner unchanged");
    STATUS(memory_file(&rejected, raw, sizeof raw, sizeof raw - 1), DL_RANGE);
    STATUS(memory_file(&rejected, raw, 0, 8), DL_RANGE);
    STATUS(memory_file(&rejected, raw, 1, 8), DL_RANGE);
    STATUS(dl_bytes_load(&rejected, "", 8), DL_STATE);
    CHECK(rejected.bytes == NULL && rejected.count == 0 && rejected.capacity == 0,
          "failed loads leave empty owner");
    STATUS(dl_byte_batch_create(&bad, 0, 3), DL_ARGUMENT);
    STATUS(dl_byte_batch_create(&bad, SIZE_MAX, 2), DL_OVERFLOW);
    STATUS(dl_byte_batch_create(&bad, 1, SIZE_MAX / 2), DL_OVERFLOW);
    CHECK(bad.storage == NULL && bad.count == 0, "batch overflow unchanged");
    ok = 1;
done:
    dl_byte_batch_destroy(&bad);
    dl_bytes_destroy(&rejected);
    dl_bytes_destroy(&data);
    dl_bytes_destroy(&data);
    return ok;
}
```

The next test is deterministic, not a statistical histogram acceptance test. With seed 3 and bound $2^{63}+1$, the rejection threshold is $2^{63}-1$. The first generator word, 2,092,789,425,003,139,053, is below it. The second, 12,918,135,221,727,111,561, is accepted and maps to 3,694,763,184,872,335,752. The asserted state proves that **two** words were consumed. A one-line biased modulo implementation fails even though every returned value would still be in range.


```c title="tests/test_data.c"
static int rng_checks(void)
{
    int ok = 0;
    DlRng rng = {3};
    uint64_t output = 123;
    STATUS(dl_rng_bounded(&rng, 0, &output), DL_ARGUMENT);
    CHECK(rng.state == 3 && output == 123, "rejected bound preserves outputs");
    /* First word 2092789425003139053 is rejected. The second is accepted. */
    OK(dl_rng_bounded(&rng, (UINT64_C(1) << 63) + 1, &output));
    CHECK(output == UINT64_C(3694763184872335752) &&
          rng.state == UINT64_C(4354685564936845357), "rejection consumes two words");
    DlRng a = {42}, b = {42};
    for (size_t i = 0; i < 100; ++i) {
        uint64_t x, y;
        OK(dl_rng_bounded(&a, 5, &x));
        OK(dl_rng_bounded(&b, 5, &y));
        CHECK(x < 5 && x == y && a.state == b.state, "bounded replay");
    }
    OK(dl_rng_bounded(&a, 1, &output));
    CHECK(output == 0, "single possible start");
    /* Small exhaustive analogue: 16 words -> 6 outcomes; reject words 0..3. */
    unsigned int biased[6] = {0}, unbiased[6] = {0};
    for (unsigned int word = 0; word < 16; ++word) {
        ++biased[word % 6];
        if (word >= 4) ++unbiased[word % 6];
    }
    for (size_t i = 0; i < 6; ++i)
        CHECK(unbiased[i] == 2 && biased[i] == (i < 4 ? 3U : 2U), "modulo trace");
    ok = 1;
done:
    return ok;
}
```

`window_checks` calculates the `ABC -> BCD` and `DEF -> EFG` batch independently, then exercises exhaustion, invalid cursors, known random offsets, state replay, split isolation, and the smallest valid file. The 14 saved `size_t` values are $2BT+B=2\times2\times3+2$. They are compared only after a successful fill, so no uninitialized integer storage is read.


```c title="tests/test_data.c"
static int window_checks(void)
{
    int ok = 0, has = 99;
    size_t cursor = 0;
    DlByteData train = {0}, validation = {0}, minimum = {0};
    DlByteBatch batch = {0}, replay = {0}, too_long = {0};
    DlRng rng = {42};
    const unsigned char text[] = "ABCDEFGH", held[] = "wxyz0123";
    OK(memory_file(&train, text, sizeof text - 1, 32));
    OK(memory_file(&validation, held, sizeof held - 1, 32));
    OK(memory_file(&minimum, text, 4, 8));
    OK(dl_byte_batch_create(&batch, 2, 3));
    OK(dl_byte_batch_create(&replay, 2, 3));
    OK(dl_byte_batch_create(&too_long, 1, 8));
    OK(dl_byte_batch_next(&train, &batch, &cursor, &has));
    CHECK(has == 1 && cursor == 6 && batch.starts[0] == 0 && batch.starts[1] == 3,
          "consecutive offsets");
    for (size_t i = 0; i < 6; ++i)
        CHECK(batch.ids[i] == text[i] && batch.targets[i] == text[i + 1],
              "explicit one-byte target shift");
    printf("hand windows: ABC -> BCD; DEF -> EFG; unscored targets=1\n");
    size_t saved[14];
    memcpy(saved, batch.storage, sizeof saved);
    OK(dl_byte_batch_next(&train, &batch, &cursor, &has));
    CHECK(has == 0 && cursor == 6 && memcmp(saved, batch.storage, sizeof saved) == 0,
          "end does not wrap or pad");
    cursor = SIZE_MAX; has = 99;
    STATUS(dl_byte_batch_next(&train, &batch, &cursor, &has), DL_RANGE);
    CHECK(cursor == SIZE_MAX && has == 99, "bad cursor unchanged");
    STATUS(dl_byte_batch_sample(&train, &too_long, &rng), DL_RANGE);
    CHECK(rng.state == 42, "short-for-time failure consumes no RNG");
    OK(dl_byte_batch_sample(&train, &batch, &rng));
    CHECK(batch.starts[0] == 3 && batch.starts[1] == 1, "known seed starts");
    for (size_t row = 0; row < 2; ++row)
        for (size_t t = 0; t < 3; ++t)
            CHECK(batch.ids[row * 3 + t] == text[batch.starts[row] + t] &&
                  batch.targets[row * 3 + t] == text[batch.starts[row] + t + 1],
                  "sample alignment");
    DlRng resume = rng;
    OK(dl_byte_batch_sample(&train, &batch, &rng));
    OK(dl_byte_batch_sample(&train, &replay, &resume));
    CHECK(rng.state == resume.state &&
          memcmp(batch.storage, replay.storage, sizeof saved) == 0, "next-batch replay");
    uint64_t train_state = rng.state;
    cursor = 0;
    OK(dl_byte_batch_next(&validation, &replay, &cursor, &has));
    for (size_t i = 0; i < 6; ++i)
        CHECK(replay.ids[i] == held[i] && replay.targets[i] == held[i + 1],
              "validation uses only its own bytes");
    CHECK(rng.state == train_state, "evaluation cannot consume training RNG");
    OK(dl_byte_batch_sample(&minimum, &batch, &rng));
    CHECK(batch.starts[0] == 0 && batch.starts[1] == 0, "N=T+1 is one valid start");
    cursor = minimum.count - 1;
    OK(dl_byte_batch_next(&minimum, &batch, &cursor, &has));
    CHECK(has == 0 && cursor == 3, "cursor at final byte is exhausted");
    ok = 1;
done:
    dl_byte_batch_destroy(&too_long);
    dl_byte_batch_destroy(&replay);
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&minimum);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&train);
    return ok;
}
```

The display test reads the escaped output back from a temporary file and compares its exact bytes. The actual loaded data contains control bytes and values 128 and 255; the expected display contains only ASCII escapes. A string-length check is appropriate for the **test's explicitly terminated expected string**, not for arbitrary loaded corpus data.


```c title="tests/test_data.c"
static int presentation_checks(void)
{
    int ok = 0;
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "temporary display stream");
    const unsigned char raw[] = {65, 0, 27, 128, 255, 34, 92, 10, 13, 9};
    const char expected[] = "\"A\\x00\\x1B\\x80\\xFF\\\"\\\\\\n\\r\\t\"";
    char actual[128] = {0};
    OK(dl_bytes_fprint(stream, raw, sizeof raw));
    CHECK(fseek(stream, 0, SEEK_SET) == 0, "rewind display");
    size_t count = fread(actual, 1, sizeof actual, stream);
    CHECK(!ferror(stream) && count == strlen(expected) &&
          memcmp(actual, expected, count) == 0, "escaped display has no raw controls");
    ok = 1;
done:
    if (stream != NULL && fclose(stream) != 0) ok = 0;
    return ok;
}
```

## Carry a real byte batch through the C decoder

The integration fixture uses vocabulary 256, one block, width 8, two query heads, one key/value head, and FFN width 12. These small dimensions are independent of the bounded diagnostic configuration in Chapter 12's `test_model` executable; the model library itself already accepts the full byte vocabulary.

It constructs a batch containing zero and high-bit values, runs forward, refills the caller's batch, and then checks that the tape still contains the original copied inputs and targets. Backward must succeed with finite gradients for both matrix backends. This catches data/model disagreement without inventing new numerical operators. It is not a new finite-difference test or a text-training result.


```c title="tests/test_data.c"
static int model_checks(void)
{
    int ok = 0;
    DlByteData data = {0};
    DlByteBatch batch = {0};
    DlModel model = {0};
    DlTrainTape tape = {0};
    DlModelConfig config = {256, 1, 8, 2, 1, 12, 1e-5f, 0.1f, 10000.0, 42};
    const unsigned char raw[] = {65, 0, 128, 255, 66, 10, 67, 68};
    OK(memory_file(&data, raw, sizeof raw, 16));
    OK(dl_byte_batch_create(&batch, 2, 3));
    OK(dl_model_create(&model, config));
    OK(dl_train_tape_create(&tape, &model, 2, 3, 1024 * 1024));
    double first_loss = 0.0;
    for (int backend = 0; backend < 2; ++backend) {
        size_t cursor = 0;
        int has = 0;
        OK(dl_byte_batch_next(&data, &batch, &cursor, &has));
        CHECK(has, "model has a batch");
        double loss = -1.0;
        OK(dl_model_forward(&model, &tape,
                            backend == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED,
                            batch.ids, batch.targets, batch.count, &loss));
        CHECK(isfinite(loss) && loss > 0.0, "finite byte-vocabulary loss");
        if (backend == 0) first_loss = loss;
        else CHECK(fabs(first_loss - loss) <= 1e-6, "model backend loss agreement");
        /* Reuse the caller's storage: forward has already copied the metadata. */
        DlRng rng = {9};
        OK(dl_byte_batch_sample(&data, &batch, &rng));
        for (size_t i = 0; i < 6; ++i)
            CHECK(tape.ids[i] == raw[i] && tape.targets[i] == raw[i + 1],
                  "model owns copied IDs and does not shift targets again");
        OK(dl_model_backward(&model, &tape, DL_SET));
        CHECK(model.gradients_valid, "complete byte-model backward");
        for (size_t p = 0; p < model.param_count; ++p)
            for (size_t i = 0; i < model.params[p].grad.count; ++i)
                CHECK(isfinite(model.params[p].grad.data[i]), "finite byte-model gradients");
    }
    printf("byte model: V=256, targets=6, loss=%.9f (both backends)\n", first_loss);
    ok = 1;
done:
    dl_train_tape_destroy(&tape);
    dl_model_destroy(&model);
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&data);
    return ok;
}
```

The first target is raw byte 0, and another is 255. Passing through a signed `char` or adding an invented special-token offset would break the exact metadata checks or the vocabulary bounds. The model receives all six targets already shifted; it has no reason to alter them.

The test runner executes every group even if an earlier group failed, while each group cleans up its own resources. A failed group still causes a nonzero process exit.


```c title="tests/test_data.c"
int main(void)
{
    int ok = 1;
    ok &= file_checks();
    ok &= rng_checks();
    ok &= window_checks();
    ok &= presentation_checks();
    ok &= model_checks();
    if (!ok) return EXIT_FAILURE;
    puts("data checks passed");
    return EXIT_SUCCESS;
}
```

## Check preparation without a network dependency

Create `tests/test_prepare.py`. It imports the local script from its sibling directory, uses temporary output roots, and provides only original synthetic fixtures. Python's `unittest` is sufficient; the offline tests do not import `datasets` or download a corpus.


```python title="tests/test_prepare.py"
#!/usr/bin/env python3
"""Offline preparation checks; all text fixtures are original."""
from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import CAPS, RECORD_CAP, hub_source, main, prepare, write_split


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = {"kind": "original-fixture", "revision": "test-v1"}

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, streams, output=None, **changes):
        options = {"byte_limits": {"train": 40, "validation": 40},
                   "record_limits": {"train": 5, "validation": 5}, "min_bytes": 2}
        options.update(changes)
        return prepare(output or self.root / "data", streams, self.source, **options)
```

The first group checks split membership, refusal to overwrite, raw encoding, the distinction between character and byte counts, and exact stopping without requesting an extra record. The generator that raises on its second request is a precise test of consumption behavior, not just a check of the final file size.


```python title="tests/test_prepare.py"
    def test_fixture_and_no_overwrite(self):
        out = self.root / "fixture"
        self.assertEqual(main(["--fixture", "--output", str(out), "--min-bytes", "5"]), 0)
        self.assertEqual((out / "train.bin").read_bytes(), b"ABCD\n\nEFGH\n\n")
        self.assertEqual((out / "validation.bin").read_bytes(), b"wxyz\n\nmnop\n\n")
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["source"]["kind"], "original-fixture")
        self.assertEqual(meta["splits"]["train"]["bytes"], 12)
        with self.assertRaises(FileExistsError):
            self.run_prepare({"train": [], "validation": []}, output=out)
        self.assertEqual((out / "train.bin").read_bytes(), b"ABCD\n\nEFGH\n\n")

    def test_raw_encoding_and_separator(self):
        text = "A\x00\r\né中"
        data = [{"text": text}]
        self.run_prepare({"train": data, "validation": [{"text": "V"}]})
        self.assertEqual((self.root / "data/train.bin").read_bytes(),
                         b"A\x00\r\n\xc3\xa9\xe4\xb8\xad\n\n")

    def test_whole_records_not_character_counts(self):
        seen = []
        def rows():
            for text in ("é", "éé", "A"):
                seen.append(text)
                yield {"text": text}
        path = self.root / "bounded.bin"
        result = write_split(path, rows(), byte_limit=8, record_limit=10, min_bytes=2)
        self.assertEqual(path.read_bytes(), b"\xc3\xa9\n\n")
        self.assertEqual(seen, ["é", "éé"])
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["inspected_records"], 2)
        self.assertEqual(result["stop"], "byte_limit")

    def test_record_count_does_not_fetch_one_more(self):
        def rows():
            yield {"text": "A"}
            raise AssertionError("must not fetch another record")
        result = write_split(self.root / "one.bin", rows(), byte_limit=100,
                             record_limit=1, min_bytes=2)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["stop"], "record_limit")

    def test_exact_byte_limit_does_not_fetch_one_more(self):
        def rows():
            yield {"text": "A"}
            raise AssertionError("must stop at exact byte cap")
        result = write_split(self.root / "exact.bin", rows(), byte_limit=3,
                             record_limit=100, min_bytes=3)
        self.assertEqual(result["bytes"], 3)
        self.assertEqual(result["stop"], "byte_limit")
```

An injected input error occurs after training has already been written to staging. Neither final file may appear, and the temporary staging root must be removed during unwinding. Other cases reject empty/short output, a missing string field, invalid surrogate text, oversized records, absent splits, and output caps above the documented maximums.


```python title="tests/test_prepare.py"
    def test_failure_does_not_publish_either_split(self):
        def broken():
            yield {"text": "V"}
            raise OSError("injected input failure")
        with self.assertRaises(OSError):
            self.run_prepare({"train": [{"text": "TRAIN"}], "validation": broken()})
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_empty_short_and_invalid_records(self):
        for rows in ([], [{"text": "A"}], [{"text": None}],
                     [{"text": "\ud800"}], [{"text": "A" * RECORD_CAP}],
                     [{"text": "中" * (RECORD_CAP // 2)}]):
            with self.subTest(kind=str(rows)[:30]):
                with self.assertRaises((ValueError, UnicodeError)):
                    self.run_prepare({"train": rows, "validation": [{"text": "V"}]},
                                     min_bytes=5)
                self.assertFalse((self.root / "data").exists())

    def test_split_and_cap_validation(self):
        with self.assertRaises(ValueError):
            self.run_prepare({"train": [{"text": "A"}]})
        streams = {"train": [{"text": "A"}], "validation": [{"text": "V"}]}
        for limits in ({"train": CAPS["train"] + 1, "validation": 40},
                       {"train": 40, "validation": CAPS["validation"] + 1},
                       {"train": 0, "validation": 40}):
            with self.assertRaises(ValueError):
                self.run_prepare(streams, byte_limits=limits)
        self.assertFalse((self.root / "data").exists())
```

The final test replaces the two optional imports with small stand-ins and records their calls. It verifies exact revision pinning, separate split selection, sorted file lists, streaming mode, and disabled authentication. It is explicitly an **interface-level test**: it does not establish that an actual remote request or Parquet decode succeeded. Keeping that distinction prevents an offline mock from being reported as a corpus download.


```python title="tests/test_prepare.py"
    def test_hub_adapter_pins_both_splits(self):
        # Interface-level test, not evidence of a network transfer or Parquet decode.
        revision = "a" * 40
        calls = []
        info = types.SimpleNamespace(
            sha=revision,
            card_data=types.SimpleNamespace(to_dict=lambda: {"license": "cdla-sharing-1.0"}),
            siblings=[types.SimpleNamespace(rfilename=s) for s in
                      ("data/train-b.parquet", "data/validation-a.parquet",
                       "data/train-a.parquet", "TinyStories-train.txt")])
        class Api:
            def __init__(self, **kwargs):
                self.options = kwargs
            def dataset_info(self, name, **kwargs):
                calls.append(("info", name, kwargs, self.options))
                return info
        def load(name, **kwargs):
            calls.append(("load", name, kwargs))
            return [{"text": kwargs["split"]}]
        modules = {"datasets": types.SimpleNamespace(load_dataset=load),
                   "huggingface_hub": types.SimpleNamespace(HfApi=Api)}
        with patch.dict(sys.modules, modules):
            source, streams = hub_source("main")
        self.assertEqual(source["resolved_revision"], revision)
        self.assertEqual(source["license"], "cdla-sharing-1.0")
        self.assertEqual(set(streams), {"train", "validation"})
        self.assertEqual(calls[0][2]["revision"], "main")
        self.assertFalse(calls[0][3]["token"])
        self.assertEqual([c[2]["split"] for c in calls[1:]], ["train", "validation"])
        for _, _, options in calls[1:]:
            self.assertEqual(options["revision"], revision)
            self.assertTrue(options["streaming"])
            self.assertFalse(options["token"])
            self.assertEqual(options["data_files"]["train"],
                             ["data/train-a.parquet", "data/train-b.parquet"])
            self.assertEqual(options["data_files"]["validation"],
                             ["data/validation-a.parquet"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

## Integrate the build without changing numerical behavior

Replace the root Makefile with this version. The data module joins `CORE`, its header becomes a dependency, and `test_data` joins the cumulative C suite. `inspect_data` is built in all three modes but requires explicit file arguments, so it is not blindly run by `make test`. The separate `test-assets` target runs only offline Python checks. The numerical compiler flags and all earlier tests remain unchanged.


```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/rng.c src/model.c src/data.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/rng.h include/dl/model.h include/dl/data.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data

.PHONY: all debug sanitize release test test-assets clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

test-assets:
	python3 tests/test_prepare.py

clean:
	rm -rf build
```

Use the native toolchain already established in Chapter 1. The default is `xcrun clang`; another compatible compiler can be selected with `CC=...`. The asset script uses Python type annotations and standard-library APIs available in Python 3.10 or newer; compatible newer versions are acceptable. There is no requirement to install a particular “LTS” Python or compiler.

```sh title="Terminal — compile and run the chapter"
make build/debug/test_data build/sanitize/test_data build/release/test_data
./build/debug/test_data
./build/sanitize/test_data
./build/release/test_data
make test-assets
make test
```

The new C executable produced the following output on the checked Clang 17 x86_64 Linux builds:

```text
hand windows: ABC -> BCD; DEF -> EFG; unscored targets=1
byte model: V=256, targets=6, loss=5.653440091 (both backends)
data checks passed
```

All three build modes and the inherited cumulative tests passed without compiler warnings. A leak-enabled sanitizer run of `test_data` reported no leaks. The nine offline Python tests passed under Python 3.13.5. The reported loss is a tiny randomly initialized integration fixture, not a held-out corpus result, and its final digits are not a portability requirement. No M4/macOS run is implied.

## Run the local ASCII fixture from preparation to inspection

This path uses no network and no optional Python dependency. The parent `data` directory may already exist, but the selected output directory must not.

```sh title="Terminal — small original fixture"
python3 scripts/prepare_data.py --fixture --output data/ascii --min-bytes 5
make build/debug/inspect_data
./build/debug/inspect_data data/ascii/train.bin data/ascii/validation.bin
```

The two files contain exactly these bytes:

```text
train.bin       ABCD\n\nEFGH\n\n
validation.bin  wxyz\n\nmnop\n\n
```

These lines are escaped displays, not file contents to copy literally with backslashes. The preparation command writes the actual newline bytes. Its expected counts are 12 bytes and two records in each split, with `source_exhausted` as the stop reason.

The checked inspection output was:

```text
train row 0 start=5 x="\nEFG" y="EFGH"
train row 1 start=3 x="D\n\nE" y="\n\nEF"
validation row 0 start=0 x="wxyz" y="xyz\n"
validation row 1 start=4 x="\n\nmn" y="\nmno"
train RNG=4354685564936845396; validation cursor=8
```

Trace the first random row: byte 5 is the second separator newline, byte 6 is `E`, and bytes 7 and 8 are `F` and `G`. Its target starts one byte later and ends at byte 9, `H`. Nothing decodes the separator as a special token or prevents this window from crossing a record boundary.

Validation begins at zero regardless of how many training draws occurred. The inspection program prints only its first batch; an evaluation loop would repeatedly call `dl_byte_batch_next`, stopping at `has_batch=0`, and divide accumulated token loss by the **actual scored target count**, not by file size.

## Prepare the bounded TinyStories files locally

The remote path additionally needs maintained compatible releases of `datasets` and `huggingface_hub`, with their normal dependencies. Use a dedicated environment rather than treating asset dependencies as C compiler requirements. Network transfer is needed only for preparation; the resulting C executable reads ordinary local files.

```sh title="Terminal — optional asset environment and real split preparation"
python3 -m venv .venv-assets
.venv-assets/bin/python -m pip install datasets huggingface_hub
.venv-assets/bin/python scripts/prepare_data.py \
    --output data/tinystories-byte \
    --train-bytes 16777216 --validation-bytes 1048576 \
    --train-records 50000 --validation-records 5000 --min-bytes 129
./build/debug/inspect_data \
    data/tinystories-byte/train.bin data/tinystories-byte/validation.bin
```

The command records the exact resolved revision in `data/tinystories-byte/metadata.json`. For a repeat preparation, pass that recorded value to `--revision` and use a **new** output directory. Preserve the metadata beside the files. Read the dataset's linked card and license information before redistributing source data; the course package does not redistribute the prepared corpus.

Successful output is bounded, not predetermined: record sizes decide how close each file comes to its cap. The full training file can be smaller than 16 MiB and still be correct. The command must fail rather than invent a revision, ignore a missing license field, fall back to a different split, or silently supply offline fixtures after a remote failure.

**Verification boundary:** the actual TinyStories transfer and Parquet decode were not executed in the authoring environment. Its container DNS access was unavailable and `datasets` was not installed. The source layout and current public interfaces were read in the linked primary documentation; local preparation, deterministic failure injection, interface tests, and C integration were executed. A remote transport or dependency failure is therefore a distinct local check, not a claimed completed download. Do not compare a fixture loss with a real corpus result.

## Make a correct boundary variation

Prepare the same original fixture with an exact 12-byte cap in a fresh directory:

```sh title="Terminal — exact cap without truncating a record"
python3 scripts/prepare_data.py --fixture --output data/ascii-exact \
    --train-bytes 12 --validation-bytes 12 --min-bytes 5
cmp data/ascii/train.bin data/ascii-exact/train.bin
cmp data/ascii/validation.bin data/ascii-exact/validation.bin
```

Both comparisons should succeed; the stop reason changes from `source_exhausted` to `byte_limit`. This variation was executed successfully. Predict the result with an 11-byte cap: only the first six-byte record fits in each split. The result must not be an 11-byte prefix ending midway through the second record. At that size, the inspection application's fixed eight-target validation batch is too large and should fail; that is a workload-size mismatch, not a reason to pad the data.

## Break one data invariant at a time

Make each change in a temporary copy, rebuild the relevant test, then restore the original. These are diagnostic experiments, not retained changes.

**Remove the shift.** In `copy_window`, use `start + t` for targets. `window_checks` rejects the explicit one-byte alignment, and the model integration rejects the copied target metadata. All memory accesses can be legal while the training objective is wrong.

**Add one too many starts.** Change the sampling bound from `count - time` to `count - time + 1`. The known-seed test rejects the changed offsets. In general the extra start needs a target beyond the valid byte count. Because a file's allocation can be larger than its valid count, an allocation sanitizer need not catch this logical overread; tests must respect `count`, not merely `capacity`.

**Skip rejection.** Replace the rejection condition in `dl_rng_bounded` with an always-false condition, retaining the remainder. The forced two-word rejection case fails. A range-only check would not find this defect, and ordinary small-bound sampling would almost never exercise the branch.

**Accept the extra byte.** In the loader, turn the oversize probe's `DL_RANGE` into `DL_OK`. The exact oversize fixture now returns success where a rejection is required. Nothing has overflowed the allocated buffer: the bug is silent data truncation.

**Count characters as bytes.** In Python, compare `len(text) + len(SEPARATOR)` with the remaining budget instead of `len(payload)`. The `é`/`éé` case writes ten bytes under an eight-byte cap and fails its exact-output test. ASCII-only fixtures cannot expose this mistake.

All four C variants failed their intended checks on the sanitizer build without AddressSanitizer or undefined-behavior reports. The Python character-count variant failed its dedicated test. A clean rebuild passed after restoration. These are semantic and state defects; they do not need a memory fault to invalidate a training result.

## Diagnose the data boundary before changing the model

A loss that falls unusually easily can come from unshifted targets. Print a small window's numerical IDs and source offsets before adjusting the learning rate. A failure only on bytes above 127 points to signed-character conversion or a mismatched vocabulary. A negative `EOF` accidentally converted into a byte points to storing `fgetc` in `char` before checking it.

A repeated final validation batch points to ignoring `has_batch`. A changing training sequence after an evaluation call points to shared random state. A changed sequence after changing model dimensions points to sharing the initialization RNG. Replay requires the saved data RNG state and unchanged source bytes, dimensions, and sampling algorithm; restoring a seed alone restarts the stream rather than continuing it.

A file accepted at its cap without an EOF probe may be a silently truncated prefix. A prepared output above its cap points to counting records or characters rather than encoded bytes and separators. A missing output directory after a preparation failure is expected: the script refuses to expose a half-prepared train/validation set. Inspect the reported failure; do not rename the staging folder into place manually.

The decoder now has explicit, reproducible training examples and a separate validation traversal. It has not yet trained on a sustained corpus run. **Chapter 14 — AdamW, accumulation, and gradient clipping** will use these batches to define a checked optimizer update, including token-weighted microbatch accumulation and a single update for every unique parameter.

[tinystories]: https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/README.md
[datafiles]: https://huggingface.co/datasets/roneneldan/TinyStories/tree/main/data
[streaming]: https://huggingface.co/docs/datasets/stream
[loading]: https://huggingface.co/docs/datasets/loading
[hub-api]: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.dataset_info
[python-io]: https://docs.python.org/3/library/io.html#binary-i-o
[tempfile]: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
[rename]: https://docs.python.org/3/library/os.html#os.rename

