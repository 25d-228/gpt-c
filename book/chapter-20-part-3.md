# Chapter 20 — Executable import fixtures and the C inspection path

This continues [the loader](https://25d-228.github.io/gpt-c/chapter-20-part-2.html). Start with small synthetic files whose bytes can be independently predicted. A 500-MiB file is a poor place to discover that an offset is measured in elements rather than bytes.

## Generate recognizable values, not plausible language weights

Add `tests/import_fixture.py`. Its configuration and source tensor names are written independently of the converter's `tensor_map`. The fixture has vocabulary 16, two blocks, hidden width 8, four query heads, two key/value heads, feed-forward width 12, and position limit 8.

Parameter number $p$ and coordinate $i$ determine a BF16 pattern using `0x3d00 + (97*p+i) % 256`; every third combined index gets a negative sign. This keeps values finite and makes tensor swaps and layout errors visible. For example, parameter zero coordinate zero is negative `0x3d00`, or $-0.03125$. Coordinate one is positive `0x3d01`, or $0.031494140625$.

`fixture_source` emits parameters in reverse physical order and reverse JSON order. The converter must recover canonical order from names, not from iteration. F32 fixture values use the same exact representable values, allowing a byte-for-byte comparison of the two converted outputs.

```python title="tests/import_fixture.py"
"""Small synthetic assets, never pretrained values. No network or model execution."""
from __future__ import annotations

import json
from pathlib import Path
import struct
```


```python title="tests/import_fixture.py"
def raw_config():
    return {
        "architectures": ["LlamaForCausalLM"], "attention_bias": False,
        "attention_dropout": 0.0, "bos_token_id": 0, "eos_token_id": 0,
        "hidden_act": "silu", "hidden_size": 8, "initializer_range": 0.125,
        "intermediate_size": 12, "is_llama_config": True,
        "max_position_embeddings": 8, "model_type": "llama",
        "num_attention_heads": 4, "num_hidden_layers": 2,
        "num_key_value_heads": 2, "pretraining_tp": 1, "rms_norm_eps": 1e-5,
        "rope_interleaved": False, "rope_scaling": None, "rope_theta": 10000,
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "transformers_version": "fixture", "use_cache": True, "vocab_size": 16
    }


def fixture_records():
    records = [("model.embed_tokens.weight", [16, 8])]
    suffixes = (
        ("input_layernorm.weight", [8]), ("self_attn.q_proj.weight", [8, 8]),
        ("self_attn.k_proj.weight", [4, 8]), ("self_attn.v_proj.weight", [4, 8]),
        ("self_attn.o_proj.weight", [8, 8]), ("post_attention_layernorm.weight", [8]),
        ("mlp.gate_proj.weight", [12, 8]), ("mlp.up_proj.weight", [12, 8]),
        ("mlp.down_proj.weight", [8, 12])
    )
    for layer in range(2):
        records.extend((f"model.layers.{layer}.{name}", shape) for name, shape in suffixes)
    return records + [("model.norm.weight", [8])]
```

The test generator may hold this tiny payload in a Python byte array. That is deliberately different from the production converter's bounded streaming path; the fixture contains only 1,128 values. JSON header padding is included in the declared length, so the payload begins after the full padded header.

```python title="tests/import_fixture.py"
def fixture_source(dtype="BF16"):
    records = fixture_records()
    # Deliberately reverse physical and JSON order: converter must use names.
    payload = bytearray()
    header = {"__metadata__": {"format": "pt"}}
    for p in range(len(records)-1, -1, -1):
        name, shape = records[p]
        count = shape[0] * (shape[1] if len(shape) == 2 else 1)
        begin = len(payload)
        for i in range(count):
            bf = 0x3d00 + (97 * p + i) % 256
            if (p + i) % 3 == 0:
                bf |= 0x8000
            payload.extend(struct.pack("<H", bf) if dtype == "BF16"
                           else struct.pack("<I", bf << 16))
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[begin, len(payload)])
    return header, bytes(payload)


def write_source(directory, header=None, payload=None, config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if header is None:
        header, payload = fixture_source()
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (directory / "config.json").write_text(json.dumps(raw_config() if config is None else config))
    return directory
```


```python title="tests/import_fixture.py"
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 tests/import_fixture.py NEW_DIRECTORY")
    if Path(sys.argv[1]).exists():
        raise SystemExit("refusing to overwrite an existing fixture directory")
    write_source(sys.argv[1])
```

## Independently write and corrupt the C wire format

Add `tests/test_import.c`. With no arguments it constructs an independent binary fixture, loads it, and exercises malformed variants. With one argument it verifies a file produced by the Python converter. This connects two implementations at the byte boundary rather than testing only one serializer against its own parser.

The macros below report explicit failures even in optimized builds; they do not depend on `assert` being enabled. All failure paths jump to cleanup. `put_u` writes bytes explicitly and shares no production checkpoint helper. `wire_fixture` uses the existing empty-weight directory only to obtain local names/shapes; fixed header counts are independent literal fixture expectations.

```c title="tests/test_import.c"
#define _POSIX_C_SOURCE 200809L
#include "dl/checkpoint.h"
#include "dl/cache.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #condition); goto done; \
} } while (0)
#define OK(call) CHECK((call) == DL_OK)
```

The C fixture-value function and the Python fixture use the same documented pattern, but perform conversion through different APIs. Every loaded coordinate will be compared, including all scales and both non-square projection orientations.


```c title="tests/test_import.c"
static DlModelConfig fixture_config(void)
{
    return (DlModelConfig){.vocab=16, .layers=2, .width=8, .query_heads=4,
        .kv_heads=2, .ffn_width=12, .epsilon=1e-5f, .init_std=0.125f,
        .rope_theta=10000.0, .seed=42};
}
```


```c title="tests/test_import.c"
static float fixture_value(size_t parameter, size_t coordinate)
{
    uint32_t bf = UINT32_C(0x3d00) + (uint32_t)((97 * parameter + coordinate) % 256);
    if ((parameter + coordinate) % 3 == 0) bf |= UINT32_C(0x8000);
    uint32_t bits = bf << 16;
    float value = 0.0f;
    memcpy(&value, &bits, 4);
    return value;
}
```

The writer's small helpers are complete. A short write or failed close is a failed fixture construction; tests do not proceed with a partially written file.


```c title="tests/test_import.c"
static void put_u(unsigned char *p, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i) {
        p[i] = (unsigned char)(value & 255);
        value >>= 8;
    }
}
```


```c title="tests/test_import.c"
static void put_f(unsigned char *p, float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, 4);
    put_u(p, bits, 4);
}
```


```c title="tests/test_import.c"
static int write_bytes(const char *path, const unsigned char *data, size_t size)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) return 0;
    int result = fwrite(data, 1, size, file) == size;
    if (fclose(file) != 0) result = 0;
    return result;
}
```

The complete file is only 6,568 bytes: a 128-byte header, 1,920-byte directory, 4,512-byte weight payload, and eight-byte trailer. The extra allocation byte belongs only to the trailing-data test and is not part of a valid file.


```c title="tests/test_import.c"
static unsigned char *wire_fixture(size_t *length)
{
    DlWeights shape = {0};
    DlModelConfig c = fixture_config();
    if (dl_weights_create(&shape, c, 1024 * 1024) != DL_OK) return NULL;
    size_t payload = 128 + 96 * shape.param_count;
    size_t size = payload + 4 * shape.arena.count + 8;
    unsigned char *data = calloc(size + 1, 1); /* Extra byte for trailing-data test. */
    if (data != NULL) {
        memcpy(data, "DLWGHT01", 8);
        const uint64_t fields[] = {1,1,16,2,8,4,2,12,20,1128,8};
        for (size_t i = 0; i < 11; ++i) put_u(data + 8 + 8 * i, fields[i], 8);
        put_f(data + 96, c.epsilon);
        put_f(data + 100, c.init_std);
        uint64_t theta = 0;
        memcpy(&theta, &c.rope_theta, 8);
        put_u(data + 104, theta, 8);
        put_u(data + 112, 42, 8);
        put_u(data + 120, payload, 8);
        size_t at = payload;
        for (size_t p = 0; p < shape.param_count; ++p) {
            size_t record = 128 + 96 * p;
            DlMatrix v = shape.params[p].value;
            memcpy(data + record, shape.params[p].name, 64);
            put_u(data + record + 64, v.rows, 8);
            put_u(data + record + 72, v.cols, 8);
            put_u(data + record + 80, at, 8);
            put_u(data + record + 88, v.rows * v.cols, 8);
            for (size_t i = 0; i < v.rows * v.cols; ++i) {
                put_f(data + at, fixture_value(p, i));
                at += 4;
            }
        }
        memcpy(data + at, "ENDWGT01", 8);
        *length = size;
    }
    dl_weights_destroy(&shape);
    return data;
}
```

A rejected load must leave more than a null payload pointer: directory pointer, record count, readiness, position limit, and configuration must remain empty. The helper destroys any unexpected candidate too, so a failing test does not mask the issue with an avoidable leak.


```c title="tests/test_import.c"
static int rejected(const char *path, DlStatus expected, uint64_t file_budget,
                    size_t weight_budget)
{
    DlWeights out = {0};
    DlWeights before = out;
    DlStatus status = dl_checkpoint_load_weights(path, &out, fixture_config(),
                                                 file_budget, weight_budget);
    int result = status == expected && out.params == before.params &&
        out.arena.data == before.arena.data && out.param_count == 0 &&
        out.position_limit == 0 && out.ready == 0 && out.config.vocab == 0;
    if (!result) fprintf(stderr, "rejection: got %s, wanted %s\n",
                         dl_status_name(status), dl_status_name(expected));
    dl_weights_destroy(&out);
    return result;
}
```

### Check values and then execute every inference strategy

The `loaded` test requires the exact weight payload budget to succeed. It checks all 1,128 values, verifies that every matrix borrows the one arena, and rejects reloading into the live owner without altering its address.

It then constructs full-prefix, tiled, and cached workspaces at the exact eight-position bound. Both matrix backends process eight prefixes. At each prefix, every vocabulary logit is compared against the full-prefix result, yielding $2\cdot8\cdot16\cdot2=512$ scalar comparisons. The tolerance is $2\times10^{-6}+2\times10^{-5}|a|$, where $a$ is the full-prefix logit. This checks execution integration, not agreement with a separately implemented pretrained model.

```c title="tests/test_import.c"
static int loaded(const char *path)
{
    int result = 0;
    DlWeights w = {0};
    DlInference full = {0}, tiled = {0}, no = {0};
    DlKvCache cache = {0}, no_cache = {0};
    OK(dl_checkpoint_load_weights(path, &w, fixture_config(), 6568, 4512));
    CHECK(w.ready && w.position_limit == 8 && w.param_count == 20 && w.arena.count == 1128);
    for (size_t p = 0; p < w.param_count; ++p) {
        DlMatrix v = w.params[p].value;
        CHECK(v.base == w.arena.data);
        for (size_t i = 0; i < v.rows * v.cols; ++i)
            CHECK(v.base[v.offset + i] == fixture_value(p, i));
    }
    float *old = w.arena.data;
    CHECK(dl_checkpoint_load_weights(path, &w, fixture_config(), 6568, 4512) == DL_STATE);
    CHECK(w.arena.data == old && w.ready && w.position_limit == 8);
    CHECK(dl_inference_create(&no, &w, 9, 1048576) == DL_RANGE);
    CHECK(dl_inference_create_tiled(&no, &w, 9, 2, 3, 1048576) == DL_RANGE);
    CHECK(dl_cache_create(&no_cache, &w, 9, 1048576) == DL_RANGE);
    CHECK(no.arena.data == NULL && no_cache.history.data == NULL);
    OK(dl_inference_create(&full, &w, 8, 1048576));
    OK(dl_inference_create_tiled(&tiled, &w, 8, 2, 3, 1048576));
    OK(dl_cache_create(&cache, &w, 8, 1048576));
    const size_t ids[] = {3,1,14,2,11,8,5,7};
    float a[16], b[16], d[16];
    DlMatrix av = {a,16,0,1,16,16}, bv = {b,16,0,1,16,16}, dv = {d,16,0,1,16,16};
    for (int backend = 0; backend < 2; ++backend) {
        OK(dl_cache_reset(&cache));
        for (size_t n = 1; n <= 8; ++n) {
            OK(dl_inference_forward(&full, (DlBackend)backend, ids, n, n-1, av));
            OK(dl_inference_forward(&tiled, (DlBackend)backend, ids, n, n-1, bv));
            if (n == 1) OK(dl_cache_prefill(&cache, (DlBackend)backend, ids, 1, dv));
            else OK(dl_cache_decode(&cache, (DlBackend)backend, ids[n-1], dv));
            for (size_t i = 0; i < 16; ++i) {
                CHECK(isfinite(a[i]) && isfinite(b[i]) && isfinite(d[i]));
                CHECK(fabs((double)a[i]-b[i]) <= 2e-6 + 2e-5*fabs(a[i]));
                CHECK(fabs((double)a[i]-d[i]) <= 2e-6 + 2e-5*fabs(a[i]));
            }
        }
        CHECK(dl_cache_decode(&cache, (DlBackend)backend, 1, dv) == DL_RANGE);
        CHECK(cache.length == 8);
    }
    puts("import: 1128 exact values; 512 full/tiled/cache logit comparisons; position cap passed");
    result = 1;
done:
    dl_cache_destroy(&no_cache);
    dl_cache_destroy(&cache);
    dl_inference_destroy(&no);
    dl_inference_destroy(&tiled);
    dl_inference_destroy(&full);
    dl_weights_destroy(&w);
    return result;
}
```

### Exercise early, middle, and final failures

`malformed` changes one field at a time in an otherwise valid file. The wrong-field array includes magic, version, architecture tag, dimensions, counts, scalar settings, seed, payload offset, directory names/shapes/spans, and trailer. A separate duplicate-name case tests identity rather than total byte count.

Truncation points include the first magic, header, directory boundary, first float, and trailer. One nonfinite payload is at the beginning and another is the final float, exercising cleanup after almost the entire candidate has been filled. Two budget tests put the limit exactly one byte below the valid requirement.

The 39 damaged/truncated variants and two budget failures have different expected error classes. Treating all failures as generic success/failure would miss accidental changes in the API contract.

```c title="tests/test_import.c"
static int malformed(void)
{
    int result = 0, fd = -1;
    char path[] = "/tmp/dl-import-XXXXXX";
    size_t size = 0;
    unsigned char *good = wire_fixture(&size), *bad = NULL;
    CHECK(good != NULL && size == 6568);
    bad = malloc(size + 1);
    CHECK(bad != NULL);
    fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(close(fd) == 0);
    fd = -1;
    CHECK(write_bytes(path, good, size));
    CHECK(loaded(path));
    CHECK(rejected(path, DL_RANGE, size-1, 4512));
    CHECK(rejected(path, DL_RANGE, size, 4511));
    const size_t wrong[] = {0,8,16,24,32,40,48,56,64,72,80,96,100,104,112,120,
                            128,192,200,208,216,128+96,128+96+80,6560};
    for (size_t j = 0; j < sizeof(wrong)/sizeof(wrong[0]); ++j) {
        memcpy(bad, good, size);
        bad[wrong[j]] ^= 1;
        CHECK(write_bytes(path, bad, size));
        CHECK(rejected(path, DL_SHAPE, size, 4512));
    }
    memcpy(bad, good, size);
    memcpy(bad + 224, bad + 128, 64); /* Duplicate name without moving payload. */
    CHECK(write_bytes(path, bad, size));
    CHECK(rejected(path, DL_SHAPE, size, 4512));
    const size_t truncated[] = {0,7,95,127,2047,2048,2051,6560,6567};
    for (size_t j = 0; j < sizeof(truncated)/sizeof(truncated[0]); ++j) {
        CHECK(write_bytes(path, good, truncated[j]));
        CHECK(rejected(path, DL_RANGE, size, 4512));
    }
    memcpy(bad, good, size);
    bad[size] = 123;
    CHECK(write_bytes(path, bad, size+1));
    CHECK(rejected(path, DL_RANGE, size+1, 4512));
    memcpy(bad, good, size);
    put_u(bad + 88, 0, 8);
    CHECK(write_bytes(path, bad, size));
    CHECK(rejected(path, DL_RANGE, size, 4512));
    put_u(bad + 88, UINT64_MAX, 8);
    CHECK(write_bytes(path, bad, size));
    CHECK(rejected(path, DL_RANGE, size, 4512));
    for (size_t j = 0; j < 2; ++j) {
        memcpy(bad, good, size);
        put_u(bad + (j == 0 ? 2048 : 6556), j == 0 ? 0x7fc00000 : 0x7f800000, 4);
        CHECK(write_bytes(path, bad, size));
        CHECK(rejected(path, DL_NUMERIC, size, 4512));
    }
    puts("import: 39 damaged/truncated files and two exact-budget failures rejected");
    result = 1;
done:
    if (fd >= 0) (void)close(fd);
    if (path[0] != '\0') (void)unlink(path);
    free(bad);
    free(good);
    return result;
}
```


```c title="tests/test_import.c"
int main(int argc, char **argv)
{
    if (argc == 2) return loaded(argv[1]) ? 0 : 1;
    if (argc != 1) return 2;
    return malformed() ? 0 : 1;
}
```

## A load-only program must not quietly create a trainer

Add `app/inspect_weights.c`. With a configuration and a weight file, it loads and reports ownership counts, then exits. It never constructs `DlModel`, `DlAdamW`, or a training tape. This makes it suitable for an isolated process-memory observation.

An optional integer token ID requests one C forward pass and a greedy ID. It intentionally does not read text, map UTF-8 bytes to this vocabulary, add special tokens, or decode the result. Those are Chapter 21 boundaries. The tiled workspace is one-token capacity; its memory budget is separate from the weight budget.

The command-line parser rejects negative, empty, whitespace-prefixed, overflowing, and trailing-character inputs. `strtoumax` alone would accept some of those forms, so the first-character and end-pointer checks are deliberate. Every exit destroys logits, then the borrowing workspace, then weights.

```c title="app/inspect_weights.c"
#include "dl/checkpoint.h"
#include "dl/sample.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s CONFIG WEIGHTS [TOKEN_ID]\n", argv[0]);
        return 2;
    }
    int result = 1;
    DlModelConfig config = {0};
    size_t batch = 0, time = 0;
    DlWeights weights = {0};
    DlInference workspace = {0};
    DlTensor logits = {0};
    DlStatus status = dl_model_config_load(argv[1], &config, &batch, &time);
    if (status != DL_OK) goto done;
    status = dl_checkpoint_load_weights(argv[2], &weights, config,
        UINT64_C(2) * 1024 * 1024 * 1024, (size_t)1024 * 1024 * 1024);
    if (status != DL_OK) goto done;
    printf("tensors=%zu values=%zu weight_payload_bytes=%zu position_limit=%zu\n",
           weights.param_count, weights.arena.count,
           weights.arena.count * sizeof(float), weights.position_limit);
    printf("directory_bytes=%zu gradients=0 moments=0 loader_chunk_bytes=4096\n",
           weights.param_count * sizeof(DlWeight));
    if (argc == 4) {
        if (argv[3][0] < '0' || argv[3][0] > '9') { status = DL_ARGUMENT; goto done; }
        char *end = NULL;
        errno = 0;
        uintmax_t parsed = strtoumax(argv[3], &end, 10);
        if (errno != 0 || *end != '\0' || parsed >= config.vocab) {
            status = DL_RANGE; goto done;
        }
        size_t id = (size_t)parsed;
        status = dl_inference_create_tiled(&workspace, &weights, 1, 1, 1,
                                            16 * 1024 * 1024);
        if (status != DL_OK) goto done;
        status = dl_tensor_create(&logits, 1, &config.vocab);
        if (status != DL_OK) goto done;
        DlMatrix row = {logits.data, logits.count, 0, 1, config.vocab, config.vocab};
        status = dl_inference_forward(&workspace, DL_BACKEND_BLOCKED, &id, 1, 0, row);
        if (status != DL_OK) goto done;
        size_t selected = 0;
        status = dl_greedy(logits.data, config.vocab, &selected);
        if (status != DL_OK) goto done;
        printf("input_id=%zu next_argmax_id=%zu selected_logit=%.9g\n",
               id, selected, (double)logits.data[selected]);
        puts("Finite C execution only; not a tokenizer or independent numerical comparison.");
    }
    result = 0;
done:
    if (status != DL_OK) fprintf(stderr, "inspect_weights: %s\n", dl_status_name(status));
    dl_tensor_destroy(&logits);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    return result;
}
```

## Connect the programs to the existing Makefile

In the root Makefile, replace `PROGRAMS` and the three build-mode aggregate target lines with the following four lines. Existing compile flags, core source lists, headers, recipes, and CLI test targets remain unchanged. A Make prerequisite line is not a shell command, so these four lines have no leading tab.

```makefile title="Makefile"
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim test_training test_checkpoint test_inference test_sample test_cache test_tiled test_tiled_training test_import
debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data build/debug/train build/debug/generate build/debug/inspect_weights
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data build/sanitize/train build/sanitize/generate build/sanitize/inspect_weights
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data build/release/train build/release/generate build/release/inspect_weights
```

Append these complete recipes and the asset-test target. The command lines begin with literal tabs. There is no new core C translation unit: the loader extends `src/checkpoint.c`, while the new application and test are linked against the existing core.

```makefile title="Makefile"

build/debug/inspect_weights: app/inspect_weights.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/inspect_weights: app/inspect_weights.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/inspect_weights: app/inspect_weights.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

.PHONY: test-import-assets
test-import-assets: build/release/test_import build/release/inspect_weights
	python3 tests/test_export_smollm2.py build/release/test_import build/release/inspect_weights
```

From the learner implementation root, run:

```sh
make debug sanitize release
./build/debug/test_import
./build/sanitize/test_import
./build/release/test_import
make test-import-assets
```

The Makefile defaults to `xcrun clang` on the native Mac. An equivalent C17 compiler can be selected, for example `make CC=clang debug` in a non-Apple environment. Sanitizer availability is a toolchain capability, not an exact version gate. Do not use `-ffast-math` to optimize a loader or numerical program that relies on explicit nonfinite checks.

Expected successful C test output is:

```text
import: 1128 exact values; 512 full/tiled/cache logit comparisons; position cap passed
import: 39 damaged/truncated files and two exact-budget failures rejected
```

The cross-language Python target is supplied completely in [Part 4](https://25d-228.github.io/gpt-c/chapter-20-part-4.html). Run it before attempting a real download. A missing network connection cannot prevent these format, value, and ownership checks.
