# Chapter 17 · Part 3 — Checking cache equivalence and failure recovery

[Part 1](https://25d-228.github.io/gpt-c/chapter-17.html) allocated history; [Part 2](https://25d-228.github.io/gpt-c/chapter-17-part-2.html) computed and committed it. This final part of the same **4.50-hour** chapter verifies that storage scheduling did not change the model, exercises failure recovery, and runs cached generation from an actually trained checkpoint.

A plausible continuation is a weak correctness check. A broken head mapping can still produce readable tokens from some weights. Compare logits at every prefix, verify an independent hand calculation, and inspect the state boundaries. These checks answer different questions.

## Test helpers that make visibility explicit

Create `tests/test_cache.c`. The status/check helpers print the failed expression and jump to each test's cleanup label. `near` combines absolute and relative tolerances and rejects nonfinite values. `prefix_same` compares only committed rows in every layer's key and value planes, using exact bytes: a numerical tolerance is inappropriate for memory that should not have been written at all.

```c title="tests/test_cache.c"
#include "dl/cache.h"
#include "dl/sample.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int require(int condition, const char *where)
{
    if (!condition) fprintf(stderr, "FAILED: %s\n", where);
    return condition;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "FAILED: %s: %s, expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

#define CHECK(yes) do { if (!require((yes), #yes)) goto done; } while (0)
#define STATUS(call, want) do { if (!status_is((call), (want), #call)) goto done; } while (0)
#define OK(call) STATUS((call), DL_OK)

static int near(float a, float b)
{
    return isfinite(a) && isfinite(b) &&
           fabs((double)a - (double)b) <= 2e-6 + 2e-5 * fabs((double)b);
}

static int prefix_same(const DlKvCache *s, const float *old, size_t length)
{
    DlModelConfig c = s->weights->config;
    size_t kv = c.kv_heads * (c.width / c.query_heads);
    for (size_t plane = 0; plane < 2 * c.layers; ++plane) {
        size_t offset = plane * s->capacity * kv;
        if (memcmp(s->history.data + offset, old + offset,
                   length * kv * sizeof(float)) != 0) return 0;
    }
    return 1;
}

static void poison(DlTensor *tensor)
{
    for (size_t i = 0; i < tensor->count; ++i) tensor->data[i] = NAN;
}
```

`poison` intentionally writes NaNs into test-owned storage before it should be populated. It is not production initialization. Poisoning unused history catches reading capacity instead of length; poisoning scratch between successful calls catches retaining pointers into temporary arrays instead of real history.

## Every prefill split reaches the same prefix logits

For each head-sharing configuration and matrix backend, compute independent full-prefix logits for lengths 1 through 17. Then try **every** prompt length from 1 through 17, decoding the remainder one token at a time. This exercises both many-row and one-row scratch layouts, reset, an odd capacity, and exact exhaustion.

The output matrix has an offset and padding. Those guard cells must remain unchanged. The histories are copied only inside the test to detect unintended writes; the cache implementation itself does not copy them. A second cache proves that resetting the first does not reset all users of the same weights.

```c title="tests/test_cache.c"
static int parity(size_t kv_heads, size_t layers, DlBackend backend)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlInference full = {0};
    DlKvCache cache = {0}, other = {0};
    float *old = NULL;
    enum { C = 17, V = 16, D = 16, H = 4, F = 20, STORAGE = 21 };
    DlModelConfig config = {.vocab = V, .layers = layers, .width = D,
        .query_heads = H, .kv_heads = kv_heads, .ffn_width = F,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    size_t ids[C];
    float expected[C][V], output[STORAGE];
    DlMatrix out = {output, STORAGE, 2, 1, V, V + 3};
    for (size_t i = 0; i < C; ++i) ids[i] = (3 * i + 1) % V;
    OK(dl_model_create(&model, config));
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_inference_create(&full, &weights, C, 1024 * 1024));
    OK(dl_cache_create(&cache, &weights, C, 1024 * 1024));
    OK(dl_cache_create(&other, &weights, C, 1024 * 1024));
    size_t kv = kv_heads * (D / H);
    CHECK(cache.history.count == 2 * layers * C * kv);
    CHECK(cache.arena.count == 7 * C * D + C * kv + 3 * C * F + 2 * C);
    old = malloc(cache.history.count * sizeof(float));
    CHECK(old != NULL);
    for (size_t n = 1; n <= C; ++n) {
        DlMatrix row = {expected[n - 1], V, 0, 1, V, V};
        OK(dl_inference_forward(&full, backend, ids, n, n - 1, row));
    }
    size_t comparisons = 0;
    for (size_t prompt = 1; prompt <= C; ++prompt) {
        OK(dl_cache_reset(&cache));
        poison(&cache.history); /* Unwritten capacity must never be read. */
        for (size_t n = prompt; n <= C; ++n) {
            for (size_t i = 0; i < STORAGE; ++i) output[i] = 88.0f;
            poison(&cache.arena); /* Previous temporary data is not history. */
            memcpy(old, cache.history.data, cache.history.count * sizeof(float));
            size_t committed = cache.length;
            if (n == prompt) OK(dl_cache_prefill(&cache, backend, ids, n, out));
            else OK(dl_cache_decode(&cache, backend, ids[n - 1], out));
            CHECK(cache.length == n && prefix_same(&cache, old, committed));
            for (size_t j = 0; j < V; ++j) CHECK(near(output[2 + j], expected[n - 1][j]));
            for (size_t i = 0; i < STORAGE; ++i)
                if (i < 2 || i >= 2 + V) CHECK(output[i] == 88.0f);
            ++comparisons;
        }
        float saved[STORAGE];
        memcpy(saved, output, sizeof(saved));
        memcpy(old, cache.history.data, cache.history.count * sizeof(float));
        STATUS(dl_cache_decode(&cache, backend, 0, out), DL_RANGE);
        CHECK(cache.length == C && prefix_same(&cache, old, C));
        CHECK(memcmp(saved, output, sizeof(saved)) == 0);
    }
    OK(dl_cache_prefill(&other, backend, ids, 2, out));
    OK(dl_cache_reset(&cache));
    CHECK(cache.length == 0 && other.length == 2);
    OK(dl_cache_decode(&other, backend, ids[2], out));
    for (size_t j = 0; j < V; ++j) CHECK(near(output[2 + j], expected[2][j]));
    printf("cache parity: kv=%zu layers=%zu backend=%d: %zu prefix logits, exact-capacity guards\n",
           kv_heads, layers, (int)backend, comparisons);
    result = 1;
done:
    free(old);
    dl_cache_destroy(&other);
    dl_cache_destroy(&cache);
    dl_inference_destroy(&full);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

There are $17+16+\cdots+1=153$ logit-row comparisons per configuration/backend pair. Three sharing cases and two backends make 918. They cover one-, two-, and three-block models, with $d=4$ so rotary pairing has more than one coordinate pair. The additional second-cache continuation is checked separately.

This test allows small numerical error in computed logits but permits **no** change to old committed bytes. A cache implementation can satisfy one property and violate the other; both are required.

## Check a result that does not come from the reference decoder

The following fixture implements the numerical example from Part 2. All weights are edited only after `dl_model_mark_changed`, before they are copied into immutable inference weights. The test then checks two-token logits, three-token logits, all rotated key coordinates, and all unrotated value coordinates.

```c title="tests/test_cache.c"
/* Deliberately simple weights: identity Q/O, K/V read the first two columns;
 * the second query head is zero before attention; all feed-forward weights zero. */
static int hand_row(void)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlKvCache cache = {0};
    DlModelConfig config = {.vocab = 4, .layers = 1, .width = 4,
        .query_heads = 2, .kv_heads = 1, .ffn_width = 4,
        .epsilon = 1e-5f, .init_std = 0.1f, .rope_theta = 10000.0, .seed = 1};
    const size_t ids[] = {0, 1, 2};
    float output[4];
    DlMatrix out = {output, 4, 0, 1, 4, 4};
    OK(dl_model_create(&model, config));
    OK(dl_model_mark_changed(&model));
    for (size_t p = 0; p < model.param_count; ++p)
        for (size_t i = 0; i < model.params[p].value.count; ++i)
            model.params[p].value.data[i] = model.params[p].decay ? 0.0f : 1.0f;
    const float embedding[] = {1,0,0,0, 0,1,0,0, 1,1,0,0, 0,0,0,0};
    memcpy(model.params[0].value.data, embedding, sizeof(embedding));
    for (size_t j = 0; j < 4; ++j) {
        model.params[1 + DL_P_QUERY].value.data[j * 4 + j] = 1.0f;
        model.params[1 + DL_P_OUTPUT].value.data[j * 4 + j] = 1.0f;
    }
    for (size_t j = 0; j < 2; ++j) {
        model.params[1 + DL_P_KEY].value.data[j * 4 + j] = 1.0f;
        model.params[1 + DL_P_VALUE].value.data[j * 4 + j] = 1.0f;
    }
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_cache_create(&cache, &weights, 3, 1024 * 1024));
    OK(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 2, out));
    const float expected_two[] = {0.00658205f, 1.80785865f, 1.81444070f, 0.0f};
    for (size_t j = 0; j < 4; ++j) CHECK(near(output[j], expected_two[j]));
    OK(dl_cache_decode(&cache, DL_BACKEND_REF, ids[2], out));
    const float expected_three[] = {0.96675674f, 1.50301288f, 2.46976961f, 0.0f};
    const float expected_key[] = {1.99996000f,0, -1.68290831f,1.08058300f,
                                 -1.87444251f,0.69741328f};
    const float expected_value[] = {1.99996000f,0, 0,1.99996000f, 1.41419942f,1.41419942f};
    for (size_t j = 0; j < 4; ++j) CHECK(near(output[j], expected_three[j]));
    for (size_t i = 0; i < 6; ++i) {
        CHECK(near(cache.history.data[i], expected_key[i]));
        CHECK(near(cache.history.data[6 + i], expected_value[i]));
    }
    puts("cache hand row: full logits, once-rotated keys, unrotated shared values");
    result = 1;
done:
    dl_cache_destroy(&cache);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

The last two arrays reveal different mistakes. A key mismatch with a correct value suggests a position or rotary-coordinate problem. Both mismatching suggests a projection, head mapping, or history address problem. Matching histories but wrong logits points toward attention normalization, the output path, or a later residual/normalization operation.

## Reject invalid requests before mutation

The next fixture checks the combined float budget at its exact boundary. With $L=2$, $C=3$, $D=4$, $K=2$, and $F=4$, history has 24 floats and scratch has 132. A 155-float-plus-three-byte budget cannot fit 156 floats. This is deliberately not a test of whether the operating system can satisfy a large allocation.

It also checks unready weights, empty decode, nonempty prefill, invalid IDs/backend, malformed output views, aliases into all three float owners, and reset's non-wiping behavior. Failed-construction cleanup is implemented in the constructor, but this test does not claim exhaustive allocator-failure injection at every `malloc` site.

```c title="tests/test_cache.c"
static int failures(void)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0}, unready = {0};
    DlKvCache cache = {0}, rejected = {0};
    DlModelConfig config = {.vocab = 4, .layers = 2, .width = 4,
        .query_heads = 2, .kv_heads = 1, .ffn_width = 4,
        .epsilon = 1e-5f, .init_std = 0.1f, .rope_theta = 10000.0, .seed = 1};
    size_t ids[] = {0, 2}, invalid[] = {0, 4};
    float output[4] = {77,77,77,77}, before[4], old[24];
    DlMatrix out = {output, 4, 0, 1, 4, 4};
    OK(dl_model_create(&model, config));
    OK(dl_weights_create(&unready, config, 1024 * 1024));
    STATUS(dl_cache_create(&rejected, &unready, 3, 1024 * 1024), DL_STATE);
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    STATUS(dl_cache_create(&rejected, &weights, 0, 1024 * 1024), DL_ARGUMENT);
    /* history=24, scratch=3*(7*4+2+3*4+2)=132 floats. */
    STATUS(dl_cache_create(&rejected, &weights, 3, 156 * sizeof(float) - 1), DL_RANGE);
    CHECK(rejected.weights == NULL && rejected.history.data == NULL && rejected.arena.data == NULL);
    if (SIZE_MAX > UINT32_MAX)
        STATUS(dl_cache_create(&rejected, &weights, (size_t)UINT32_MAX + 2, SIZE_MAX), DL_RANGE);
    OK(dl_cache_create(&cache, &weights, 3, 156 * sizeof(float)));
    STATUS(dl_cache_create(&cache, &weights, 3, 1024 * 1024), DL_STATE);
    CHECK(cache.history.count == 24 && cache.arena.count == 132);
    for (size_t i = 0; i < 24; ++i) cache.history.data[i] = 99.0f;
    memcpy(old, cache.history.data, sizeof(old));
    memcpy(before, output, sizeof(before));
    STATUS(dl_cache_decode(&cache, DL_BACKEND_REF, 0, out), DL_STATE);
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 0, out), DL_RANGE);
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 4, out), DL_RANGE);
    STATUS(dl_cache_prefill(&cache, (DlBackend)99, ids, 2, out), DL_ARGUMENT);
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, invalid, 2, out), DL_RANGE);
    DlMatrix bad = out;
    bad.cols = 3;
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 2, bad), DL_SHAPE);
    bad = out;
    bad.offset = 2;
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 2, bad), DL_RANGE);
    DlTensor *owners[] = {&cache.history, &cache.arena, &weights.arena};
    for (size_t i = 0; i < 3; ++i) {
        OK(dl_matrix_view(owners[i], 0, 1, 4, 4, &bad));
        STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 2, bad), DL_ALIAS);
    }
    CHECK(cache.length == 0 && memcmp(old, cache.history.data, sizeof(old)) == 0);
    CHECK(memcmp(before, output, sizeof(before)) == 0);
    OK(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 2, out));
    memcpy(old, cache.history.data, sizeof(old));
    memcpy(before, output, sizeof(before));
    STATUS(dl_cache_prefill(&cache, DL_BACKEND_REF, ids, 1, out), DL_STATE);
    STATUS(dl_cache_decode(&cache, DL_BACKEND_REF, 4, out), DL_RANGE);
    CHECK(cache.length == 2 && memcmp(old, cache.history.data, sizeof(old)) == 0);
    CHECK(memcmp(before, output, sizeof(before)) == 0);
    OK(dl_cache_reset(&cache));
    CHECK(memcmp(old, cache.history.data, sizeof(old)) == 0); /* Not a wipe. */
    OK(dl_cache_prefill(&cache, DL_BACKEND_REF, ids + 1, 1, out));
    result = 1;
    puts("cache preflight: budgets, lifecycle, ranges, output aliases; reset forgets without wiping");
done:
    dl_cache_destroy(&rejected);
    dl_cache_destroy(&cache);
    dl_cache_destroy(&cache);
    dl_weights_destroy(&unready);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

The oversized prefill passes a pointer to only two IDs and requests four. Capacity rejection must occur before reading the IDs; sanitizers would expose a misplaced loop. Conversely, the invalid second ID must be found before any prompt projections begin. Checking only IDs as they are processed would violate the preflight guarantee for a multi-row prompt.

## Fail after every layer has written its suffix

An early invalid-ID rejection cannot prove that the commit boundary is correct. Construct a finite model where token 1 overflows **final** RMSNorm, after both blocks have already written keys and values. Set the first final scale to `FLT_MAX`. Token 1's normalized first coordinate is approximately two, so its product cannot be represented as a float. Safe tokens have a zero first coordinate and do not trigger the overflow.

All fixture weights are set before copying. The test does not violate inference immutability to induce the failure. After the rejected token, append a safe token **without resetting** and compare with a fresh full-prefix execution. Then test a failed full-prompt prefill and a successful replacement prompt.

```c title="tests/test_cache.c"
static int late_failure(DlBackend backend)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlKvCache cache = {0};
    DlInference full = {0};
    DlModelConfig config = {.vocab = 4, .layers = 2, .width = 4,
        .query_heads = 2, .kv_heads = 1, .ffn_width = 4,
        .epsilon = 1e-5f, .init_std = 0.1f, .rope_theta = 10000.0, .seed = 1};
    float output[4], expected[4], old[24];
    DlMatrix out = {output, 4, 0, 1, 4, 4}, ref = {expected, 4, 0, 1, 4, 4};
    size_t good[] = {0, 2}, bad[] = {0, 1};
    OK(dl_model_create(&model, config));
    OK(dl_model_mark_changed(&model));
    for (size_t p = 0; p < model.param_count; ++p)
        for (size_t i = 0; i < model.params[p].value.count; ++i)
            model.params[p].value.data[i] = model.params[p].decay ? 0.0f : 1.0f;
    const float embedding[] = {0,1,0,0, 2,0,0,0, 0,0.5f,0,0, 0,0,0,0};
    memcpy(model.params[0].value.data, embedding, sizeof(embedding));
    model.params[model.param_count - 1].value.data[0] = FLT_MAX;
    for (size_t layer = 0; layer < 2; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        for (size_t j = 0; j < 2; ++j) {
            model.params[base + DL_P_KEY].value.data[j * 4 + j] = (float)(layer + 1);
            model.params[base + DL_P_VALUE].value.data[j * 4 + j] = (float)(layer + 1);
        }
    }
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_cache_create(&cache, &weights, 3, 1024 * 1024));
    OK(dl_inference_create(&full, &weights, 3, 1024 * 1024));
    for (size_t i = 0; i < 24; ++i) cache.history.data[i] = 99.0f;
    OK(dl_cache_prefill(&cache, backend, good, 1, out));
    memcpy(old, cache.history.data, sizeof(old));
    STATUS(dl_cache_decode(&cache, backend, 1, out), DL_NUMERIC);
    CHECK(cache.length == 1 && prefix_same(&cache, old, 1));
    /* A final-norm overflow occurs AFTER all layers wrote new K and V. */
    for (size_t plane = 0; plane < 4; ++plane) {
        size_t offset = plane * 3 * 2 + 2;
        CHECK(cache.history.data[offset] != 99.0f);
    }
    OK(dl_cache_decode(&cache, backend, 2, out)); /* No reset or restoration. */
    OK(dl_inference_forward(&full, backend, good, 2, 1, ref));
    CHECK(cache.length == 2 && prefix_same(&cache, old, 1));
    for (size_t j = 0; j < 4; ++j) CHECK(near(output[j], expected[j]));
    OK(dl_cache_reset(&cache));
    STATUS(dl_cache_prefill(&cache, backend, bad, 2, out), DL_NUMERIC);
    CHECK(cache.length == 0);
    OK(dl_cache_prefill(&cache, backend, good, 2, out));
    for (size_t j = 0; j < 4; ++j) CHECK(near(output[j], expected[j]));
    printf("cache late failure: backend=%d, both-layer suffix written, old prefix preserved, retry agrees\n",
           (int)backend);
    result = 1;
done:
    dl_cache_destroy(&cache);
    dl_inference_destroy(&full);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

Each layer's tentative suffix is known to have changed, yet every committed prefix coordinate is bitwise unchanged and length is still one. That combination demonstrates the intended recovery mechanism: uncommitted writes are allowed; publication of an invalid token is not. No array-wide rollback is needed.

## Keep selection and cache advancement separate

The next fixture runs eight greedy or sampled choices against full-prefix logits, starting from three prompt IDs. Equal seeds must consume the same number of draws, and greedy must consume none. After producing eleven transcript IDs, only ten inputs have been consumed. This is a test of the generation schedule, not only attention arithmetic.

```c title="tests/test_cache.c"
static int selection(DlBackend backend, int sample)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlInference full = {0};
    DlKvCache cache = {0};
    DlModelConfig config = {.vocab = 16, .layers = 2, .width = 16,
        .query_heads = 4, .kv_heads = 2, .ffn_width = 20,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    float a[16], b[16];
    double probabilities[16];
    size_t ids[11] = {3, 7, 1}, length = 3;
    DlMatrix ma = {a, 16, 0, 1, 16, 16}, mb = {b, 16, 0, 1, 16, 16};
    DlRng ra = {42}, rb = {42};
    OK(dl_model_create(&model, config));
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_inference_create(&full, &weights, 11, 1024 * 1024));
    OK(dl_cache_create(&cache, &weights, 11, 1024 * 1024));
    OK(dl_cache_prefill(&cache, backend, ids, length, mb));
    for (size_t step = 0; step < 8; ++step) {
        OK(dl_inference_forward(&full, backend, ids, length, length - 1, ma));
        if (step != 0) OK(dl_cache_decode(&cache, backend, ids[length - 1], mb));
        for (size_t j = 0; j < 16; ++j) CHECK(near(a[j], b[j]));
        size_t na = 0, nb = 0;
        if (sample) {
            OK(dl_sample_next(a, 16, 0.8, &ra, probabilities, &na));
            OK(dl_sample_next(b, 16, 0.8, &rb, probabilities, &nb));
        } else {
            OK(dl_greedy(a, 16, &na));
            OK(dl_greedy(b, 16, &nb));
        }
        CHECK(na == nb && ra.state == rb.state);
        ids[length++] = na;
    }
    CHECK(length == 11 && cache.length == 10); /* Last selected token unconsumed. */
    if (!sample) CHECK(ra.state == 42);
    result = 1;
done:
    dl_cache_destroy(&cache);
    dl_inference_destroy(&full);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

The driver runs each independent group even if a previous group failed, so diagnostics are not hidden by one short-circuiting `&&`. Bitwise `&` and `&=` combine integer success flags here; they are not substitutes for logical operators in preflight conditions.

```c title="tests/test_cache.c"
int main(void)
{
    int passed = hand_row() & failures();
    const size_t heads[] = {1, 2, 4};
    for (int b = 0; b < 2; ++b) {
        DlBackend backend = b == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED;
        for (size_t i = 0; i < 3; ++i) passed &= parity(heads[i], i + 1, backend);
        passed &= late_failure(backend);
        passed &= selection(backend, 0);
        passed &= selection(backend, 1);
    }
    if (!passed) return EXIT_FAILURE;
    puts("all cache checks passed: 918 prefix logits and 32 successive token choices");
    return EXIT_SUCCESS;
}
```

## Exercise the executable, not a Python model

Replace `tests/test_generate_cli.py` with the following blocks. Python launches the C trainer and generator, manipulates binary fixtures, and compares results. It does not compute model outputs. Its optional third positional argument selects which inference strategy receives the full boundary/I/O suite; each run also compares with the other strategy and the old command-line form.

The initial helper makes a valid-format all-zero-weight fixture. Its purpose is predictable generated NUL bytes and lowest-ID ties, not a substitute for the nonzero-model logit checks above.

```python title="tests/test_generate_cli.py"
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
```

`run` assembles the actual command. The special `legacy` choice omits the new argument and checks backward-compatible parsing. It is a test-helper value, not an accepted C application mode. The C checkpoint is trained locally, then its original corpus directory is removed before generation.

```python title="tests/test_generate_cli.py"
def main() -> None:
    train = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    generate = Path(sys.argv[2] if len(sys.argv) > 2 else "build/release/generate").resolve()
    engine = sys.argv[3] if len(sys.argv) > 3 else "full"
    if engine not in ("full", "cache"):
        raise ValueError("engine must be full or cache")
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
```

The next block compares fresh processes, both matrix backends, both inference strategies, the legacy command, and a shorter generation request. It also checks the consumed count rather than inferring it from output length.

```python title="tests/test_generate_cli.py"
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
            raise AssertionError("cached/full/legacy output disagreement")
        if f"inference={engine} consumed=15 " not in log_a.stdout:
            raise AssertionError("selected final token was incorrectly counted as consumed")
        if len(a.read_bytes()) != 16 or not a.read_bytes().startswith(b"A"):
            raise AssertionError("wrong prompt/count")
        prefix = work / "prefix.bin"
        run(prefix, new="7")
        if prefix.read_bytes() != a.read_bytes()[:8]:
            raise AssertionError("longer request changed earlier sampling draws")
        print(f"generation cli: {engine}, full/cache/legacy equality, consumed count, seed/backend replay; no datasets")
```

Greedy output must not depend on the RNG seed. Existing output and checkpoint paths must not be truncated. Invalid arguments must not create output files. A zero-new-token request preserves prompt bytes and RNG while reporting zero consumed inputs.

```python title="tests/test_generate_cli.py"
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
```

Finally, check binary prompts containing NUL and high-bit bytes, a truncated checkpoint, and a real operating-system write failure where supported. The file-size limit applies to the child process; it does not modify the model's math or simulate a successful write. Only a newly created incomplete output may be removed.

```python title="tests/test_generate_cli.py"
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
```

## Build the new module in every mode

Replace the root Makefile with this complete version. `src/cache.c` belongs to `CORE`, `cache.h` belongs to the header dependencies, and `test_cache` belongs to all three test builds. The generation CLI target runs the complete process-level suite for both strategies.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/rng.c src/model.c src/data.c src/optim.c src/train.c src/checkpoint.c src/inference.c src/sample.c src/cache.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/rng.h include/dl/model.h include/dl/data.h include/dl/optim.h include/dl/train.h include/dl/checkpoint.h include/dl/inference.h include/dl/sample.h include/dl/cache.h tests/tiny.h tests/training_fixture.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim test_training test_checkpoint test_inference test_sample test_cache

.PHONY: all debug sanitize release test test-assets test-training-cli test-generation-cli clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data build/debug/train build/debug/generate
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data build/sanitize/train build/sanitize/generate
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data build/release/train build/release/generate

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

build/debug/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-training-cli: build/release/train
	python3 tests/test_train_cli.py

build/debug/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-generation-cli: build/release/train build/release/generate
	python3 tests/test_generate_cli.py build/release/train build/release/generate full
	python3 tests/test_generate_cli.py build/release/train build/release/generate cache
```

From your Chapter 17 implementation root, run:

```sh
make test
make test-assets test-training-cli test-generation-cli
ASAN_OPTIONS=detect_leaks=1 build/sanitize/test_cache
```

The default compiler command targets native macOS through `xcrun clang`. On a system where Clang is invoked directly, `make CC=clang test` is the equivalent override, not a requirement to change the Mac's compiler. Leak detection support varies by sanitizer/runtime; the ordinary AddressSanitizer/UndefinedBehaviorSanitizer test remains useful when leak detection is unavailable.

To run the complete supplied source snapshot from an extracted course package without mixing build outputs into the book, copy it to a disposable work directory first:

```sh
mkdir -p /tmp/decoder-chapter17
cp -R book/media/chapter-17/reference/. /tmp/decoder-chapter17/
cd /tmp/decoder-chapter17
make test test-generation-cli
```

The new C test's expected final line is:

```text
all cache checks passed: 918 prefix logits and 32 successive token choices
```

In the actual authoring run, Clang 17 on x86_64 Linux passed the cumulative debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized test builds without compiler warnings. The cache test passed all three modes; a separate leak-enabled run reported no leaks. The generation CLI suite passed for cached execution in all three modes and for both strategies in optimized mode. These are executed Linux results, not claims of native M4 execution.

## Generate the same continuation by two storage schedules

Reuse a matching pilot checkpoint from Chapter 16, or reproduce it below. Choose fresh output paths: neither training nor generation silently overwrites existing files. The original-text generator is included in the companion, so this example requires no network dataset.

```sh
make release
mkdir -p runs/chapter17
python3 scripts/make_training_fixture.py runs/chapter17/text
build/release/train new configs/byte-pilot.conf \
  runs/chapter17/text/train.bin runs/chapter17/text/valid.bin \
  runs/chapter17/pilot.bin 160 160 2 random blocked
printf 'A red fox ' > runs/chapter17/prompt.bin
for engine in full cache; do
  for mode in greedy sample; do
    build/release/generate configs/byte-pilot.conf runs/chapter17/pilot.bin \
      runs/chapter17/prompt.bin "runs/chapter17/$engine-$mode.bin" \
      64 40 "$mode" 0.8 42 blocked "$engine"
  done
done
cmp runs/chapter17/full-greedy.bin runs/chapter17/cache-greedy.bin
cmp runs/chapter17/full-sample.bin runs/chapter17/cache-sample.bin
```

These commands were executed in the Linux run. The 29,856-parameter pilot completed 160 updates, ending with held-out prefix loss `2.270204153` over 128 scored targets and 264 unscored target positions. Both `cmp` calls succeeded. The actual cached greedy report was:

```text
prompt=10 new=40 total=50 capacity=64 config_time=16
payload_bytes weights=119424 workspace=123392 logits=1024 probabilities=2048
inference=cache consumed=49 kv_payload_bytes=16384
sampling_rng=42 mode=greedy backend=blocked
bytes="A red fox the athe athe a at athed a atooa atoathe"
```

The actual sampled continuation at temperature 0.8 used RNG state `13306735003898700658` after forty draws:

```text
bytes="A red fox te cMoat pam loa iges ooad trotoored at "
```

Both strategies produced identical bytes for these runs. That is a local observation, not a cross-platform guarantee. Tiny logit differences can cross an argmax tie or a categorical interval boundary; once different tokens are selected, later computations receive different prefixes. Debug the first mismatching logit row before interpreting divergent text. The pilot's output is still repetitive and malformed, and generation exceeds its 16-token training windows. Caching makes neither the weights nor their learned context behavior better.

## Count storage and repeated work without inventing a speedup

For this pilot, $L=2$, $C=64$, $D=32$, $K=16$, and $F=80$. The persistent cache contains

$$
2LCK=2\cdot2\cdot64\cdot16=4096
$$

floats, or 16,384 bytes when a float occupies four bytes. The retained scratch contains

$$
7CD+CK+3CF+2C=30{,}848
$$

floats, or 123,392 bytes. Together they are 139,776 float-payload bytes. Chapter 16's full-prefix workspace at the same capacity contains 262,144 bytes. Weights and caller output/sampling arrays are unchanged. The cache owns another 64 positions and four frequency doubles. None of these counts includes allocator overhead, stack, loaded code, or process memory.

The reduction in temporary storage is partly a **different attention scratch schedule**, not an intrinsic promise that adding a cache always lowers memory. A full-prefix implementation could also reuse one score row. Conversely, our cache retains enough activation scratch for a full-capacity prompt even while decoding one row. Do not compare only the two score arrays and omit the new per-layer history.

For the ten-token prompt and forty selected outputs, full-prefix execution processes $10+11+\cdots+49=1180$ token rows per block through its row-wise projections and feed-forward path. Cached execution processes $10+39=49$. It still scans the visible K/V prefix for each new query, and its prompt attention still has quadratic pair count. At this workload the number of allowed query/key pairs per head is 20,660 under repeated full-prefix execution versus 1,225 with prefill plus decode. These are counts derived from the implemented loops, not timing ratios.

The API allocates nothing during execution, but checks and weight reads still have costs. A one-row matrix product may use a blocked kernel less efficiently than a large prompt matrix. No cache-throughput, process-memory, or M4 speed measurement is asserted here. A later measurement must separate prompt prefill, steady one-token decode, weight loading, and sampling/I/O, while reporting context length and the same output workload.

## Four mistakes worth making once

Make one isolated edit in a disposable copy, predict the failure, run `test_cache`, and restore the correct code. All four variants below were compiled and executed under sanitizers in the Linux checks; each failed explicit numerical or state assertions without a sanitizer memory report.

| Deliberate change | What fails | What the failure teaches |
|---|---|---|
| Set `positions[i] = i` rather than `start+i` | Three-token hand logits and cached/full-prefix parity | A one-row buffer still has a nonzero absolute position. |
| Map shared heads with `h % kv_heads` | GQA logit comparisons and token-choice checks | In-bounds indexing can still implement the wrong head partition. |
| Use `allowed = key.rows` for every new query | Multi-block prefill parity | Stored future prompt rows are not causally visible; a last-row-only one-block test can miss the error. |
| Advance `length` before the block loop | Late-failure committed-length assertion | Successful normal paths do not prove safe failure publication. |

For the head-mapping and allowed-length edits, remove or explicitly mark any now-unused local variable when compiling with warnings. The normal implementation keeps them because they express the correct indexing rules.

As a useful variation, change the **test fixture's** capacity from 17 to 19 and derive the new expected comparison count before running it: $6\cdot19\cdot20/2=1140$. Update the printed count as well. No cache kernel needs a power-of-two capacity or a full matrix tile. This is an exercise prediction, not an additional executed result in the reported checks.

## What the chapter establishes

The decoder now has a prompt-prefill path and a stateful one-token path, both implemented in C. Old K/V rows remain stable, rotary positions remain absolute, shared heads remain shared, capacity is checked before mutation, and failed arithmetic does not advance the public prefix. The full-prefix path remains available for diagnosis, and the generator exposes both strategies without changing its selection rule.

This is single-sequence inference with immutable weights. It does not add batch scheduling, padding masks, prefix sharing between requests, sliding-window eviction, cache serialization, training-through-cache, pretrained compatibility, or online tiled normalization. Those are not hidden dependencies of the runnable program. **Chapter 18 — Online normalization and tiled attention forward** next derives the bounded-tile computation used to avoid retaining an entire row of scores and to prepare for a nonquadratic training tape.
