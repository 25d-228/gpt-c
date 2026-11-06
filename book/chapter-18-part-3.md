# Chapter 18 · Part 3 — Checking tiled outputs, storage, and generation

A lower allocation count is not sufficient evidence that attention is correct. This part checks the online recurrence against hand arithmetic, an independent two-pass implementation, the materialized sublayer, and the complete decoder. It also verifies that changing tile boundaries changes storage scheduling rather than causal visibility or head mapping.

This is the final part of the **5.00-hour Chapter 18**. The implementation is in [Part 1](https://25d-228.github.io/gpt-c/chapter-18.html) and inference integration in [Part 2](https://25d-228.github.io/gpt-c/chapter-18-part-2.html). The full [companion source state](https://25d-228.github.io/gpt-c/media/chapter-18/reference/Makefile) includes every prerequisite file and the following complete tests.

## Start with assertions whose answers do not come from the new kernel

Create `tests/test_tiled.c`. `CHECK`, `STATUS`, and `OK` preserve explicit failures under optimized builds; they do not depend on `assert` remaining enabled. Test-owned objects begin empty and share one cleanup label. The small `matrix` helper is restricted to live fixture arrays with known `rows*cols` storage; it is not a replacement for checked allocation in the application.

```c title="tests/test_tiled.c"
#include "dl/attention_tiled.h"
#include "dl/inference.h"
#include "dl/sample.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int require(int yes, const char *where)
{
    if (!yes) fprintf(stderr, "FAILED: %s\n", where);
    return yes;
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

static int near(double got, double want, double tol)
{
    return isfinite(got) && isfinite(want) && fabs(got - want) <= tol * (1.0 + fabs(want));
}

static DlMatrix matrix(float *p, size_t rows, size_t cols)
{
    return (DlMatrix){p, rows * cols, 0, rows, cols, cols};
}

static float *cell(DlMatrix m, size_t r, size_t j)
{
    return &m.base[m.offset + r * m.ld + j];
}
```

The comparison checks finiteness before subtracting and uses `tol * (1 + abs(want))`, combining an absolute floor with a relative allowance. The target value is not first rounded to float. That matters when checking the final float result against a double reference.

### The three-key calculation crosses a rescaling boundary

The first fixture uses the vectors derived in Part 1. All three causal rows are checked, not just the final average. The first row must equal the first value exactly up to the stated tolerance, because it has only one allowed key. The middle row has weights $1/3$ and $2/3$. The last row has weights $1/7,2/7,4/7$.

```c title="tests/test_tiled.c"
static int hand_row(void)
{
    int result = 0;
    DlTiledWorkspace s = {0};
    float q[] = {1, 0, 1, 0, 1, 0};
    float k[] = {0, 0, 0, 0, 0, 0};
    float v[] = {1, 0, 0, 2, 3, 1};
    float output[6];
    k[2] = (float)(sqrt(2.0) * log(2.0));
    k[4] = (float)(sqrt(2.0) * log(4.0));
    DlAttentionConfig c = {1, 3, 1, 1, 2};
    DlConstMatrix qm = dl_matrix_read(matrix(q, 3, 2));
    DlConstMatrix km = dl_matrix_read(matrix(k, 3, 2));
    DlConstMatrix vm = dl_matrix_read(matrix(v, 3, 2));
    const size_t tiles[][2] = {{1, 1}, {2, 2}, {16, 32}};
    for (size_t t = 0; t < 3; ++t) {
        OK(dl_tiled_workspace_create(&s, 3, 2, tiles[t][0], tiles[t][1], 65536));
        OK(dl_attention_tiled_core(c, qm, km, vm, &s, matrix(output, 3, 2)));
        const double want[] = {1, 0, 1.0 / 3, 4.0 / 3, 13.0 / 7, 8.0 / 7};
        for (size_t i = 0; i < 6; ++i) CHECK(near(output[i], want[i], 2e-7));
        CHECK(near(s.data[2], log(4.0), 2e-7));
        CHECK(near(s.data[5], 1.75, 2e-7));
        CHECK(near(s.data[8], log(7.0), 2e-7));
        if (t == 0) printf("tiled hand: context=[%.6f, %.6f] max=%.6f sum=%.6f lse=%.6f\n",
                            output[4], output[5], s.data[2], s.data[5], s.data[8]);
        dl_tiled_workspace_destroy(&s);
    }
    result = 1;
done:
    dl_tiled_workspace_destroy(&s);
    return result;
}
```

The one-key tiles force the maximum to rise twice, making a lost rescaling visible. Two-key tiles produce the same result with a different intermediate grouping. A 16-by-32 tile is larger than the fixture in both dimensions and checks that only active rows and columns participate. The saved maximum, scaled sum, and log-sum-exp are checked independently of the returned context.

## An independent two-pass reference for the core

The next helper deliberately uses a different algorithm: form a complete score row in a **test-only** fixed array, find its maximum, sum its shifted exponentials, and then calculate each output coordinate. Its score is rounded to float like both production paths, but its probabilities remain double like the tiled path. This isolates tiling errors from the reference implementation's extra float probability rounding.

The oracle enumerates a key/value head and then its consuming query heads. That is a different traversal from the production loop's query-head-to-key-head division. Independent traversal is not a proof by itself, but it makes a mistaken modulo mapping less likely to be repeated blindly.

```c title="tests/test_tiled.c"
/* Independent two-pass oracle. TEST ONLY: complete score row of at most 65.
 * It deliberately reproduces score rounding, but not float probability rounding. */
static int oracle(DlAttentionConfig c, DlMatrix q, DlMatrix k, DlMatrix v,
                   DlMatrix output, const DlTiledWorkspace *s)
{
    double scores[65];
    if (c.time > 65) return 0;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t kh = 0; kh < c.kv_heads; ++kh) {
            for (size_t local = 0; local < c.query_heads / c.kv_heads; ++local) {
                size_t h = kh * (c.query_heads / c.kv_heads) + local;
                for (size_t t = 0; t < c.time; ++t) {
                    double maximum = -INFINITY, denominator = 0.0;
                    for (size_t key = 0; key <= t; ++key) {
                        double dot = 0.0;
                        for (size_t j = 0; j < c.head_dim; ++j)
                            dot += (double)*cell(q, b * c.time + t, h * c.head_dim + j)
                                 * (double)*cell(k, b * c.time + key, kh * c.head_dim + j);
                        scores[key] = (double)(float)(dot * (1.0 / sqrt((double)c.head_dim)));
                        if (scores[key] > maximum) maximum = scores[key];
                    }
                    for (size_t key = 0; key <= t; ++key)
                        denominator += exp(scores[key] - maximum);
                    size_t sr = (b * c.query_heads + h) * c.time + t;
                    if (!near(s->data[sr], maximum, 2e-13) ||
                        !near(s->data[s->row_capacity + sr], denominator, 2e-13) ||
                        !near(s->data[2 * s->row_capacity + sr], maximum + log(denominator), 2e-13))
                        return 0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        double want = 0.0;
                        for (size_t key = 0; key <= t; ++key)
                            want += exp(scores[key] - maximum) / denominator
                                  * (double)*cell(v, b * c.time + key, kh * c.head_dim + j);
                        if (!near(*cell(output, b * c.time + t, h * c.head_dim + j), want, 3e-7))
                            return 0;
                    }
                }
            }
        }
    }
    return 1;
}
```

The 65-score array belongs to this bounded fixture only. Production code never calls this function. It would defeat the memory goal to quietly move a sequence-sized array from the heap to an unchecked stack variable and call that tiled attention.

### Sweep tails, padding, sharing, and batch isolation

`core_case` constructs two batch items, four query heads, and a four-coordinate head dimension. The main function later varies the sequence length, key/value head count, and both tile dimensions. Logical input cells receive deterministic finite values; padding receives NaNs. The output's nonlogical storage receives a finite sentinel.

```c title="tests/test_tiled.c"
static int core_case(size_t time, size_t kv_heads, size_t qr, size_t kc)
{
    int result = 0;
    DlAttentionConfig c = {2, time, 4, kv_heads, 4};
    DlTensor owners[4] = {{0}};
    DlMatrix m[4] = {{0}};
    DlTiledWorkspace s = {0};
    size_t widths[] = {16, kv_heads * 4, kv_heads * 4, 16};
    size_t rows = c.batch * time;
    for (size_t a = 0; a < 4; ++a) {
        size_t count = 1 + rows * (widths[a] + 3) + 2;
        OK(dl_tensor_create(&owners[a], 1, &count));
        for (size_t i = 0; i < count; ++i) owners[a].data[i] = a == 3 ? 77.0f : NAN;
        OK(dl_matrix_view(&owners[a], 1, rows, widths[a], widths[a] + 3, &m[a]));
        if (a < 3)
            for (size_t r = 0; r < rows; ++r)
                for (size_t j = 0; j < widths[a]; ++j)
                    *cell(m[a], r, j) = (float)((int)((r * 13 + j * 7 + a * 3) % 29) - 14) * 0.13f;
    }
    size_t stat_rows = rows * c.query_heads;
    OK(dl_tiled_workspace_create(&s, stat_rows + 5, 4, qr, kc, 1024 * 1024));
    CHECK(s.count == 3 * (stat_rows + 5) + qr * kc + qr * 4 + 2 * qr);
    for (size_t i = 0; i < s.count; ++i) s.data[i] = NAN;
    OK(dl_attention_tiled_core(c, dl_matrix_read(m[0]), dl_matrix_read(m[1]),
                               dl_matrix_read(m[2]), &s, m[3]));
    CHECK(oracle(c, m[0], m[1], m[2], m[3], &s));
    for (size_t plane = 0; plane < 3; ++plane)
        for (size_t i = stat_rows; i < s.row_capacity; ++i)
            CHECK(isnan(s.data[plane * s.row_capacity + i]));
    for (size_t i = 0; i < owners[3].count; ++i) {
        int logical = i >= 1 && (i - 1) / m[3].ld < rows && (i - 1) % m[3].ld < 16;
        if (!logical) CHECK(owners[3].data[i] == 77.0f);
    }
    /* Changing only a late value cannot change earlier causal rows or batch 1. */
    float before[16], other[16];
    for (size_t j = 0; j < 16; ++j) {
        before[j] = *cell(m[3], 0, j);
        other[j] = *cell(m[3], time, j);
    }
    *cell(m[2], time - 1, 0) += 19.0f;
    OK(dl_attention_tiled_core(c, dl_matrix_read(m[0]), dl_matrix_read(m[1]),
                               dl_matrix_read(m[2]), &s, m[3]));
    for (size_t j = 0; j < 16; ++j) {
        if (time > 1) CHECK(*cell(m[3], 0, j) == before[j]);
        CHECK(*cell(m[3], time, j) == other[j]);
    }
    result = 1;
done:
    dl_tiled_workspace_destroy(&s);
    for (size_t a = 0; a < 4; ++a) dl_tensor_destroy(&owners[a]);
    return result;
}
```

Each row has three physical padding elements, plus allocation guards. A kernel that treats the row stride as its logical width can read NaNs or produce wrong values without going outside the allocation. The oracle uses the actual views. Output guard checks prove that only the logical cells were stored.

The workspace has five extra row-capacity cells in each statistic plane, poisoned with NaNs. Those cells must remain untouched. This checks the distinction between active $BH_qT$ rows and reserved row capacity; the offsets between planes must still use the latter.

After the numerical comparison, the test changes only a late value in batch item zero. Earlier causal outputs in that item must remain unchanged, and the first output of batch item one must remain unchanged. Single-token sequences omit the first assertion because their only row is also their last. This is a direct dependency test, not merely another tolerance comparison.

## Numerical limits and the failure boundary

A caller needs to know whether a rejected computation preserves its old outputs and whether a fresh retry can recover. The following fixture checks checked-size overflow, an insufficient budget, illegal dimensions, repeated construction, malformed output geometry, output/input overlap, and a nonfinite input before any arithmetic begins.

It then creates a different failure: query zero succeeds, while query one's allowed dot product exceeds float score range. Under one-row tiles, the first row has already been stored. The test observes that partial output, restores valid inputs, and requires a complete retry to match the independent reference.

```c title="tests/test_tiled.c"
static int limits_and_failures(void)
{
    int result = 0;
    DlTiledWorkspace s = {0}, rejected = {0};
    float q[] = {1, 0, 1, 0, 1, 0}, k[] = {0, 0, 1, 0, 2, 0};
    float v[] = {1, 2, 3, 4, 5, 6}, output[6];
    DlAttentionConfig c = {1, 3, 1, 1, 2};
    DlConstMatrix qm = dl_matrix_read(matrix(q, 3, 2));
    DlConstMatrix km = dl_matrix_read(matrix(k, 3, 2));
    DlConstMatrix vm = dl_matrix_read(matrix(v, 3, 2));
    DlMatrix out = matrix(output, 3, 2);
    STATUS(dl_tiled_workspace_create(&rejected, SIZE_MAX, 2, 1, 1, SIZE_MAX), DL_OVERFLOW);
    STATUS(dl_tiled_workspace_create(&rejected, 3, 3, 1, 1, 65536), DL_SHAPE);
    STATUS(dl_tiled_workspace_create(&rejected, 3, 2, 0, 1, 65536), DL_SHAPE);
    STATUS(dl_tiled_workspace_create(&rejected, 3, 2, 1, 1, 8), DL_RANGE);
    CHECK(rejected.data == NULL && rejected.count == 0);
    OK(dl_tiled_workspace_create(&s, 3, 2, 1, 1, 14 * sizeof(double)));
    CHECK(s.count == 14);
    STATUS(dl_tiled_workspace_create(&s, 3, 2, 1, 1, 65536), DL_STATE);
    for (size_t i = 0; i < 6; ++i) output[i] = 77.0f;
    for (size_t i = 0; i < s.count; ++i) s.data[i] = 88.0;
    DlAttentionConfig bad = c;
    bad.query_heads = 3; bad.kv_heads = 2;
    STATUS(dl_attention_tiled_core(bad, qm, km, vm, &s, out), DL_SHAPE);
    DlMatrix wrong = out; wrong.cols = 1;
    STATUS(dl_attention_tiled_core(c, qm, km, vm, &s, wrong), DL_SHAPE);
    STATUS(dl_attention_tiled_core(c, qm, km, vm, &s, matrix(q, 3, 2)), DL_ALIAS);
    k[4] = NAN;
    STATUS(dl_attention_tiled_core(c, qm, km, vm, &s, out), DL_NUMERIC);
    k[4] = 2;
    for (size_t i = 0; i < 6; ++i) CHECK(output[i] == 77.0f);
    for (size_t i = 0; i < s.count; ++i) CHECK(s.data[i] == 88.0);
    /* Row 0 succeeds, then row 1's permitted product exceeds float range. */
    q[0] = 0; q[2] = FLT_MAX; k[2] = FLT_MAX;
    STATUS(dl_attention_tiled_core(c, qm, km, vm, &s, out), DL_NUMERIC);
    CHECK(output[0] == 1 && output[1] == 2 && output[2] == 77);
    q[0] = 1; q[2] = 1; k[2] = 1;
    OK(dl_attention_tiled_core(c, qm, km, vm, &s, out));
    CHECK(oracle(c, matrix(q, 3, 2), matrix(k, 3, 2), matrix(v, 3, 2), out, &s));
    /* Equal enormous scores: lse-max loses log(3), but split statistics survive. */
    for (size_t i = 0; i < 3; ++i) { q[2 * i] = 1e19f; k[2 * i] = 1e19f; }
    OK(dl_attention_tiled_core(c, qm, km, vm, &s, out));
    CHECK(output[4] == 3 && output[5] == 4 && s.data[5] == 3);
    CHECK(s.data[8] == s.data[2]); /* Not enough double spacing for adding log(3). */
    CHECK(exp(s.data[2] - s.data[8]) == 1.0);
    CHECK(near(exp(0.0) / s.data[5], 1.0 / 3, 1e-15));
    /* A dominant maximum discards tiny old terms without exp overflow. */
    for (size_t i = 0; i < 3; ++i) q[2 * i] = 1;
    k[0] = -1000; k[2] = 0; k[4] = 1000;
    OK(dl_attention_tiled_core(c, qm, km, vm, &s, out));
    CHECK(output[4] == 5 && output[5] == 6);
    puts("tiled limits: budgets/overflow/preflight, partial failure/retry, huge equal scores and rescaling");
    result = 1;
done:
    dl_tiled_workspace_destroy(&rejected);
    dl_tiled_workspace_destroy(&s);
    dl_tiled_workspace_destroy(&s);
    return result;
}
```

The huge-equal-score test intentionally demonstrates a limitation rather than hiding it. Three equal enormous scores must still average the values to $(3,4)$. The stored scaled sum must be three. The saved scalar log-sum-exp equals the maximum after rounding, so reconstructing a maximum with that scalar alone returns one. Reconstructing with the retained maximum and sum gives $1/3$.

The final score pattern has a strongly dominant last key. It checks that large changes of maximum do not form overflowing positive exponentials. Some very small weights may underflow; that is allowed and is not confused with an empty causal row.

## Check the whole attention sublayer, including nonstandard width

The core does not know input projections or RoPE. A complete integration test must therefore exercise them rather than stopping at supplied projected arrays. The next fixture uses two batches, five tokens, four query heads sharing two key/value heads, hidden width six, and total query width eight. The independent materialized cache retains scores and probabilities; the tiled cache has only linear views.

```c title="tests/test_tiled.c"
static int sublayer(DlBackend backend)
{
    int result = 0;
    DlTiledWorkspace s = {0};
    DlAttentionConfig c = {2, 5, 4, 2, 2};
    /* Two independent float cache owners, with eight/six non-overlapping views. */
    float x[60], wq[48], wk[24], wv[24], wo[48], ref[760], tiled[360];
    float y[60], got[60];
    size_t pos[10] = {0, 1, 3, 4, 7, 2, 3, 4, 6, 9};
    double freq[] = {1.0};
    for (size_t i = 0; i < 60; ++i) x[i] = (float)((int)(i % 13) - 6) * 0.09f;
    for (size_t i = 0; i < 48; ++i) {
        wq[i] = (float)((int)(i % 11) - 5) * 0.11f;
        wo[i] = (float)((int)(i % 7) - 3) * 0.08f;
    }
    for (size_t i = 0; i < 24; ++i) {
        wk[i] = (float)((int)(i % 5) - 2) * 0.12f;
        wv[i] = (float)((int)(i % 9) - 4) * 0.07f;
    }
    DlAttentionWeights w = {dl_matrix_read(matrix(wq, 8, 6)), dl_matrix_read(matrix(wk, 4, 6)),
        dl_matrix_read(matrix(wv, 4, 6)), dl_matrix_read(matrix(wo, 6, 8))};
    DlAttentionCache a = {
        {ref,760,0,10,8,8}, {ref,760,80,10,4,4}, {ref,760,120,10,4,4},
        {ref,760,160,10,8,8}, {ref,760,240,10,4,4}, {ref,760,280,40,5,5},
        {ref,760,480,40,5,5}, {ref,760,680,10,8,8}};
    DlTiledAttentionCache ta = {
        {tiled,360,0,10,8,8}, {tiled,360,80,10,4,4}, {tiled,360,120,10,4,4},
        {tiled,360,160,10,8,8}, {tiled,360,240,10,4,4}, {tiled,360,280,10,8,8}};
    OK(dl_tiled_workspace_create(&s, 40, 2, 3, 2, 65536));
    OK(dl_attention_forward(backend, c, dl_matrix_read(matrix(x, 10, 6)), w,
                             pos, 10, freq, 1, a, matrix(y, 10, 6)));
    OK(dl_attention_tiled_forward(backend, c, dl_matrix_read(matrix(x, 10, 6)), w,
                                   pos, 10, freq, 1, ta, &s, matrix(got, 10, 6)));
    for (size_t i = 0; i < 60; ++i) CHECK(near(got[i], y[i], 2e-6));
    CHECK(memcmp(ref, tiled, 280 * sizeof(float)) == 0); /* All projected/rotated inputs. */
    for (size_t i = 0; i < 80; ++i) CHECK(near(ref[680 + i], tiled[280 + i], 5e-7));
    for (size_t i = 0; i < 360; ++i) tiled[i] = 77.0f;
    for (size_t i = 0; i < 60; ++i) got[i] = 77.0f;
    for (size_t i = 0; i < s.count; ++i) s.data[i] = 88.0;
    pos[9] = DL_ROPE_MAX_POSITION + 1;
    STATUS(dl_attention_tiled_forward(backend, c, dl_matrix_read(matrix(x, 10, 6)), w,
                                       pos, 10, freq, 1, ta, &s, matrix(got, 10, 6)), DL_RANGE);
    pos[9] = 9;
    STATUS(dl_attention_tiled_forward(backend, c, dl_matrix_read(matrix(x, 10, 6)), w,
                                       pos, 10, freq, 1, ta, &s, matrix(x, 10, 6)), DL_ALIAS);
    wv[23] = NAN;
    STATUS(dl_attention_tiled_forward(backend, c, dl_matrix_read(matrix(x, 10, 6)), w,
                                       pos, 10, freq, 1, ta, &s, matrix(got, 10, 6)), DL_NUMERIC);
    for (size_t i = 0; i < 360; ++i) CHECK(tiled[i] == 77);
    for (size_t i = 0; i < 60; ++i) CHECK(got[i] == 77);
    for (size_t i = 0; i < s.count; ++i) CHECK(s.data[i] == 88);
    result = 1;
done:
    dl_tiled_workspace_destroy(&s);
    return result;
}
```

The first 280 floats of each cache represent the same projected and rotated arrays; they are compared byte-for-byte because these preceding operations are identical. Context and final output use numerical tolerances because the materialized reference rounds probabilities to float. Both matrix backends execute the test.

The fixture then resets all new destinations and statistics to sentinels and rejects an out-of-range absolute position, a destructive output alias, and a nonfinite value weight. Those preflight failures must not begin even the first projection. Numeric failures later in the computation have the broader invalidation contract explained in Part 1.

## Check complete logits and the actual owners

For the model test, use capacities crossing both starting tile dimensions. Prefixes from one through 33 include exact boundaries at 16 and 32 and their tails. Four query heads use one, two, or four key/value heads; the corresponding configurations have one, two, or three decoder blocks. Both matrix backends are exercised.

```c title="tests/test_tiled.c"
static int model_parity(size_t kv_heads, size_t layers, DlBackend backend)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlInference full = {0}, tiled = {0}, rejected = {0}, doubled = {0};
    enum { C = 33, V = 16, D = 16, H = 4, F = 20 };
    DlModelConfig config = {.vocab = V, .layers = layers, .width = D,
        .query_heads = H, .kv_heads = kv_heads, .ffn_width = F,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    size_t ids[C];
    float expected[C * V], output[C * (V + 2) + 2];
    for (size_t i = 0; i < C; ++i) ids[i] = (3 * i + 1) % V;
    OK(dl_model_create(&model, config));
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_inference_create(&full, &weights, C, 1024 * 1024));
    OK(dl_inference_create_tiled(&tiled, &weights, C, 16, 32, 1024 * 1024));
    size_t linear = 7 * C * D + 3 * C * kv_heads * (D / H) + 3 * C * F;
    size_t doubles = 3 * H * C + 16 * 32 + 16 * (D / H) + 2 * 16;
    CHECK(tiled.arena.count == linear && tiled.tiled.count == doubles);
    CHECK(full.arena.count == linear + 2 * H * C * C && full.tiled.data == NULL);
    size_t bytes = linear * sizeof(float) + doubles * sizeof(double);
    STATUS(dl_inference_create_tiled(&rejected, &weights, C, 16, 32, bytes - 1), DL_RANGE);
    CHECK(rejected.arena.data == NULL && rejected.tiled.data == NULL && rejected.weights == NULL);
    OK(dl_inference_create_tiled(&rejected, &weights, C, 16, 32, bytes));
    STATUS(dl_inference_create_tiled(&rejected, &weights, C, 16, 32, bytes), DL_STATE);
    OK(dl_inference_create_tiled(&doubled, &weights, 2 * C, 16, 32, 1024 * 1024));
    CHECK(doubled.arena.count == 2 * linear);
    CHECK(doubled.tiled.count - tiled.tiled.count == 3 * H * C);
    for (size_t n = 1; n <= C; ++n) {
        DlMatrix want = matrix(expected, n, V);
        DlMatrix got = {output, sizeof(output) / sizeof(output[0]), 1, n, V, V + 2};
        for (size_t i = 0; i < sizeof(output) / sizeof(output[0]); ++i) output[i] = 77;
        for (size_t i = 0; i < tiled.arena.count; ++i) tiled.arena.data[i] = NAN;
        for (size_t i = 0; i < tiled.tiled.count; ++i) tiled.tiled.data[i] = NAN;
        OK(dl_inference_forward(&full, backend, ids, n, 0, want));
        OK(dl_inference_forward(&tiled, backend, ids, n, 0, got));
        for (size_t r = 0; r < n; ++r)
            for (size_t j = 0; j < V; ++j) CHECK(near(*cell(got, r, j), expected[r * V + j], 5e-6));
        for (size_t i = 0; i < sizeof(output) / sizeof(output[0]); ++i) {
            int logical = i >= 1 && (i - 1) / (V + 2) < n && (i - 1) % (V + 2) < V;
            if (!logical) CHECK(output[i] == 77);
        }
        DlMatrix last = {output, sizeof(output) / sizeof(output[0]), 1, 1, V, V + 2};
        OK(dl_inference_forward(&tiled, backend, ids, n, n - 1, last));
        for (size_t j = 0; j < V; ++j) CHECK(near(output[1 + j], expected[(n - 1) * V + j], 5e-6));
    }
    /* Same prefix and RNG for each strategy; token equality is a fixture fact,
     * never a promise for all near-ties or categorical boundary cases. */
    for (int sampled = 0; sampled < 2; ++sampled) {
        DlRng rng = {2026}, other_rng = {2026};
        double probabilities[V];
        ids[0] = 2; ids[1] = 3; ids[2] = 4;
        for (size_t step = 0; step < 8; ++step) {
            size_t n = 3 + step, a = 0, b = 0;
            OK(dl_inference_forward(&full, backend, ids, n, n - 1, matrix(expected, 1, V)));
            OK(dl_inference_forward(&tiled, backend, ids, n, n - 1, matrix(output, 1, V)));
            if (sampled) {
                OK(dl_sample_next(expected, V, 0.8, &rng, probabilities, &a));
                OK(dl_sample_next(output, V, 0.8, &other_rng, probabilities, &b));
            } else {
                OK(dl_greedy(expected, V, &a));
                OK(dl_greedy(output, V, &b));
            }
            CHECK(a == b && rng.state == other_rng.state);
            ids[n] = a;
        }
    }
    result = 1;
done:
    dl_inference_destroy(&doubled);
    dl_inference_destroy(&rejected);
    dl_inference_destroy(&tiled);
    dl_inference_destroy(&full);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}
```

For each prefix, the test compares all vocabulary rows and then a last-row-only request. Poisoned float and double scratch prevent an accidental dependency on previous lengths or layers. Output padding remains unchanged. Exact combined-budget tests cover the float arena and double owner together, and a doubled-capacity owner must show linear growth when tile dimensions stay fixed.

The test also performs eight greedy and eight sampled choices for each configuration/backend pair. Both strategies use identical prefixes and independently initialized equal RNG states. Agreement of all 96 choices is useful for this deterministic fixture. It is not a universal claim about near-tied logits or sampling thresholds.

The main function makes the full sweep and expected case counts explicit:

```c title="tests/test_tiled.c"
int main(void)
{
    if (!hand_row() || !limits_and_failures()) return EXIT_FAILURE;
    const size_t times[] = {1, 2, 3, 15, 16, 17, 31, 32, 33, 65};
    const size_t heads[] = {1, 2, 4};
    const size_t tiles[][2] = {{1,1}, {2,3}, {3,2}, {7,5}, {16,32}, {64,128}};
    size_t cases = 0;
    for (size_t t = 0; t < 10; ++t)
        for (size_t h = 0; h < 3; ++h)
            for (size_t z = 0; z < 6; ++z) {
                if (!core_case(times[t], heads[h], tiles[z][0], tiles[z][1])) return EXIT_FAILURE;
                ++cases;
            }
    printf("tiled core: %zu independent two-pass cases; tails, masks, padding and batch isolation\n", cases);
    for (int backend = 0; backend < 2; ++backend) {
        if (!sublayer((DlBackend)backend)) return EXIT_FAILURE;
        for (size_t i = 0; i < 3; ++i)
            if (!model_parity(heads[i], i + 1, (DlBackend)backend)) return EXIT_FAILURE;
    }
    puts("tiled integration: two batched sublayers, 396 full/subset logit calls, 96 token choices");
    puts("tiled storage: exact combined budgets and linear capacity growth checked");
    puts("all tiled forward checks passed");
    return EXIT_SUCCESS;
}

#undef OK
#undef STATUS
#undef CHECK
```

There are $10\cdot3\cdot6=180$ core cases. Their tile dimensions include singleton tiles, unequal narrow tiles, the initial 16-by-32 choice, and tiles larger than the entire input. The six decoder configurations each compare 33 all-row and 33 last-row calls, for 396 prefix/subset comparisons. Two batched sublayer cases test the complete attention wrapper separately.

## Extend the process-level generator test

Replace `tests/test_generate_cli.py`. Python still prepares fixtures, launches C binaries, compares resulting bytes, and checks process behavior; it does not calculate model logits. The only strategy extension is admitting `tiled`. Every selected strategy is compared against full-prefix/legacy execution; the suite invokes all three so cached and tiled variants are both covered.

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

The zero-weight helper is an existing diagnostic asset transformation. It creates a model with equal logits, making greedy's lowest-ID tie rule produce zero bytes. That tests binary output and escaping without relying on a trained model to happen to sample a NUL byte.

```python title="tests/test_generate_cli.py"
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
```

Deleting the prepared training files before generation is deliberate: the weight-only checkpoint path must not consult original datasets. The operating-system file-size limit causes a real output write failure when the platform exposes `resource`; it tests cleanup rather than replacing `fwrite` with a mock. Platforms without that module print a narrow skip instead of claiming the failure injection ran.

## Build and run the chapter

From the complete Chapter 18 source tree, run:

```sh
make test
make test-assets test-training-cli test-generation-cli
ASAN_OPTIONS=detect_leaks=1 ./build/sanitize/test_tiled
```

For a compatible compiler outside the default Apple toolchain, use `make CC=clang test` and the same override for the other make targets. For a focused first build, `make build/debug/test_tiled` and `./build/debug/test_tiled` avoid running the whole existing suite until the new checks work.

The tiled executable's expected output is:

```text
tiled hand: context=[1.857143, 1.142857] max=1.386294 sum=1.750000 lse=1.945910
tiled limits: budgets/overflow/preflight, partial failure/retry, huge equal scores and rescaling
tiled core: 180 independent two-pass cases; tails, masks, padding and batch isolation
tiled integration: two batched sublayers, 396 full/subset logit calls, 96 token choices
tiled storage: exact combined budgets and linear capacity growth checked
all tiled forward checks passed
```

This output was actually observed in the authoring environment with Clang 17 on x86_64 Linux in debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized builds. The cumulative suite passed all eighteen C test programs, the inherited comparison checks, nine offline preparation tests, and training/generation CLI tests. The compiler emitted no warnings. The leak-enabled tiled test reported no leaks. These are not native M4/macOS observations.

Source reconstructed from the chapter's named code blocks is checked separately against the companion snapshot. A passing companion is insufficient if an essential helper or build edit is missing from the actual teaching text.

## Make four wrong algorithms fail for the right reason

Work in a disposable copy of the source tree and restore the correct file between variations. Keep `make`'s non-fast-math flags. Each change below is small enough to inspect line by line.

**Lose the old numerator's scale.** In `attend`, replace `u[r * d + j] *= alpha;` with `u[r * d + j] *= 1.0;`. The denominator still rescales, but old values remain in the previous exponential scale. The three-key hand comparison fails. No memory access has become illegal, so a sanitizer alone cannot diagnose the error.

**Forget the old denominator.** Replace `denominator[r] *= alpha;` with `denominator[r] = 0.0;`. This discards previous key tiles rather than merging them. The one-key-tile hand result fails even though the algorithm may look plausible when a whole sequence fits into one tile. That is why testing only one large tile is weak evidence.

**Expose future columns.** Change the score-generation branch from `if (key_time > t)` to `if (0)`, and change `const size_t allowed = smaller(nk, t - kb + 1);` to `const size_t allowed = nk;`. Both changes are needed because production code enforces causality at score generation and at merge. The two-key tile now gives query zero access to a future value, so the hand fixture fails. Changing only one guard can leave the other guard masking the defect.

**Map shared heads by modulo.** Replace `const size_t kh = h / group;` with `const size_t kh = h % c.kv_heads;`, removing the now-unused `group` declaration. Singleton-head hand checks still pass, but the independent multihead oracle fails. Contiguous groups `[0,0,1,1]` are not the alternating map `[0,1,0,1]` when four query heads share two key/value heads.

All four variants were compiled and run under AddressSanitizer/UndefinedBehaviorSanitizer in the authoring environment. Each exited through an explicit numerical assertion, with no sanitizer memory report. They are intended to teach numerical and structural diagnosis, not to suggest that a quiet sanitizer validates an algorithm.

For a **valid** variation, replace `{7,5}` in the test's tile table with `{4,11}`. Predict the allocation change from $RS+Rd+2R$, then rerun the same oracle, sublayer, and model checks. This variation also passed in the authoring environment. Tile dimensions are a scheduling choice, not part of the model's learned architecture.

## Run a trained checkpoint through all three strategies

Use a fresh run directory so the trainer's no-overwrite rule remains useful. The supplied original diagnostic text is small, deterministic, and generated locally. No network dataset is needed for this experiment.

```sh
mkdir runs-chapter18
python3 scripts/make_training_fixture.py runs-chapter18/data
./build/release/train new configs/byte-pilot.conf \
  runs-chapter18/data/train.bin runs-chapter18/data/valid.bin \
  runs-chapter18/pilot.bin 160 160 2 random blocked
printf 'Once there' > runs-chapter18/prompt.bin
for selection in greedy sample; do
  for strategy in full cache tiled; do
    ./build/release/generate configs/byte-pilot.conf \
      runs-chapter18/pilot.bin runs-chapter18/prompt.bin \
      "runs-chapter18/${selection}-${strategy}.bin" \
      64 40 "$selection" 0.8 2026 blocked "$strategy"
  done
  cmp "runs-chapter18/${selection}-full.bin" "runs-chapter18/${selection}-cache.bin"
  cmp "runs-chapter18/${selection}-full.bin" "runs-chapter18/${selection}-tiled.bin"
done
```

`cmp` produces no output when the two files are identical. A repeat must use another run directory or intentionally chosen new output paths; do not turn a refusal into a command that silently overwrites unrelated files.

The actual authoring run used the 29,856-parameter pilot for 160 materialized-training updates. Validation on 128 targets from the separate diagnostic validation file fell from 5.725389095 to 2.270204153; 264 trailing targets were not scored by that bounded validation request. This repeats a small mechanism check, not a claim about broad text quality. Tiled training is not implemented in this chapter.

The tiled greedy run printed:

```text
prompt=10 new=40 total=50 capacity=64 config_time=16
payload_bytes weights=119424 workspace=131072 logits=1024 probabilities=2048
inference=tiled consumed=49 kv_payload_bytes=0
tiled_double_payload_bytes=11520
sampling_rng=2026 mode=greedy backend=blocked
bytes="Once thered a athed a athe at athed a a atooooat a"
```

The sampled run used the same prompt, temperature 0.8, and seed 2026:

```text
sampling_rng=13306735003898702642 mode=sample backend=blocked
bytes="Once thereind thed a thed a ond s foowoped in sern"
```

Both complete 50-byte transcripts matched their materialized full-prefix and cached counterparts byte-for-byte in this run. The text remains repetitive and malformed. Showing the actual output is more informative than silently replacing it with a plausible continuation.

These runs provide a complete route from a checkpoint to tiled C inference and byte selection. They do not measure an attention speedup. The inherited matrix timing command is not an attention benchmark; its results should not be relabeled as one. Likewise, the printed payload counts are not process resident memory. No native M4 run, pretrained-model comparison, or measured process footprint is claimed here.

## Read symptoms as evidence about a particular layer

| Symptom | First place to inspect |
|---|---|
| Correct for one tile, wrong for several | Rescaling or persistence of old numerator/denominator |
| Correct without head sharing, wrong with sharing | Contiguous query-head grouping and key/value coordinate offset |
| Early rows change after a future value changes | Per-row causal boundary inside the diagonal tile |
| Only padded or tail cases fail | Physical leading dimensions, tile stride, and active lengths |
| Context agrees, final logits diverge | Output projection, residual/norm integration, or layer workspace lifetime |
| Outputs agree but memory still grows quadratically | The constructor/layout still reserves unused score/probability arrays |
| Huge tied scores give probabilities near one | Reconstruction from a single rounded log-sum-exp instead of split statistics |
| Retry fails after an earlier numeric error | Reading stale scratch or treating partial statistics as a successful tape |

Do not “fix” a structural error by widening every tolerance. Start with the smallest failing hand or independent-oracle case, inspect the first divergent state $(m,\ell,u)$, and connect its addresses to the shapes. Use the model comparison only after the core and complete sublayer agree.

## What carries into the next chapter

The engine now has materialized training, materialized full-prefix inference, persistent cached inference, and tiled full-prefix inference. The last of these retains linear projected arrays and row statistics, not square scores or probabilities. Its output is numerically checked against independent and materialized references.

The next chapter adds the missing paired tiled backward and a training tape that preserves the right per-layer state. It will reconstruct allowed probabilities from scores and saved normalization, accumulate shared-head derivatives, and compare complete updates. Until then, the existing trainer remains on its materialized forward/backward pair.

### Technical references

[Milakov and Gimelshein, *Online normalizer calculation for softmax*, Sections 3 and 3.1](https://arxiv.org/html/1805.02867v1#S3), gives the online normalizer and merge formulation. [Dao et al., *FlashAttention*, Section 3.1 and Appendix B.3](https://arxiv.org/html/2205.14135v2#S3.SS1), connects attention tiling, saved normalization statistics, and memory traffic. The recurrence, address traces, C owner policy, numerical boundary choices, and fixtures in this chapter are specified directly above; neither reference is a substitute for missing implementation steps.
