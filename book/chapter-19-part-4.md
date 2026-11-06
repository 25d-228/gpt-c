# Chapter 19 — Checking tiled core and sublayer gradients

This continues [Part 3's training integration](https://25d-228.github.io/gpt-c/chapter-19-part-3.html). Before trusting loss curves, isolate three different questions: did we differentiate the intended formula, did the tiled traversal preserve it, and did the surrounding projections/RoPE connect the right buffers?

## Use independent routes to the same answer

Comparing tiled gradients only with materialized gradients is insufficient: both implementations could share a mistaken formula. Comparing only finite differences is also insufficient: both forward and backward could consistently use the wrong causal mask or head mapping. Keep the hand calculation, an independently written dense oracle, parity with the earlier operator, and numerical differentiation.

The new dense backward oracle uses a complete **test-only** score/probability row. It forms the softmax correction as $\sum_s p_{ts}\bar p_{ts}$, rather than using the implementation's saved-context identity. It then visits query heads and adds into key/value derivatives directly. This deliberately differs from the implementation's two output-owned sweeps.

For numerical differences, define the scalar objective

$$
F(Q,K,V)=\sum_{b,t,h,j} C_{b,t,h,j}\,g_{b,t,h,j}.
$$

The upstream array $g$ is fixed while one input coordinate changes. For a float coordinate $x$, compute representable samples $x_+=\operatorname{float}(x+0.002)$ and $x_-=\operatorname{float}(x-0.002)$, rerun the entire forward for each, and divide by the **represented difference** $\operatorname{double}(x_+)-\operatorname{double}(x_-)$. Chapter 3 explains why writing a smaller nominal step is not automatically more accurate.

The checker uses $|a-n|\leq\epsilon(1+|n|)$ with finite checks on both values. Core finite differences use $\epsilon=4\times10^{-4}$; direct operator comparisons use substantially tighter tolerances. These are explicit diagnostic tolerances for the supplied small inputs, not a universal error guarantee. Keep the analytic gradients from the unperturbed forward and restore successful unperturbed statistics before another backward call.

## Replace the complete test file

The following blocks concatenate to `tests/test_tiled.c`. Chapter 18's forward, storage, and inference tests remain. The full-sublayer test gains backward checks; three new helpers then exercise the core independently. All helper definitions and cleanup paths are included.

### Small assertions and padded indexing

`near` rejects NaN and infinity before comparing error. `matrix` is only a tiny-fixture dense-view constructor; the larger cases use checked owned tensors and padded views. `cell` applies both offset and leading dimension, so the test does not accidentally assume compact rows when the implementation promises padding support.

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

### Retain the independently calculated forward row

The known probabilities $[1,2,4]/7$ establish coordinate conventions before backward is considered. Run several tile sizes, including capacities larger than the sequence. The saved statistics are checked independently, not inferred from the final context alone.

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

### Keep the separate forward oracle

This helper owns a bounded stack row of 65 scores and refuses larger test sequences. It reproduces the documented float score boundary but performs a conventional two-pass normalization. It does not call the online-normalization implementation to obtain the expected answer.

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

### Retain tails, padding, and batch isolation

Initialize every owner to a guard value, then overwrite only logical matrix cells. The allocated row stride is wider than the logical width. Success requires numerical agreement **and** unchanged guards, including the unused normalization capacity. A batch-boundary or tail-indexing mistake can otherwise escape a dense, single-batch fixture.

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

### Retain constructor and forward failure tests

An impossibly large tile must fail checked arithmetic without publishing an owner. A bad input found during preflight must preserve destination values. A later numerical failure can invalidate a partial result; a complete successful retry must reconstruct it. The extreme equal-score example checks the split maximum/sum representation even when the combined log-sum-exp loses its small correction.

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

## Differentiate the complete sublayer, not just Q/K/V

This fixture has two batches, five token rows per batch, four query heads, two key/value heads, head width two, and input/output width **six**. The concatenated query width is eight. Deliberately choosing $D\ne H_qd$ catches projection code that substitutes the model width where the isolated attention API allows a separate query width.

The test constructs independent materialized and tiled caches, checks their forward values, then compares input and all four weight gradients under every independent SET/ADD combination. Initial SET destinations contain NaN; ADD destinations contain $0.125$. SET must not read old values, while ADD must retain them.

Next, perturb every input and weight coordinate and differentiate $\sum_iY_i\,\bar Y_i$. There are 204 coordinates per matrix backend, hence 408 sublayer comparisons. This includes the output projection, both rotary transposes, all shared heads, and the three input-gradient contributions. The last analytic call used ADD, so subtract the known $0.125$ initial value before comparing with a numerical derivative.

```c title="tests/test_tiled.c"
static int sublayer(DlBackend backend)
{
    int result = 0;
    DlTiledWorkspace s = {0};
    DlTiledBackwardWorkspace back = {0};
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
    /* Check the entire backward wrapper, including D=6 while Q=8. */
    float rg[760], tg[360], ref_grad[204], tiled_grad[204], dy[60];
    for (size_t i=0;i<60;++i) dy[i]=(float)((int)(i%7)-3)*0.17f;
    DlAttentionScratch rs={
        {rg,760,680,10,8,8},{rg,760,480,40,5,5},{rg,760,280,40,5,5},
        {rg,760,160,10,8,8},{rg,760,240,10,4,4},{rg,760,120,10,4,4},
        {rg,760,0,10,8,8},{rg,760,80,10,4,4}};
    DlTiledAttentionScratch ts={
        {tg,360,280,10,8,8},{tg,360,160,10,8,8},{tg,360,240,10,4,4},
        {tg,360,120,10,4,4},{tg,360,0,10,8,8},{tg,360,80,10,4,4}};
    DlMatrix rx={ref_grad,204,0,10,6,6}, tx={tiled_grad,204,0,10,6,6};
    DlAttentionWeightGrads rw={
        {ref_grad,204,60,8,6,6},{ref_grad,204,108,4,6,6},
        {ref_grad,204,132,4,6,6},{ref_grad,204,156,6,8,8}};
    DlAttentionWeightGrads tw={
        {tiled_grad,204,60,8,6,6},{tiled_grad,204,108,4,6,6},
        {tiled_grad,204,132,4,6,6},{tiled_grad,204,156,6,8,8}};
    OK(dl_tiled_backward_workspace_create(&back,2,2,3,65536));
    for (int xm=0;xm<2;++xm)
        for (int wm=0;wm<2;++wm) {
            for (size_t i=0;i<204;++i)
                ref_grad[i]=tiled_grad[i]=(i<60 ? xm : wm)==0?NAN:0.125f;
            OK(dl_attention_backward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
                pos,10,freq,1,a,dl_matrix_read(matrix(dy,10,6)),rs,rx,(DlWriteMode)xm,rw,(DlWriteMode)wm));
            OK(dl_attention_tiled_backward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
                pos,10,freq,1,ta,&s,dl_matrix_read(matrix(dy,10,6)),ts,&back,
                tx,(DlWriteMode)xm,tw,(DlWriteMode)wm));
            for (size_t i=0;i<204;++i) CHECK(near(tiled_grad[i],ref_grad[i],4e-6));
            /* Compare all rotated and unrotated projection derivatives too. */
            for (size_t i=0;i<280;++i) CHECK(near(tg[i],rg[i],4e-6));
            for (size_t i=0;i<80;++i) CHECK(near(tg[280+i],rg[680+i],4e-6));
        }
    /* Last mode was ADD to .125. Its saved derivatives now support independent
     * differences of the scalar sum(Y*dy), including every input and weight. */
    float *variables[]={x,wq,wk,wv,wo};
    const size_t lengths[]={60,48,24,24,48};
    size_t coordinate=0;
    for (size_t aidx=0;aidx<5;++aidx)
        for (size_t j=0;j<lengths[aidx];++j,++coordinate) {
            float *value=&variables[aidx][j], old=*value;
            float plus=old+0.002f, minus=old-0.002f;
            double objective[2]={0};
            for (size_t side=0;side<2;++side) {
                *value=side==0?plus:minus;
                OK(dl_attention_tiled_forward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
                    pos,10,freq,1,ta,&s,matrix(got,10,6)));
                for (size_t i=0;i<60;++i) objective[side]+=(double)got[i]*(double)dy[i];
            }
            *value=old;
            double numeric=(objective[0]-objective[1])/((double)plus-(double)minus);
            CHECK(near(numeric,(double)tiled_grad[coordinate]-0.125,3e-5));
        }
    OK(dl_attention_tiled_forward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
        pos,10,freq,1,ta,&s,matrix(got,10,6)));
    for (size_t i=0;i<204;++i) tiled_grad[i]=77;
    for (size_t i=0;i<360;++i) tg[i]=88;
    pos[9]=DL_ROPE_MAX_POSITION+1;
    STATUS(dl_attention_tiled_backward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
        pos,10,freq,1,ta,&s,dl_matrix_read(matrix(dy,10,6)),ts,&back,tx,DL_SET,tw,DL_SET),DL_RANGE);
    pos[9]=9;
    DlAttentionWeightGrads wrong=tw; wrong.output=matrix(x,6,8);
    STATUS(dl_attention_tiled_backward(backend,c,dl_matrix_read(matrix(x,10,6)),w,
        pos,10,freq,1,ta,&s,dl_matrix_read(matrix(dy,10,6)),ts,&back,tx,DL_SET,wrong,DL_SET),DL_ALIAS);
    for (size_t i=0;i<204;++i) CHECK(tiled_grad[i]==77);
    for (size_t i=0;i<360;++i) CHECK(tg[i]==88);

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
    dl_tiled_backward_workspace_destroy(&back);
    dl_tiled_workspace_destroy(&s);
    return result;
}
```

### Keep inference independent of the new training tape

The existing full-prefix and tiled inference workspaces still run selected-prefix logits and deterministic token choices. They borrow immutable weights and do not construct an optimizer. Retaining this test catches an accidental regression in the forward implementation while backward is being added.

These are fixture comparisons, not a guarantee that every sampled sequence remains identical under arbitrary near-tied logits. The sampling boundary tests from Chapter 16 remain necessary.

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

## Implement a separate dense backward oracle

This helper computes probabilities and incoming probability derivatives into its own test-only rows. It obtains the correction by summing their products and immediately accumulates double-precision Q/K/V derivatives in a different traversal from the tiled implementation. The score scale is explicit in both Q and K; V is unscaled.

Only the test uses these rows and expected-gradient arrays. They are not part of the model's production tape or its reported allocation count.

```c title="tests/test_tiled.c"
/* Independent, deliberately materialized test oracle. Max 2*65 rows and width16.
 * Center comes from sum(p*dP), NOT the production context-dot identity. */
static int backward_oracle(DlAttentionConfig c, DlMatrix q, DlMatrix k, DlMatrix v,
                            DlMatrix dc, double want[3][2080])
{
    if (c.time > 65 || c.batch > 2 || c.query_heads * c.head_dim > 16) return 0;
    for (size_t a = 0; a < 3; ++a)
        for (size_t i = 0; i < 2080; ++i) want[a][i] = 0.0;
    size_t qw = c.query_heads * c.head_dim, kw = c.kv_heads * c.head_dim;
    double scale = 1.0 / sqrt((double)c.head_dim);
    for (size_t b = 0; b < c.batch; ++b)
        for (size_t h = 0; h < c.query_heads; ++h) {
            size_t kh = h / (c.query_heads / c.kv_heads);
            for (size_t t = 0; t < c.time; ++t) {
                double p[65], dp[65], maximum = -INFINITY, sum = 0.0, center = 0.0;
                for (size_t key = 0; key <= t; ++key) {
                    double score = 0.0;
                    dp[key] = 0.0;
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        score += (double)*cell(q,b*c.time+t,h*c.head_dim+j)
                               * (double)*cell(k,b*c.time+key,kh*c.head_dim+j);
                        dp[key] += (double)*cell(dc,b*c.time+t,h*c.head_dim+j)
                                 * (double)*cell(v,b*c.time+key,kh*c.head_dim+j);
                    }
                    p[key] = (double)(float)(score * scale);
                    maximum = fmax(maximum, p[key]);
                }
                for (size_t key = 0; key <= t; ++key) { p[key] = exp(p[key] - maximum); sum += p[key]; }
                for (size_t key = 0; key <= t; ++key) { p[key] /= sum; center += p[key] * dp[key]; }
                for (size_t key = 0; key <= t; ++key) {
                    double ds = p[key] * (dp[key] - center);
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        size_t qi = (b*c.time+t)*qw+h*c.head_dim+j;
                        size_t ki = (b*c.time+key)*kw+kh*c.head_dim+j;
                        want[0][qi] += scale * ds * (double)*cell(k,b*c.time+key,kh*c.head_dim+j);
                        want[1][ki] += scale * ds * (double)*cell(q,b*c.time+t,h*c.head_dim+j);
                        want[2][ki] += p[key] * (double)*cell(dc,b*c.time+t,h*c.head_dim+j);
                    }
                }
            }
        }
    return 1;
}
```

## Exercise every coordinate of small core fixtures

`backward_case` takes independent backward tile capacities while forward uses a fixed $3\times2$ tile. A compatible saved result does not require the backward partition to equal the forward partition; it requires the same input values, positions, head mapping, and successful row normalization.

The cases use two batches, $H_q=4$, $H_{kv}\in\{1,2,4\}$, and $d=4$, with sequence lengths 1, 3, 17, and 33. Six tile pairs include one-cell tiles, unequal dimensions, incomplete tails, and capacities larger than the sequence: $(1,1)$, $(2,3)$, $(3,2)$, $(7,5)$, $(16,32)$, and $(64,128)$. That is 72 direct backward-oracle cases.

The three-token cases perform 624 coordinate differences across Q, K, and V. Afterwards the test checks ADD to $0.25$, then isolates one incoming gradient at the first token of the first batch and head. With only one visible key, the softmax probability is exactly one and its score derivative is zero. Only that key's value gradient may remain; other batches, future rows, and unrelated heads must stay zero.

```c title="tests/test_tiled.c"
static size_t tiled_fd_checks = 0;
static double tiled_fd_error = 0.0;

static int backward_case(size_t time, size_t heads, size_t qr, size_t kc, int differences)
{
    int result = 0;
    DlAttentionConfig c = {2,time,4,heads,4};
    DlTensor owners[8] = {{0}};
    DlMatrix m[8] = {{0}}; /* Q,K,V,context,upstream,dQ,dK,dV. */
    DlTiledWorkspace forward = {0};
    DlTiledBackwardWorkspace backward = {0};
    double want[3][2080];
    size_t rows = 2*time, widths[] = {16,heads*4,heads*4,16,16,16,heads*4,heads*4};
    for (size_t a = 0; a < 8; ++a) {
        size_t count = 2 + rows * (widths[a]+2);
        OK(dl_tensor_create(&owners[a],1,&count));
        for (size_t i=0;i<count;++i) owners[a].data[i] = -777;
        OK(dl_matrix_view(&owners[a],1,rows,widths[a],widths[a]+2,&m[a]));
        for (size_t r=0;r<rows;++r)
            for (size_t j=0;j<widths[a];++j)
                *cell(m[a],r,j) = a==3 || a>=5 ? NAN
                    : (float)((int)((r*11+j*7+a*5)%23)-11)*0.09f;
    }
    OK(dl_tiled_workspace_create(&forward,rows*4+3,4,3,2,1024*1024));
    OK(dl_tiled_backward_workspace_create(&backward,4,qr,kc,1024*1024));
    CHECK(backward.count == 2*qr*kc+qr*4+2*kc*4+qr);
    OK(dl_attention_tiled_core(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
                               dl_matrix_read(m[2]),&forward,m[3]));
    OK(dl_attention_tiled_core_backward(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
        dl_matrix_read(m[2]),dl_matrix_read(m[3]),dl_matrix_read(m[4]),&forward,&backward,
        m[5],m[6],m[7],DL_SET));
    CHECK(backward_oracle(c,m[0],m[1],m[2],m[4],want));
    for (size_t a=0;a<3;++a)
        for (size_t r=0;r<rows;++r)
            for (size_t j=0;j<widths[a];++j)
                CHECK(near(*cell(m[5+a],r,j),want[a][r*widths[a]+j],4e-6));
    if (differences) {
        /* Every Q/K/V coordinate; only small time=3 fixtures take this path. */
        for (size_t a=0;a<3;++a)
            for (size_t r=0;r<rows;++r)
                for (size_t j=0;j<widths[a];++j) {
                    float *x=cell(m[a],r,j), old=*x, plus=old+0.002f, minus=old-0.002f;
                    double losses[2]={0};
                    for (size_t side=0;side<2;++side) {
                        *x = side==0 ? plus : minus;
                        OK(dl_attention_tiled_core(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
                                                   dl_matrix_read(m[2]),&forward,m[3]));
                        for (size_t rr=0;rr<rows;++rr)
                            for (size_t jj=0;jj<16;++jj)
                                losses[side]+=(double)*cell(m[3],rr,jj)*(double)*cell(m[4],rr,jj);
                    }
                    *x=old;
                    double numeric=(losses[0]-losses[1])/((double)plus-(double)minus);
                    double analytic=*cell(m[5+a],r,j);
                    tiled_fd_error=fmax(tiled_fd_error,fabs(numeric-analytic));
                    CHECK(near(numeric,analytic,4e-4));
                    ++tiled_fd_checks;
                }
        /* Restore successful state, not the final perturbed forward. */
        OK(dl_attention_tiled_core(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
                                   dl_matrix_read(m[2]),&forward,m[3]));
    }
    for (size_t a=5;a<8;++a)
        for (size_t r=0;r<rows;++r)
            for (size_t j=0;j<widths[a];++j) *cell(m[a],r,j)=0.25f;
    OK(dl_attention_tiled_core_backward(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
        dl_matrix_read(m[2]),dl_matrix_read(m[3]),dl_matrix_read(m[4]),&forward,&backward,
        m[5],m[6],m[7],DL_ADD));
    for (size_t a=0;a<3;++a)
        for (size_t r=0;r<rows;++r)
            for (size_t j=0;j<widths[a];++j)
                CHECK(near(*cell(m[5+a],r,j),0.25+want[a][r*widths[a]+j],4e-6));
    /* Only row0, batch0, head0 has upstream signal: future rows, other heads
     * and every other batch have zero derivative (K/V share only head0's group). */
    for (size_t r=0;r<rows;++r)
        for (size_t j=0;j<16;++j) *cell(m[4],r,j) = r==0 && j<4 ? 1.0f : 0.0f;
    OK(dl_attention_tiled_core_backward(c,dl_matrix_read(m[0]),dl_matrix_read(m[1]),
        dl_matrix_read(m[2]),dl_matrix_read(m[3]),dl_matrix_read(m[4]),&forward,&backward,
        m[5],m[6],m[7],DL_SET));
    for (size_t r=0;r<rows;++r) {
        for (size_t j=0;j<16;++j) CHECK(*cell(m[5],r,j)==0);
        for (size_t j=0;j<heads*4;++j) {
            CHECK(*cell(m[6],r,j)==0);
            CHECK(*cell(m[7],r,j)==(r==0 && j<4 ? 1.0f : 0.0f));
        }
    }
    for (size_t a=0;a<8;++a)
        for (size_t i=0;i<owners[a].count;++i) {
            int logical=i>=1 && (i-1)/m[a].ld<rows && (i-1)%m[a].ld<widths[a];
            if (!logical) CHECK(owners[a].data[i]==-777);
        }
    result=1;
done:
    dl_tiled_backward_workspace_destroy(&backward);
    dl_tiled_workspace_destroy(&forward);
    for (size_t a=0;a<8;++a) dl_tensor_destroy(&owners[a]);
    return result;
}
```

## Check shared-head arithmetic and failure boundaries by hand

The hand fixture uses two identical query heads sharing one key/value head. Each head's last-row score derivative is $[-4,-64,68]/49$. Key/value gradients receive the sum of both heads, while each query head retains its own gradient.

The same function tests zero tile capacities, size overflow, insufficient budgets, repeated destruction, invalid statistics, forbidden aliasing, bad modes, non-finite upstream gradients, and ADD to a NaN destination. Before preflight failures, it fills both destinations and scratch with sentinels and requires them to remain unchanged.

The enormous-score fixture then checks $\bar V=2/3$ for every coordinate: reconstructing probabilities from lossy log-sum-exp alone would produce two instead. Finally, finite `FLT_MAX` upstream values overflow a shared value derivative **after** query outputs were written. That proves the documented late-failure distinction. Repairing the input and restarting with SET recomputes the entire result.

```c title="tests/test_tiled.c"
static int backward_hand_and_failures(void)
{
    int result=0;
    DlTiledWorkspace f={0};
    DlTiledBackwardWorkspace s={0}, rejected={0};
    DlAttentionConfig c={1,3,2,1,2};
    float q[12], k[6]={0}, v[]={1,0,0,2,3,1}, context[12], dc[12]={0};
    float dq[12], dk[6], dv[6];
    for (size_t i=0;i<12;++i) q[i]=i%2==0?1.0f:0.0f;
    k[2]=(float)(sqrt(2.0)*log(2.0)); k[4]=(float)(sqrt(2.0)*log(4.0));
    dc[8]=2; dc[9]=-1; dc[10]=2; dc[11]=-1;
    DlConstMatrix qm=dl_matrix_read(matrix(q,3,4)), km=dl_matrix_read(matrix(k,3,2));
    DlConstMatrix vm=dl_matrix_read(matrix(v,3,2)), cm=dl_matrix_read(matrix(context,3,4));
    DlConstMatrix gm=dl_matrix_read(matrix(dc,3,4));
    DlMatrix qg=matrix(dq,3,4), kg=matrix(dk,3,2), vg=matrix(dv,3,2);
    STATUS(dl_tiled_backward_workspace_create(&rejected,2,0,2,65536),DL_SHAPE);
    STATUS(dl_tiled_backward_workspace_create(&rejected,2,SIZE_MAX,2,SIZE_MAX),DL_OVERFLOW);
    STATUS(dl_tiled_backward_workspace_create(&rejected,2,2,2,8),DL_RANGE);
    CHECK(rejected.data==NULL && rejected.count==0);
    OK(dl_tiled_backward_workspace_create(&s,2,2,2,22*sizeof(double)));
    CHECK(s.count==22);
    OK(dl_tiled_workspace_create(&f,6,2,2,2,65536));
    OK(dl_attention_tiled_core(c,qm,km,vm,&f,matrix(context,3,4)));
    OK(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET));
    /* Each head: p=[1,2,4]/7, dP=[2,-2,5], center=18/7,
     * dS=[-4,-64,68]/49. K/V receive TWO heads. */
    double ds[]={-4.0/49,-64.0/49,68.0/49}, pp[]={1.0/7,2.0/7,4.0/7};
    for (size_t key=0;key<3;++key) {
        CHECK(near(dk[2*key],2*ds[key]/sqrt(2.0),3e-7));
        CHECK(dk[2*key+1]==0);
        CHECK(near(dv[2*key],4*pp[key],3e-7));
        CHECK(near(dv[2*key+1],-2*pp[key],3e-7));
    }
    double want_q=ds[1]*log(2.0)+ds[2]*log(4.0);
    CHECK(near(dq[8],want_q,3e-7) && near(dq[10],want_q,3e-7));
    printf("tiled backward hand: dQ=%.6f shared dK=[%.6f, %.6f, %.6f]\n",
           dq[8],dk[0],dk[2],dk[4]);
    for (size_t i=0;i<12;++i) dq[i]=77;
    for (size_t i=0;i<6;++i) dk[i]=dv[i]=77;
    for (size_t i=0;i<s.count;++i) s.data[i]=88;
    double saved=f.data[f.row_capacity]; f.data[f.row_capacity]=0;
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET),DL_NUMERIC);
    f.data[f.row_capacity]=saved;
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,matrix(q,3,4),kg,vg,DL_SET),DL_ALIAS);
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,(DlWriteMode)77),DL_ARGUMENT);
    dc[11]=NAN;
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET),DL_NUMERIC);
    dc[11]=-1;
    for (size_t i=0;i<12;++i) CHECK(dq[i]==77);
    for (size_t i=0;i<6;++i) CHECK(dk[i]==77 && dv[i]==77);
    for (size_t i=0;i<s.count;++i) CHECK(s.data[i]==88);
    dq[0]=NAN;
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_ADD),DL_NUMERIC);
    OK(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET));
    /* Huge tied scores need split normalization in BACKWARD too. Equal values
     * make dS exactly zero, while dV independently reveals P=1/3 not P=1. */
    for (size_t i=0;i<12;++i) { q[i]=i%2==0?1e19f:0; dc[i]=i>=8?1.0f:0; }
    for (size_t i=0;i<6;++i) { k[i]=i%2==0?1e19f:0; v[i]=1; }
    OK(dl_attention_tiled_core(c,qm,km,vm,&f,matrix(context,3,4)));
    CHECK(f.data[2]==f.data[2*f.row_capacity+2]);
    OK(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET));
    for (size_t i=0;i<6;++i) CHECK(near(dv[i],2.0/3,1e-7) && dk[i]==0);
    /* Valid forward, enormous shared dV: earlier dQ writes succeed, then a
     * finite-input reduction exceeds float range. Retry SET after repair. */
    for (size_t i=0;i<12;++i) { q[i]=0; dc[i]=FLT_MAX; dq[i]=77; }
    for (size_t i=0;i<6;++i) { k[i]=0; v[i]=0; dk[i]=dv[i]=77; }
    OK(dl_attention_tiled_core(c,qm,km,vm,&f,matrix(context,3,4)));
    STATUS(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET),DL_NUMERIC);
    CHECK(dq[0]==0 && dv[0]==77);
    for (size_t i=0;i<12;++i) dc[i]=1;
    OK(dl_attention_tiled_core_backward(c,qm,km,vm,cm,gm,&f,&s,qg,kg,vg,DL_SET));
    CHECK(near(dv[0],11.0/3,2e-7));
    result=1;
done:
    dl_tiled_backward_workspace_destroy(&rejected);
    dl_tiled_backward_workspace_destroy(&s);
    dl_tiled_backward_workspace_destroy(&s);
    dl_tiled_workspace_destroy(&f);
    return result;
}
```

### Run the retained and new groups in one executable

The program stops at the first failed assertion. It reports case and difference counts so a silently skipped loop is visible. The new 72 backward-oracle cases and 1,032 core/sublayer numerical differences augment, rather than replace, Chapter 18's forward and inference tests.

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
    if (!backward_hand_and_failures()) return EXIT_FAILURE;
    size_t backward_cases=0;
    const size_t backward_times[]={1,3,17,33};
    for (size_t t=0;t<4;++t)
        for (size_t h=0;h<3;++h)
            for (size_t z=0;z<6;++z) {
                if (!backward_case(backward_times[t],heads[h],tiles[z][0],tiles[z][1],
                                   t==1 && z==1)) return EXIT_FAILURE;
                ++backward_cases;
            }
    printf("tiled backward: %zu oracle cases, %zu finite differences, max error %.6g\n",
           backward_cases,tiled_fd_checks,tiled_fd_error);
    puts("tiled sublayer backward: 408 finite differences, 8 independent SET/ADD combinations");
    puts("all tiled forward/backward checks passed");
    return EXIT_SUCCESS;
}

#undef OK
#undef STATUS
#undef CHECK
```

## Read the failure, not only the exit code

After building the updated sources, run `build/debug/test_tiled` first. The new numerical lines on the tested Linux/Clang build were:

```text
tiled backward hand: dQ=1.018502 shared dK=[-0.115446, -1.847136, 1.962582]
tiled backward: 72 oracle cases, 624 finite differences, max error 2.74503e-05
tiled sublayer backward: 408 finite differences, 8 independent SET/ADD combinations
all tiled forward/backward checks passed
```

The first line agrees with Part 1's fractions after rounding. The finite-difference error is nonzero because the forward stores float values and the numerical difference itself subtracts nearby results. The criterion is the stated tolerance and preserved mechanism, not printing zero error.

A failure in the huge-score value derivative with ordinary finite differences still passing suggests a normalization representation mistake, not a missing derivative term. A shared K/V mismatch but correct Q suggests missing head accumulation or premature stores. A failure only with SET-from-NaN suggests an unwanted read of an overwritten destination. A failure only when a tile crosses the diagonal suggests causal visibility or a stale masked cell.

Continue with [Part 5 — Whole-model, training, and memory checks](https://25d-228.github.io/gpt-c/chapter-19-part-5.html).
