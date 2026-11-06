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
