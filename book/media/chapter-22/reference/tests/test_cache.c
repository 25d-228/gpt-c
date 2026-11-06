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
