#include "dl/inference.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #condition); goto done; \
} } while (0)
#define OK(call) CHECK((call) == DL_OK)

typedef struct {
    size_t calls, fail_at, layers, time, width, vocab;
    int invalid;
    float last[16];
} Observer;

static DlStatus observe(void *context, const char *name, size_t index, DlConstMatrix a)
{
    Observer *o = context;
    ++o->calls;
    size_t end = 0;
    if (dl_matrix_end(a, &end) != DL_OK) o->invalid = 1;
    if (strcmp(name, "embedding") == 0 &&
        (index != 0 || a.rows != o->time || a.cols != o->width)) o->invalid = 1;
    if (strcmp(name, "logits") == 0) {
        if (index != o->layers || a.rows != 1 || a.cols != o->vocab) o->invalid = 1;
        for (size_t j = 0; j < a.cols && j < 16; ++j) o->last[j] = a.base[a.offset+j];
    }
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (!isfinite(a.base[a.offset+i*a.ld+j])) o->invalid = 1;
    return o->calls == o->fail_at ? DL_STATE : DL_OK;
}

static int check(DlBackend backend, int tiled)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlInference workspace = {0};
    DlModelConfig c = {.vocab=16, .layers=2, .width=8, .query_heads=4,
        .kv_heads=2, .ffn_width=12, .epsilon=1e-5f, .rope_theta=10000,
        .init_std=0.125f, .seed=42};
    size_t ids[] = {0, 3, 9, 3, 15};
    float baseline[16], actual[16];
    DlMatrix expected = {baseline, 16, 0, 1, 16, 16};
    DlMatrix output = {actual, 16, 0, 1, 16, 16};
    OK(dl_model_create(&model, c));
    OK(dl_weights_from_model(&weights, &model, 1024*1024));
    if (tiled) { OK(dl_inference_create_tiled(&workspace, &weights, 5, 2, 3, 1024*1024)); }
    else { OK(dl_inference_create(&workspace, &weights, 5, 1024*1024)); }
    OK(dl_inference_forward(&workspace, backend, ids, 5, 4, expected));
    Observer o = {.layers=2, .time=5, .width=8, .vocab=16};
    OK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output, observe, &o));
    CHECK(o.calls == 33 && !o.invalid);
    CHECK(memcmp(baseline, actual, sizeof baseline) == 0);
    CHECK(memcmp(o.last, actual, sizeof actual) == 0);
    const size_t faults[] = {1, 2, 17, 32, 33};
    for (size_t i = 0; i < sizeof faults/sizeof faults[0]; ++i) {
        o.calls=0; o.fail_at=faults[i];
        CHECK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output,
                                         observe, &o) == DL_STATE);
        CHECK(o.calls == o.fail_at);
        /* A failed sink invalidates the result, even after final logits were written. */
        OK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output, NULL, NULL));
        CHECK(memcmp(baseline, actual, sizeof baseline) == 0);
    }
    o.calls=0; ids[4]=16;
    for (size_t i=0; i<16; ++i) actual[i]=77.0f;
    CHECK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output,
                                     observe, &o) == DL_RANGE);
    CHECK(o.calls == 0);
    for (size_t i=0; i<16; ++i) CHECK(actual[i] == 77.0f);
    result=1;
done:
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}

int main(void)
{
    for (int backend=0; backend<2; ++backend)
        for (int tiled=0; tiled<2; ++tiled)
            if (!check((DlBackend)backend, tiled)) return 1;
    puts("trace: four dispatch cases, callback failures and clean retries pass");
    return 0;
}
