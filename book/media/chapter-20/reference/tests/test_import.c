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

static DlModelConfig fixture_config(void)
{
    return (DlModelConfig){.vocab=16, .layers=2, .width=8, .query_heads=4,
        .kv_heads=2, .ffn_width=12, .epsilon=1e-5f, .init_std=0.125f,
        .rope_theta=10000.0, .seed=42};
}

static float fixture_value(size_t parameter, size_t coordinate)
{
    uint32_t bf = UINT32_C(0x3d00) + (uint32_t)((97 * parameter + coordinate) % 256);
    if ((parameter + coordinate) % 3 == 0) bf |= UINT32_C(0x8000);
    uint32_t bits = bf << 16;
    float value = 0.0f;
    memcpy(&value, &bits, 4);
    return value;
}

static void put_u(unsigned char *p, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i) {
        p[i] = (unsigned char)(value & 255);
        value >>= 8;
    }
}

static void put_f(unsigned char *p, float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, 4);
    put_u(p, bits, 4);
}

static int write_bytes(const char *path, const unsigned char *data, size_t size)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) return 0;
    int result = fwrite(data, 1, size, file) == size;
    if (fclose(file) != 0) result = 0;
    return result;
}

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

int main(int argc, char **argv)
{
    if (argc == 2) return loaded(argv[1]) ? 0 : 1;
    if (argc != 1) return 2;
    return malformed() ? 0 : 1;
}
