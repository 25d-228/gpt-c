#define _POSIX_C_SOURCE 200809L
#include "dl/inference.h"
#include "dl/sample.h"
#include "training_fixture.h"

#include <math.h>
#include <unistd.h>

static int near(float a, float b)
{
    return isfinite(a) && isfinite(b) &&
           fabs((double)a - (double)b) <= 2e-6 + 2e-5 * fabs((double)b);
}

static int parity(size_t kv_heads, size_t layers, DlBackend backend)
{
    int result = 0;
    DlModel model = {0};
    DlTrainTape tape = {0};
    DlWeights weights = {0};
    DlInference workspace = {0};
    const DlModelConfig config = {.vocab = 16, .layers = layers, .width = 8,
        .query_heads = 4, .kv_heads = kv_heads, .ffn_width = 12,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    enum { CAPACITY = 17, VOCAB = 16, STRIDE = 19, STORAGE = CAPACITY * STRIDE + 4 };
    float full[STORAGE], selected[2 * STRIDE + 4], last[VOCAB];
    size_t ids[CAPACITY], targets[CAPACITY];
    for (size_t i = 0; i < CAPACITY; ++i) {
        ids[i] = (3 * i + 1) % VOCAB;
        targets[i] = (5 * i + 2) % VOCAB;
    }
    OK(dl_model_create(&model, config));
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    CHECK(weights.ready && weights.arena.count == model.value_count);
    OK(dl_inference_create(&workspace, &weights, CAPACITY, 1024 * 1024));
    const size_t kv = kv_heads * 2;
    const size_t expected = 7 * CAPACITY * 8 + 3 * CAPACITY * kv +
                            3 * CAPACITY * 12 + 2 * 4 * CAPACITY * CAPACITY;
    CHECK(workspace.arena.count == expected); /* No layers factor. */
    const size_t lengths[] = {1, 17, 3, 5}; /* Grow, then shrink/remap dirty arena. */
    for (size_t case_index = 0; case_index < 4; ++case_index) {
        size_t n = lengths[case_index];
        OK(dl_train_tape_create(&tape, &model, 1, n, 1024 * 1024));
        OK(dl_model_zero_grad(&model));
        double loss = 0.0;
        OK(dl_model_forward(&model, &tape, backend, ids, targets, n, &loss));
        uint64_t revision = model.revision;
        for (size_t i = 0; i < workspace.arena.count; ++i) workspace.arena.data[i] = NAN;
        for (size_t i = 0; i < STORAGE; ++i) full[i] = 88.0f;
        DlMatrix all = {full, STORAGE, 2, n, VOCAB, STRIDE};
        OK(dl_inference_forward(&workspace, backend, ids, n, 0, all));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < VOCAB; ++j)
                CHECK(near(full[2 + i * STRIDE + j],
                    tape.logits.base[tape.logits.offset + i * tape.logits.ld + j]));
        for (size_t i = 0; i < STORAGE; ++i) {
            int logical = i >= 2 && (i - 2) / STRIDE < n && (i - 2) % STRIDE < VOCAB;
            if (!logical) CHECK(full[i] == 88.0f);
        }
        DlMatrix one = {last, VOCAB, 0, 1, VOCAB, VOCAB};
        OK(dl_inference_forward(&workspace, backend, ids, n, n - 1, one));
        for (size_t j = 0; j < VOCAB; ++j) CHECK(near(last[j], full[2 + (n - 1) * STRIDE + j]));
        size_t first = n / 2, rows = n - first < 2 ? n - first : 2;
        for (size_t i = 0; i < 2 * STRIDE + 4; ++i) selected[i] = 88.0f;
        DlMatrix middle = {selected, 2 * STRIDE + 4, 2, rows, VOCAB, STRIDE};
        OK(dl_inference_forward(&workspace, backend, ids, n, first, middle));
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < VOCAB; ++j)
                CHECK(near(selected[2 + i * STRIDE + j], full[2 + (first + i) * STRIDE + j]));
        for (size_t i = 0; i < 2 * STRIDE + 4; ++i) {
            int logical = i >= 2 && (i - 2) / STRIDE < rows && (i - 2) % STRIDE < VOCAB;
            if (!logical) CHECK(selected[i] == 88.0f);
        }
        if (n > 1) {
            OK(dl_inference_forward(&workspace, backend, ids, n - 1, n - 2, one));
            for (size_t j = 0; j < VOCAB; ++j) CHECK(near(last[j], full[2 + (n - 2) * STRIDE + j]));
        }
        CHECK(tape.ready && model.revision == revision && model.gradients_valid);
        for (size_t p = 0; p < model.param_count; ++p)
            for (size_t i = 0; i < model.params[p].grad.count; ++i)
                CHECK(model.params[p].grad.data[i] == 0.0f);
        OK(dl_model_backward(&model, &tape, DL_SET)); /* Inference did not consume the tape. */
        dl_train_tape_destroy(&tape);
    }
    result = 1;
done:
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    dl_train_tape_destroy(&tape);
    dl_model_destroy(&model);
    return result;
}

static int failures(void)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0}, rejected = {0};
    DlInference workspace = {0}, other = {0};
    const DlModelConfig config = {.vocab = 16, .layers = 1, .width = 8,
        .query_heads = 4, .kv_heads = 2, .ffn_width = 12,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    size_t ids[] = {0, 2, 3, 1, 4};
    float logits[16];
    for (size_t i = 0; i < 16; ++i) logits[i] = 77.0f;
    DlMatrix out = {logits, 16, 0, 1, 16, 16};
    STATUS(dl_weights_create(&rejected, config, 1), DL_RANGE);
    CHECK(rejected.params == NULL && rejected.arena.data == NULL && !rejected.ready);
    OK(dl_weights_create(&rejected, config, 1024 * 1024));
    STATUS(dl_inference_create(&other, &rejected, 4, 1024 * 1024), DL_STATE);
    dl_weights_destroy(&rejected);
    OK(dl_model_create(&model, config));
    float saved = model.params[0].value.data[0];
    model.params[0].value.data[0] = NAN; /* Isolated corrupt-source fixture. */
    STATUS(dl_weights_from_model(&rejected, &model, 1024 * 1024), DL_NUMERIC);
    CHECK(rejected.params == NULL && rejected.arena.data == NULL);
    model.params[0].value.data[0] = saved;
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    STATUS(dl_weights_from_model(&weights, &model, 1024 * 1024), DL_STATE);
    STATUS(dl_inference_create(&other, &weights, 4, 1), DL_RANGE);
    CHECK(other.weights == NULL && other.arena.data == NULL);
    STATUS(dl_inference_create(&other, &weights, 0, 1024 * 1024), DL_ARGUMENT);
    CHECK(dl_inference_create(&other, &weights, SIZE_MAX, 1024 * 1024) != DL_OK);
    OK(dl_inference_create(&workspace, &weights, 4, 1024 * 1024));
    STATUS(dl_inference_create(&workspace, &weights, 4, 1024 * 1024), DL_STATE);
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 0, 0, out), DL_RANGE);
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 5, 0, out), DL_RANGE);
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 4, out), DL_RANGE);
    STATUS(dl_inference_forward(&workspace, (DlBackend)99, ids, 4, 3, out), DL_ARGUMENT);
    DlMatrix bad_shape = out;
    bad_shape.cols = 15;
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, bad_shape), DL_SHAPE);
    ids[2] = 16;
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, out), DL_RANGE);
    ids[2] = 3;
    DlMatrix alias = {workspace.arena.data, workspace.arena.count, 0, 1, 16, 16};
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, alias), DL_ALIAS);
    alias.base = weights.arena.data;
    alias.count = weights.arena.count;
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, alias), DL_ALIAS);
    for (size_t i = 0; i < 16; ++i) CHECK(logits[i] == 77.0f);
    /* Deliberate corruption only to exercise failure recovery; not a supported edit API. */
    saved = weights.arena.data[0];
    weights.arena.data[0] = NAN;
    STATUS(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, out), DL_NUMERIC);
    weights.arena.data[0] = saved;
    OK(dl_inference_forward(&workspace, DL_BACKEND_REF, ids, 4, 3, out));
    /* Source training model need not survive the copied weights. */
    dl_model_destroy(&model);
    OK(dl_inference_forward(&workspace, DL_BACKEND_BLOCKED, ids, 1, 0, out));
    puts("inference: bounds, aliases, budgets, unready owners, copy independence and failure recovery");
    result = 1;
done:
    dl_inference_destroy(&other);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&rejected);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}

static int write_bytes(const char *path, const unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) return 0;
    int ok = fwrite(bytes, 1, count, file) == count;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int rejects_file(const char *path, DlModelConfig config)
{
    DlWeights out = {0};
    DlStatus status = dl_checkpoint_load_weights(path, &out, config, 1024 * 1024, 1024 * 1024);
    int ok = status != DL_OK && out.params == NULL && out.arena.data == NULL && !out.ready;
    if (!ok) fprintf(stderr, "FAILED: malformed weight import returned %s or changed output\n", dl_status_name(status));
    dl_weights_destroy(&out);
    return ok;
}

static int weight_loading(void)
{
    int result = 0;
    TrainingFixture f = {0}, baseline = {0};
    DlWeights weights = {0}, rejected = {0};
    DlInference workspace = {0};
    unsigned char *bytes = NULL;
    FILE *file = NULL;
    char path[] = "/tmp/dl-inference-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    if (close(fd) != 0) { (void)unlink(path); return 0; }
    OK(fixture_create(&f, DL_BACKEND_BLOCKED, 0));
    OK(fixture_create(&baseline, DL_BACKEND_BLOCKED, 0));
    DlTrainStats stats = {0};
    OK(dl_train_update(&f.optim, &f.state, &f.training, &f.batch, &f.tape, &stats));
    OK(dl_train_update(&baseline.optim, &baseline.state, &baseline.training,
                        &baseline.batch, &baseline.tape, &stats));
    OK(dl_checkpoint_save(path, &f.optim, &f.state, &f.training, &f.valid, 1024 * 1024));
    OK(dl_checkpoint_load_weights(path, &weights, f.model.config, 1024 * 1024, 1024 * 1024));
    CHECK(weights.ready && weights.arena.count == f.model.value_count);
    for (size_t p = 0; p < f.model.param_count; ++p) {
        DlMatrix v = weights.params[p].value;
        CHECK(memcmp(v.base + v.offset, f.model.params[p].value.data,
                     v.rows * v.cols * sizeof(float)) == 0);
    }
    CHECK(fixture_equal(&f, &baseline, 1)); /* Neither source model nor trainer RNG changed. */
    STATUS(dl_checkpoint_load_weights(path, &weights, f.model.config, 1024 * 1024, 1024 * 1024), DL_STATE);
    OK(dl_inference_create(&workspace, &weights, 8, 1024 * 1024));
    float last[256];
    DlMatrix output = {last, 256, 0, 1, 256, 256};
    double loss = 0.0;
    OK(dl_model_forward(&f.model, &f.tape, DL_BACKEND_BLOCKED,
                         f.batch.ids, f.batch.targets, 8, &loss));
    OK(dl_inference_forward(&workspace, DL_BACKEND_BLOCKED, f.batch.ids, 8, 7, output));
    for (size_t j = 0; j < 256; ++j)
        CHECK(near(last[j], f.tape.logits.base[f.tape.logits.offset + 7 * 256 + j]));
    STATUS(dl_checkpoint_load_weights(path, &rejected, f.model.config, 1, 1024 * 1024), DL_RANGE);
    STATUS(dl_checkpoint_load_weights(path, &rejected, f.model.config, 1024 * 1024, 1), DL_RANGE);
    DlModelConfig wrong = f.model.config;
    ++wrong.seed;
    STATUS(dl_checkpoint_load_weights(path, &rejected, wrong, 1024 * 1024, 1024 * 1024), DL_SHAPE);
    file = fopen(path, "rb");
    CHECK(file != NULL && fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    CHECK(length > 0 && length < 1024 * 1024 && fseek(file, 0, SEEK_SET) == 0);
    size_t count = (size_t)length;
    bytes = malloc(count + 1);
    CHECK(bytes != NULL && fread(bytes, 1, count, file) == count);
    int closed = fclose(file);
    file = NULL;
    CHECK(closed == 0);
    const size_t changed[] = {0, 8, 16, 72, 80, 88, 96, 104, 112, 120,
                              160, 168, 240, 256, 320, 328, 336, count - 1};
    for (size_t i = 0; i < sizeof(changed) / sizeof(changed[0]); ++i) {
        unsigned char old = bytes[changed[i]];
        bytes[changed[i]] ^= 0x80;
        CHECK(write_bytes(path, bytes, count) && rejects_file(path, f.model.config));
        bytes[changed[i]] = old;
    }
    size_t moments = 256 + weights.param_count * 88 + weights.arena.count * 4;
    const size_t bad_float_offsets[] = {344, moments, moments + weights.arena.count * 4};
    for (size_t i = 0; i < 3; ++i) {
        size_t at = bad_float_offsets[i];
        unsigned char old[4];
        memcpy(old, bytes + at, 4);
        /* qNaN weights/first moment; negative second moment. */
        bytes[at] = 0; bytes[at + 1] = 0;
        bytes[at + 2] = i == 2 ? 0x80 : 0xc0;
        bytes[at + 3] = i == 2 ? 0xbf : 0x7f;
        CHECK(write_bytes(path, bytes, count) && rejects_file(path, f.model.config));
        memcpy(bytes + at, old, 4);
    }
    const size_t truncated[] = {0, 7, 255, 343, count - 9, count - 1};
    for (size_t i = 0; i < 6; ++i)
        CHECK(write_bytes(path, bytes, truncated[i]) && rejects_file(path, f.model.config));
    bytes[count] = 0;
    CHECK(write_bytes(path, bytes, count + 1) && rejects_file(path, f.model.config));
    CHECK(write_bytes(path, bytes, count));
    STATUS(dl_checkpoint_load_weights(path, &rejected, f.model.config,
                                      (uint64_t)count - 1, 1024 * 1024), DL_RANGE);
    /* Last attempt accepts the exact byte budget; source files are not arguments. */
    OK(dl_checkpoint_load_weights(path, &rejected, f.model.config, (uint64_t)count, 1024 * 1024));
    CHECK(rejected.ready);
    CHECK(fixture_equal(&f, &baseline, 1));
    puts("weights: trained checkpoint logits agree; 21 damaged fields, six truncations and trailing bytes rejected");
    result = 1;
done:
    if (file != NULL) (void)fclose(file);
    free(bytes);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&rejected);
    dl_weights_destroy(&weights);
    fixture_destroy(&baseline);
    fixture_destroy(&f);
    (void)unlink(path);
    return result;
}

static int generation_choices(void)
{
    int result = 0;
    DlModel model = {0};
    DlTrainTape tape = {0};
    DlWeights weights = {0};
    DlInference workspace = {0};
    const DlModelConfig config = {.vocab = 16, .layers = 2, .width = 8,
        .query_heads = 4, .kv_heads = 2, .ffn_width = 12,
        .epsilon = 1e-5f, .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    OK(dl_model_create(&model, config));
    OK(dl_weights_from_model(&weights, &model, 1024 * 1024));
    OK(dl_inference_create(&workspace, &weights, 12, 1024 * 1024));
    for (int backend = 0; backend < 2; ++backend) {
        for (int sampled = 0; sampled < 2; ++sampled) {
            size_t ids[12] = {2, 3, 4}, targets[12] = {0};
            float logits[16];
            double probabilities[16];
            DlRng rng = {2026}, reference_rng = {2026};
            for (size_t step = 0; step < 8; ++step) {
                size_t length = 3 + step;
                OK(dl_train_tape_create(&tape, &model, 1, length, 1024 * 1024));
                double loss = 0.0;
                OK(dl_model_forward(&model, &tape, (DlBackend)backend, ids, targets, length, &loss));
                DlMatrix output = {logits, 16, 0, 1, 16, 16};
                OK(dl_inference_forward(&workspace, (DlBackend)backend, ids, length, length - 1, output));
                const float *reference = tape.logits.base + tape.logits.offset + (length - 1) * 16;
                size_t got = 99, want = 99;
                if (sampled) {
                    OK(dl_sample_next(logits, 16, 0.8, &rng, probabilities, &got));
                    OK(dl_sample_next(reference, 16, 0.8, &reference_rng, probabilities, &want));
                } else {
                    OK(dl_greedy(logits, 16, &got));
                    OK(dl_greedy(reference, 16, &want));
                }
                CHECK(got == want && rng.state == reference_rng.state);
                ids[length] = got;
                dl_train_tape_destroy(&tape);
            }
        }
    }
    puts("generation: 32 successive token choices match the training-forward baseline");
    result = 1;
done:
    dl_train_tape_destroy(&tape);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}

int main(void)
{
    const size_t heads[] = {1, 2, 4}, layers[] = {1, 2, 2};
    for (size_t i = 0; i < 3; ++i)
        for (int backend = 0; backend < 2; ++backend)
            if (!parity(heads[i], layers[i], (DlBackend)backend)) return EXIT_FAILURE;
    puts("inference: 24 training/full/subset/last-row cases across both backends and three head-sharing ratios");
    if (!failures() || !weight_loading() || !generation_choices()) return EXIT_FAILURE;
    puts("all inference checks passed");
    return EXIT_SUCCESS;
}
