#ifndef DL_TEST_TRAINING_FIXTURE_H
#define DL_TEST_TRAINING_FIXTURE_H

#include "dl/checkpoint.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* All owners stay at this fixture's address until destruction. */
typedef struct {
    DlModel model;
    DlAdamW optim;
    DlTrainTape tape, evaluation;
    DlByteData training, valid;
    DlByteBatch batch, eval_batch;
    DlTrainState state;
} TrainingFixture;

static void fixture_destroy(TrainingFixture *f)
{
    dl_byte_batch_destroy(&f->eval_batch);
    dl_byte_batch_destroy(&f->batch);
    dl_train_tape_destroy(&f->evaluation);
    dl_train_tape_destroy(&f->tape);
    dl_adamw_destroy(&f->optim);
    dl_model_destroy(&f->model);
    dl_bytes_destroy(&f->valid);
    dl_bytes_destroy(&f->training);
}

static DlStatus fixture_text(DlByteData *out, const char *text)
{
    FILE *file = tmpfile();
    if (file == NULL) return DL_STATE;
    size_t n = strlen(text);
    DlStatus status = DL_STATE;
    if (fwrite(text, 1, n, file) == n && fflush(file) == 0 &&
        fseek(file, 0, SEEK_SET) == 0) status = dl_bytes_read(out, file, 4096);
    if (fclose(file) != 0) status = DL_STATE;
    return status;
}

static DlStatus fixture_create(TrainingFixture *f, DlBackend backend, int fixed)
{
    const DlModelConfig config = {.vocab = 256, .layers = 1, .width = 8,
        .query_heads = 2, .kv_heads = 1, .ffn_width = 12, .epsilon = 1e-5f,
        .init_std = 0.125f, .rope_theta = 10000.0, .seed = 42};
    f->state = (DlTrainState){.schedule = {160, 4, 0.01, 0.002}, .rng = {2026},
        .batch = 1, .time = 8, .microbatches = 2, .backend = backend, .fixed = fixed};
#define BUILD(call) do { DlStatus s_ = (call); if (s_ != DL_OK) return s_; } while (0)
    BUILD(fixture_text(&f->training,
        "red fox sat by a tree. blue bird flew over a pond. "
        "the cat saw the fox and the bird. a small boat went down the river.\n"));
    BUILD(fixture_text(&f->valid,
        "the fox saw a boat by the pond. a blue bird sat in the tree.\n"));
    BUILD(dl_model_create(&f->model, config));
    BUILD(dl_adamw_create(&f->optim, &f->model,
        (DlAdamWConfig){0.9, 0.99, 1e-8, 0.01, 1.0}, 1024 * 1024));
    BUILD(dl_adamw_abort(&f->optim));
    BUILD(dl_train_tape_create(&f->tape, &f->model, 1, 8, 1024 * 1024));
    BUILD(dl_train_tape_create(&f->evaluation, &f->model, 1, 8, 1024 * 1024));
    BUILD(dl_byte_batch_create(&f->batch, 1, 8));
    BUILD(dl_byte_batch_create(&f->eval_batch, 1, 8));
#undef BUILD
    return DL_OK;
}

/* Semantic persistent state; revisions are address-lifetime guards, not time.
 * scratch=1 also compares stale gradient/sum bytes and gradient validity. */
static int fixture_equal(const TrainingFixture *a, const TrainingFixture *b, int scratch)
{
    if (a->optim.updates != b->optim.updates || a->optim.tokens != b->optim.tokens ||
        a->optim.active != b->optim.active || a->state.rng.state != b->state.rng.state ||
        a->state.tokens_seen != b->state.tokens_seen || a->optim.count != b->optim.count)
        return 0;
    size_t count = a->optim.count;
    if (memcmp(a->optim.moments, b->optim.moments, 2 * count * sizeof(float)) != 0)
        return 0;
    if (scratch && (a->model.gradients_valid != b->model.gradients_valid ||
        memcmp(a->optim.sum, b->optim.sum, count * sizeof(double)) != 0)) return 0;
    for (size_t p = 0; p < a->model.param_count; ++p) {
        size_t bytes = a->model.params[p].value.count * sizeof(float);
        if (memcmp(a->model.params[p].value.data, b->model.params[p].value.data, bytes) != 0)
            return 0;
        if (scratch && memcmp(a->model.params[p].grad.data,
                              b->model.params[p].grad.data, bytes) != 0) return 0;
    }
    return 1;
}

static int fixture_require(int condition, const char *where)
{
    if (!condition) fprintf(stderr, "FAILED: %s\n", where);
    return condition;
}

static int fixture_status(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "FAILED: %s: %s, expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

#define CHECK(yes) do { if (!fixture_require((yes), #yes)) goto done; } while (0)
#define STATUS(call, want) do { if (!fixture_status((call), (want), #call)) goto done; } while (0)
#define OK(call) STATUS((call), DL_OK)

#endif
