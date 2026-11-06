#define _POSIX_C_SOURCE 200809L
#include "dl/checkpoint.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static int positive(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    if (value == 0) return 0;
    *out = value;
    return 1;
}

static double seconds_between(struct timespec begin, struct timespec end)
{
    return (double)(end.tv_sec - begin.tv_sec) +
           1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
}

#define TRY(call) do { status = (call); if (status != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status)); goto done; \
} } while (0)

int main(int argc, char **argv)
{
    if ((argc != 11 && argc != 12) || (strcmp(argv[1], "new") != 0 && strcmp(argv[1], "resume") != 0) ||
        (strcmp(argv[9], "fixed") != 0 && strcmp(argv[9], "random") != 0) ||
        (strcmp(argv[10], "ref") != 0 && strcmp(argv[10], "blocked") != 0) ||
        (argc == 12 && strcmp(argv[11], "full") != 0 && strcmp(argv[11], "tiled") != 0)) {
        fprintf(stderr, "usage: %s new|resume CONFIG TRAIN VALID CHECKPOINT "
                        "TOTAL STOP MICRO fixed|random ref|blocked [full|tiled]\n", argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t total = 0, stop = 0, micro = 0;
    if (!positive(argv[6], &total) || !positive(argv[7], &stop) ||
        !positive(argv[8], &micro) || total > DL_OPT_MAX_COUNT ||
        stop > total || micro > SIZE_MAX) {
        fprintf(stderr, "invalid update or microbatch count\n");
        return EXIT_FAILURE;
    }
    DlStatus status = DL_OK;
    int result = EXIT_FAILURE;
    DlModelConfig config = {0};
    DlModel model = {0};
    DlAdamW optim = {0};
    DlTrainTape tape = {0}, evaluation = {0};
    DlByteData training = {0}, validation = {0};
    DlByteBatch batch = {0}, eval_batch = {0};
    DlTrainState state = {.schedule = {total, total / 10, 0.003, 0.0003},
        .rng = {2026}, .microbatches = (size_t)micro,
        .backend = strcmp(argv[10], "ref") == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED,
        .fixed = strcmp(argv[9], "fixed") == 0};
    if (argc == 12 && strcmp(argv[11], "tiled") == 0) {
        state.attention = DL_ATTENTION_TILED;
        state.query_tile = 16;
        state.key_tile = 32;
    }
    const size_t mib = (size_t)1024 * 1024;
    const uint64_t file_budget = UINT64_C(512) * 1024 * 1024;

    if (strcmp(argv[1], "new") == 0) {
        struct stat metadata;
        if (lstat(argv[5], &metadata) == 0) {
            fprintf(stderr, "new refuses an existing checkpoint; use a new path or resume\n");
            goto done;
        }
        if (errno != ENOENT) { perror("checkpoint path"); goto done; }
    }
    TRY(dl_model_config_load(argv[2], &config, &state.batch, &state.time));
    if (config.vocab != 256) { fprintf(stderr, "byte training needs vocab=256\n"); goto done; }
    TRY(dl_bytes_load(&training, argv[3], 32 * mib));
    TRY(dl_bytes_load(&validation, argv[4], 8 * mib));
    size_t parameters = 0, values = 0, model_bytes = 0;
    TRY(dl_model_config_validate(config, &parameters, &values));
    TRY(dl_size_mul(values, 2 * sizeof(float), &model_bytes));
    if (model_bytes > 128 * mib) { fprintf(stderr, "model exceeds parameter budget\n"); goto done; }
    TRY(dl_model_create(&model, config));
    TRY(dl_adamw_create(&optim, &model, (DlAdamWConfig){0.9, 0.999, 1e-8, 0.01, 1.0},
                        256 * mib));
    TRY(dl_adamw_abort(&optim)); /* Consume constructor-zero gradients as a boundary. */
    if (state.attention == DL_ATTENTION_TILED) {
        TRY(dl_train_tape_create_tiled(&tape, &model, state.batch, state.time,
            state.query_tile, state.key_tile, 512 * mib));
        TRY(dl_train_tape_create_tiled(&evaluation, &model, state.batch, state.time,
            state.query_tile, state.key_tile, 512 * mib));
    } else {
        TRY(dl_train_tape_create(&tape, &model, state.batch, state.time, 512 * mib));
        TRY(dl_train_tape_create(&evaluation, &model, state.batch, state.time, 512 * mib));
    }
    TRY(dl_byte_batch_create(&batch, state.batch, state.time));
    TRY(dl_byte_batch_create(&eval_batch, state.batch, state.time));
    TRY(dl_train_boundary(&optim, &state, &training, &validation));
    if (strcmp(argv[1], "resume") == 0)
        TRY(dl_checkpoint_load(argv[5], &optim, &state, &training, &validation, file_budget));
    if (optim.updates > stop) { fprintf(stderr, "STOP precedes restored update\n"); goto done; }
    printf("parameters=%zu B=%zu T=%zu micro=%zu backend=%s mode=%s\n",
           model.value_count, state.batch, state.time, state.microbatches, argv[10], argv[9]);
    size_t train_bytes = 0, eval_bytes = 0;
    TRY(dl_train_tape_payload_bytes(&tape, &train_bytes));
    TRY(dl_train_tape_payload_bytes(&evaluation, &eval_bytes));
    printf("attention=%s query_tile=%zu key_tile=%zu\n",
           state.attention == DL_ATTENTION_TILED ? "tiled" : "full",
           state.query_tile, state.key_tile);
    printf("payload_bytes model=%zu optimizer=%zu train_tape=%zu eval_tape=%zu data_capacity=%zu\n",
           model_bytes, optim.count * (2 * sizeof(float) + sizeof(double)),
           train_bytes, eval_bytes,
           training.capacity + validation.capacity);
    DlEvalStats evaluation_stats = {0};
    TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch, 4,
                     &evaluation_stats));
    printf("validation before update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
           optim.updates, evaluation_stats.loss, evaluation_stats.tokens, evaluation_stats.unscored);

    while (optim.updates < stop) {
        struct timespec begin = {0}, end = {0};
        if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) goto done;
        DlTrainStats stats = {0};
        TRY(dl_train_update(&optim, &state, &training, &batch, &tape, &stats));
        if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) goto done;
        printf("update=%" PRIu64 " train_loss=%.9f targets=%" PRIu64
               " lr=%.6g norm=%.6g clip=%.6g seconds=%.6f\n",
               optim.updates, stats.loss, stats.optim.tokens, stats.optim.learning_rate,
               stats.optim.grad_norm, stats.optim.clip_scale, seconds_between(begin, end));
        if (fflush(stdout) != 0) goto done;
        if (optim.updates % 10 == 0 || optim.updates == stop) {
            TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch,
                             4, &evaluation_stats));
            printf("validation update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
                   optim.updates, evaluation_stats.loss, evaluation_stats.tokens,
                   evaluation_stats.unscored);
            TRY(dl_checkpoint_save(argv[5], &optim, &state, &training, &validation, file_budget));
        }
    }
    if (fflush(stdout) != 0) goto done;
    result = EXIT_SUCCESS;
done:
    if (result != EXIT_SUCCESS)
        fprintf(stderr, "training stopped; use the last completed checkpoint, if any\n");
    dl_byte_batch_destroy(&eval_batch);
    dl_byte_batch_destroy(&batch);
    dl_train_tape_destroy(&evaluation);
    dl_train_tape_destroy(&tape);
    dl_adamw_destroy(&optim);
    dl_model_destroy(&model);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&training);
    return result;
}

#undef TRY
