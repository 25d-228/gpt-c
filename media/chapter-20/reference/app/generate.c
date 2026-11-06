#include "dl/checkpoint.h"
#include "dl/cache.h"
#include "dl/sample.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int number(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    *out = value;
    return 1;
}

static DlStatus read_prompt(const char *path, unsigned char *bytes,
                             size_t capacity, size_t *count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) return DL_STATE;
    size_t n = fread(bytes, 1, capacity, file);
    DlStatus status = DL_OK;
    if (ferror(file)) status = DL_STATE;
    else if (n == 0) status = DL_RANGE; /* No implicit beginning-of-text token. */
    else {
        int extra = fgetc(file);
        if (extra != EOF) status = DL_RANGE;
        else if (ferror(file)) status = DL_STATE;
    }
    if (fclose(file) != 0) status = DL_STATE;
    if (status == DL_OK) *count = n;
    return status;
}

static DlStatus write_new_file(const char *path, const unsigned char *bytes,
                                size_t count)
{
    FILE *file = fopen(path, "wbx"); /* C11 exclusive creation; never truncate old output. */
    if (file == NULL) return DL_STATE;
    DlStatus status = fwrite(bytes, 1, count, file) == count ? DL_OK : DL_STATE;
    if (fclose(file) != 0) status = DL_STATE;
    if (status != DL_OK) (void)remove(path); /* Only OUR newly created incomplete file. */
    return status;
}

#define TRY(call) do { status = (call); if (status != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status)); goto done; \
} } while (0)

int main(int argc, char **argv)
{
    if ((argc != 11 && argc != 12) ||
        (argc == 12 && strcmp(argv[11], "full") != 0 && strcmp(argv[11], "cache") != 0 &&
         strcmp(argv[11], "tiled") != 0) ||
        (strcmp(argv[7], "greedy") != 0 && strcmp(argv[7], "sample") != 0) ||
        (strcmp(argv[10], "ref") != 0 && strcmp(argv[10], "blocked") != 0)) {
        fprintf(stderr, "usage: %s CONFIG CHECKPOINT PROMPT OUTPUT CAPACITY NEW "
                        "greedy|sample TEMPERATURE SEED ref|blocked [full|cache|tiled]\n", argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t capacity64 = 0, requested = 0, seed = 0;
    char *end = NULL;
    errno = 0;
    double temperature = strtod(argv[8], &end);
    if (end == argv[8] || *end != '\0' || errno == ERANGE ||
        !isfinite(temperature) || temperature <= 0.0 ||
        !number(argv[5], &capacity64) || capacity64 == 0 || capacity64 > 4096 ||
        !number(argv[6], &requested) || requested > 4096 ||
        !number(argv[9], &seed)) {
        fprintf(stderr, "invalid capacity, count, positive temperature, or seed\n");
        return EXIT_FAILURE;
    }
    int result = EXIT_FAILURE;
    DlStatus status = DL_OK;
    DlModelConfig config = {0};
    DlWeights weights = {0};
    DlInference workspace = {0};
    DlKvCache cache = {0};
    int cached = argc == 12 && strcmp(argv[11], "cache") == 0;
    int tiled = argc == 12 && strcmp(argv[11], "tiled") == 0;
    DlTensor scores = {0};
    unsigned char *bytes = NULL;
    size_t *ids = NULL;
    double *probabilities = NULL;
    size_t config_batch = 0, config_time = 0, prompt_count = 0;
    size_t capacity = (size_t)capacity64, id_bytes = 0, probability_bytes = 0;
    const size_t mib = (size_t)1024 * 1024;
    DlRng rng = {seed};
    DlBackend backend = strcmp(argv[10], "ref") == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED;
    TRY(dl_model_config_load(argv[1], &config, &config_batch, &config_time));
    if (config.vocab != 256) { fprintf(stderr, "byte generation needs vocab=256\n"); goto done; }
    TRY(dl_size_mul(capacity, sizeof(*ids), &id_bytes));
    TRY(dl_size_mul(config.vocab, sizeof(*probabilities), &probability_bytes));
    bytes = malloc(capacity);
    ids = malloc(id_bytes);
    probabilities = malloc(probability_bytes);
    if (bytes == NULL || ids == NULL || probabilities == NULL) goto done;
    TRY(read_prompt(argv[3], bytes, capacity, &prompt_count));
    if (requested > (uint64_t)(capacity - prompt_count)) {
        fprintf(stderr, "prompt + NEW must fit CAPACITY; no truncation or sliding window\n");
        goto done;
    }
    TRY(dl_checkpoint_load_weights(argv[2], &weights, config,
                                     UINT64_C(512) * 1024 * 1024, 128 * mib));
    if (cached) TRY(dl_cache_create(&cache, &weights, capacity, 512 * mib));
    else if (tiled) TRY(dl_inference_create_tiled(&workspace, &weights, capacity, 16, 32, 512 * mib));
    else TRY(dl_inference_create(&workspace, &weights, capacity, 512 * mib));
    const size_t score_shape[] = {1, config.vocab};
    TRY(dl_tensor_create(&scores, 2, score_shape));
    DlMatrix logits = {0};
    TRY(dl_tensor_rows(&scores, &logits));
    for (size_t i = 0; i < prompt_count; ++i) ids[i] = bytes[i];
    size_t length = prompt_count;
    if (cached && requested != 0)
        TRY(dl_cache_prefill(&cache, backend, ids, length, logits));
    for (uint64_t step = 0; step < requested; ++step) {
        /* Row length-1 predicts the token at position length. */
        if (!cached)
            TRY(dl_inference_forward(&workspace, backend, ids, length, length - 1, logits));
        else if (step != 0)
            TRY(dl_cache_decode(&cache, backend, ids[length - 1], logits));
        size_t next = 0;
        if (strcmp(argv[7], "greedy") == 0) TRY(dl_greedy(scores.data, config.vocab, &next));
        else TRY(dl_sample_next(scores.data, config.vocab, temperature, &rng, probabilities, &next));
        ids[length] = next;
        bytes[length] = (unsigned char)next;
        ++length;
    }
    /* Complete computation BEFORE opening output. Stored bytes include prompt. */
    TRY(write_new_file(argv[4], bytes, length));
    printf("prompt=%zu new=%" PRIu64 " total=%zu capacity=%zu config_time=%zu\n",
           prompt_count, requested, length, capacity, config_time);
    printf("payload_bytes weights=%zu workspace=%zu logits=%zu probabilities=%zu\n",
           weights.arena.count * sizeof(float),
           (cached ? cache.arena.count : workspace.arena.count) * sizeof(float),
           scores.count * sizeof(float), probability_bytes);
    printf("inference=%s consumed=%zu kv_payload_bytes=%zu\n",
           cached ? "cache" : (tiled ? "tiled" : "full"), cached ? cache.length : (requested ? length - 1 : 0),
           cache.history.count * sizeof(float));
    printf("tiled_double_payload_bytes=%zu\n", workspace.tiled.count * sizeof(double));
    printf("sampling_rng=%" PRIu64 " mode=%s backend=%s\n", rng.state, argv[7], argv[10]);
    printf("bytes=");
    TRY(dl_bytes_fprint(stdout, bytes, length));
    if (putchar('\n') == EOF || fflush(stdout) != 0) goto done;
    result = EXIT_SUCCESS;
done:
    if (result != EXIT_SUCCESS)
        fprintf(stderr, "generation failed; a completed output can exist if only its display failed\n");
    free(probabilities);
    free(ids);
    free(bytes);
    dl_tensor_destroy(&scores);
    dl_cache_destroy(&cache);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    return result;
}

#undef TRY
