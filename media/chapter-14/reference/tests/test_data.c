#include "dl/data.h"
#include "dl/model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int require(int yes, const char *where)
{
    if (!yes) fprintf(stderr, "%s\n", where);
    return yes;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "%s: %s; expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

#define CHECK(yes, where) do { if (!require((yes), (where))) goto done; } while (0)
#define STATUS(call, want) do { if (!status_is((call), (want), #call)) goto done; } while (0)
#define OK(call) STATUS((call), DL_OK)

/* Real binary stdio with no persistent test files or platform-specific paths. */
static DlStatus memory_file(DlByteData *out, const unsigned char *bytes,
                             size_t count, size_t limit)
{
    FILE *stream = tmpfile();
    if (stream == NULL) return DL_STATE;
    DlStatus status = DL_STATE;
    if ((count == 0 || fwrite(bytes, 1, count, stream) == count) &&
        fseek(stream, 0, SEEK_SET) == 0)
        status = dl_bytes_read(out, stream, limit);
    if (fclose(stream) != 0 && status == DL_OK) {
        dl_bytes_destroy(out);
        status = DL_STATE;
    }
    return status;
}

static int file_checks(void)
{
    int ok = 0;
    DlByteData data = {0}, rejected = {0};
    DlByteBatch bad = {0};
    const unsigned char raw[] = {0, 127, 128, 255, 0, 10, 13};
    OK(memory_file(&data, raw, sizeof raw, sizeof raw));
    CHECK(data.count == sizeof raw && data.capacity == sizeof raw &&
          memcmp(data.bytes, raw, sizeof raw) == 0, "all raw bytes preserved");
    unsigned char *owner = data.bytes;
    STATUS(memory_file(&data, raw, sizeof raw, sizeof raw), DL_STATE);
    CHECK(data.bytes == owner && data.count == sizeof raw, "live owner unchanged");
    STATUS(memory_file(&rejected, raw, sizeof raw, sizeof raw - 1), DL_RANGE);
    STATUS(memory_file(&rejected, raw, 0, 8), DL_RANGE);
    STATUS(memory_file(&rejected, raw, 1, 8), DL_RANGE);
    STATUS(dl_bytes_load(&rejected, "", 8), DL_STATE);
    CHECK(rejected.bytes == NULL && rejected.count == 0 && rejected.capacity == 0,
          "failed loads leave empty owner");
    STATUS(dl_byte_batch_create(&bad, 0, 3), DL_ARGUMENT);
    STATUS(dl_byte_batch_create(&bad, SIZE_MAX, 2), DL_OVERFLOW);
    STATUS(dl_byte_batch_create(&bad, 1, SIZE_MAX / 2), DL_OVERFLOW);
    CHECK(bad.storage == NULL && bad.count == 0, "batch overflow unchanged");
    ok = 1;
done:
    dl_byte_batch_destroy(&bad);
    dl_bytes_destroy(&rejected);
    dl_bytes_destroy(&data);
    dl_bytes_destroy(&data);
    return ok;
}

static int rng_checks(void)
{
    int ok = 0;
    DlRng rng = {3};
    uint64_t output = 123;
    STATUS(dl_rng_bounded(&rng, 0, &output), DL_ARGUMENT);
    CHECK(rng.state == 3 && output == 123, "rejected bound preserves outputs");
    /* First word 2092789425003139053 is rejected. The second is accepted. */
    OK(dl_rng_bounded(&rng, (UINT64_C(1) << 63) + 1, &output));
    CHECK(output == UINT64_C(3694763184872335752) &&
          rng.state == UINT64_C(4354685564936845357), "rejection consumes two words");
    DlRng a = {42}, b = {42};
    for (size_t i = 0; i < 100; ++i) {
        uint64_t x, y;
        OK(dl_rng_bounded(&a, 5, &x));
        OK(dl_rng_bounded(&b, 5, &y));
        CHECK(x < 5 && x == y && a.state == b.state, "bounded replay");
    }
    OK(dl_rng_bounded(&a, 1, &output));
    CHECK(output == 0, "single possible start");
    /* Small exhaustive analogue: 16 words -> 6 outcomes; reject words 0..3. */
    unsigned int biased[6] = {0}, unbiased[6] = {0};
    for (unsigned int word = 0; word < 16; ++word) {
        ++biased[word % 6];
        if (word >= 4) ++unbiased[word % 6];
    }
    for (size_t i = 0; i < 6; ++i)
        CHECK(unbiased[i] == 2 && biased[i] == (i < 4 ? 3U : 2U), "modulo trace");
    ok = 1;
done:
    return ok;
}

static int window_checks(void)
{
    int ok = 0, has = 99;
    size_t cursor = 0;
    DlByteData train = {0}, validation = {0}, minimum = {0};
    DlByteBatch batch = {0}, replay = {0}, too_long = {0};
    DlRng rng = {42};
    const unsigned char text[] = "ABCDEFGH", held[] = "wxyz0123";
    OK(memory_file(&train, text, sizeof text - 1, 32));
    OK(memory_file(&validation, held, sizeof held - 1, 32));
    OK(memory_file(&minimum, text, 4, 8));
    OK(dl_byte_batch_create(&batch, 2, 3));
    OK(dl_byte_batch_create(&replay, 2, 3));
    OK(dl_byte_batch_create(&too_long, 1, 8));
    OK(dl_byte_batch_next(&train, &batch, &cursor, &has));
    CHECK(has == 1 && cursor == 6 && batch.starts[0] == 0 && batch.starts[1] == 3,
          "consecutive offsets");
    for (size_t i = 0; i < 6; ++i)
        CHECK(batch.ids[i] == text[i] && batch.targets[i] == text[i + 1],
              "explicit one-byte target shift");
    printf("hand windows: ABC -> BCD; DEF -> EFG; unscored targets=1\n");
    size_t saved[14];
    memcpy(saved, batch.storage, sizeof saved);
    OK(dl_byte_batch_next(&train, &batch, &cursor, &has));
    CHECK(has == 0 && cursor == 6 && memcmp(saved, batch.storage, sizeof saved) == 0,
          "end does not wrap or pad");
    cursor = SIZE_MAX; has = 99;
    STATUS(dl_byte_batch_next(&train, &batch, &cursor, &has), DL_RANGE);
    CHECK(cursor == SIZE_MAX && has == 99, "bad cursor unchanged");
    STATUS(dl_byte_batch_sample(&train, &too_long, &rng), DL_RANGE);
    CHECK(rng.state == 42, "short-for-time failure consumes no RNG");
    OK(dl_byte_batch_sample(&train, &batch, &rng));
    CHECK(batch.starts[0] == 3 && batch.starts[1] == 1, "known seed starts");
    for (size_t row = 0; row < 2; ++row)
        for (size_t t = 0; t < 3; ++t)
            CHECK(batch.ids[row * 3 + t] == text[batch.starts[row] + t] &&
                  batch.targets[row * 3 + t] == text[batch.starts[row] + t + 1],
                  "sample alignment");
    DlRng resume = rng;
    OK(dl_byte_batch_sample(&train, &batch, &rng));
    OK(dl_byte_batch_sample(&train, &replay, &resume));
    CHECK(rng.state == resume.state &&
          memcmp(batch.storage, replay.storage, sizeof saved) == 0, "next-batch replay");
    uint64_t train_state = rng.state;
    cursor = 0;
    OK(dl_byte_batch_next(&validation, &replay, &cursor, &has));
    for (size_t i = 0; i < 6; ++i)
        CHECK(replay.ids[i] == held[i] && replay.targets[i] == held[i + 1],
              "validation uses only its own bytes");
    CHECK(rng.state == train_state, "evaluation cannot consume training RNG");
    OK(dl_byte_batch_sample(&minimum, &batch, &rng));
    CHECK(batch.starts[0] == 0 && batch.starts[1] == 0, "N=T+1 is one valid start");
    cursor = minimum.count - 1;
    OK(dl_byte_batch_next(&minimum, &batch, &cursor, &has));
    CHECK(has == 0 && cursor == 3, "cursor at final byte is exhausted");
    ok = 1;
done:
    dl_byte_batch_destroy(&too_long);
    dl_byte_batch_destroy(&replay);
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&minimum);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&train);
    return ok;
}

static int presentation_checks(void)
{
    int ok = 0;
    FILE *stream = tmpfile();
    CHECK(stream != NULL, "temporary display stream");
    const unsigned char raw[] = {65, 0, 27, 128, 255, 34, 92, 10, 13, 9};
    const char expected[] = "\"A\\x00\\x1B\\x80\\xFF\\\"\\\\\\n\\r\\t\"";
    char actual[128] = {0};
    OK(dl_bytes_fprint(stream, raw, sizeof raw));
    CHECK(fseek(stream, 0, SEEK_SET) == 0, "rewind display");
    size_t count = fread(actual, 1, sizeof actual, stream);
    CHECK(!ferror(stream) && count == strlen(expected) &&
          memcmp(actual, expected, count) == 0, "escaped display has no raw controls");
    ok = 1;
done:
    if (stream != NULL && fclose(stream) != 0) ok = 0;
    return ok;
}

static int model_checks(void)
{
    int ok = 0;
    DlByteData data = {0};
    DlByteBatch batch = {0};
    DlModel model = {0};
    DlTrainTape tape = {0};
    DlModelConfig config = {256, 1, 8, 2, 1, 12, 1e-5f, 0.1f, 10000.0, 42};
    const unsigned char raw[] = {65, 0, 128, 255, 66, 10, 67, 68};
    OK(memory_file(&data, raw, sizeof raw, 16));
    OK(dl_byte_batch_create(&batch, 2, 3));
    OK(dl_model_create(&model, config));
    OK(dl_train_tape_create(&tape, &model, 2, 3, 1024 * 1024));
    double first_loss = 0.0;
    for (int backend = 0; backend < 2; ++backend) {
        size_t cursor = 0;
        int has = 0;
        OK(dl_byte_batch_next(&data, &batch, &cursor, &has));
        CHECK(has, "model has a batch");
        double loss = -1.0;
        OK(dl_model_forward(&model, &tape,
                            backend == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED,
                            batch.ids, batch.targets, batch.count, &loss));
        CHECK(isfinite(loss) && loss > 0.0, "finite byte-vocabulary loss");
        if (backend == 0) first_loss = loss;
        else CHECK(fabs(first_loss - loss) <= 1e-6, "model backend loss agreement");
        /* Reuse the caller's storage: forward has already copied the metadata. */
        DlRng rng = {9};
        OK(dl_byte_batch_sample(&data, &batch, &rng));
        for (size_t i = 0; i < 6; ++i)
            CHECK(tape.ids[i] == raw[i] && tape.targets[i] == raw[i + 1],
                  "model owns copied IDs and does not shift targets again");
        OK(dl_model_backward(&model, &tape, DL_SET));
        CHECK(model.gradients_valid, "complete byte-model backward");
        for (size_t p = 0; p < model.param_count; ++p)
            for (size_t i = 0; i < model.params[p].grad.count; ++i)
                CHECK(isfinite(model.params[p].grad.data[i]), "finite byte-model gradients");
    }
    printf("byte model: V=256, targets=6, loss=%.9f (both backends)\n", first_loss);
    ok = 1;
done:
    dl_train_tape_destroy(&tape);
    dl_model_destroy(&model);
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&data);
    return ok;
}

int main(void)
{
    int ok = 1;
    ok &= file_checks();
    ok &= rng_checks();
    ok &= window_checks();
    ok &= presentation_checks();
    ok &= model_checks();
    if (!ok) return EXIT_FAILURE;
    puts("data checks passed");
    return EXIT_SUCCESS;
}
