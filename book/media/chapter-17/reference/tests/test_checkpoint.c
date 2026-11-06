#define _POSIX_C_SOURCE 200809L
#include "training_fixture.h"

#include <sys/stat.h>
#include <unistd.h>

static const uint64_t budget = UINT64_C(1024) * 1024;

static DlStatus update_once(TrainingFixture *f, DlTrainStats *stats)
{
    return dl_train_update(&f->optim, &f->state, &f->training, &f->batch, &f->tape, stats);
}

static int continuation(const char *path, DlBackend backend, int fixed)
{
    TrainingFixture a = {0}, b = {0};
    int passed = 0;
    OK(fixture_create(&a, backend, fixed));
    OK(fixture_create(&b, backend, fixed));
    DlTrainStats sa = {0}, sb = {0};
    for (size_t i = 0; i < 3; ++i) OK(update_once(&a, &sa));
    OK(dl_checkpoint_save(path, &a.optim, &a.state, &a.training, &a.valid, budget));
    for (size_t i = 0; i < 5; ++i) OK(update_once(&b, &sb));
    double old_loss = 0.0;
    OK(dl_model_forward(&b.model, &b.tape, backend, b.batch.ids, b.batch.targets,
                         b.batch.count, &old_loss));
    uint64_t old_revision = b.model.revision;
    DlParameter *old_directory = b.model.params;
    float *old_table = b.model.params[0].value.data;
    OK(dl_checkpoint_load(path, &b.optim, &b.state, &b.training, &b.valid, budget));
    CHECK(b.model.params == old_directory && b.model.params[0].value.data == old_table);
    CHECK(b.optim.model == &b.model && b.tape.model == &b.model);
    CHECK(b.model.revision == old_revision + 1 && b.optim.revision == b.model.revision);
    CHECK(b.optim.updates == 3 && fixture_equal(&a, &b, 0));
    STATUS(dl_model_backward(&b.model, &b.tape, DL_SET), DL_STATE);
    for (size_t i = 0; i < 3; ++i) {
        OK(update_once(&a, &sa));
        OK(update_once(&b, &sb));
        CHECK(sa.loss == sb.loss && sa.optim.grad_norm == sb.optim.grad_norm);
        CHECK(sa.optim.learning_rate == sb.optim.learning_rate &&
              sa.optim.clip_scale == sb.optim.clip_scale);
        CHECK(memcmp(a.batch.starts, b.batch.starts, a.batch.batch * sizeof(size_t)) == 0);
        CHECK(memcmp(a.batch.ids, b.batch.ids, a.batch.count * sizeof(size_t)) == 0);
        CHECK(memcmp(a.batch.targets, b.batch.targets, a.batch.count * sizeof(size_t)) == 0);
        CHECK(fixture_equal(&a, &b, 1));
    }
    printf("resume %s %s: three next groups bitwise equal; old tape rejected\n",
           backend == DL_BACKEND_REF ? "ref" : "blocked", fixed ? "fixed" : "random");
    passed = 1;
done:
    fixture_destroy(&b);
    fixture_destroy(&a);
    return passed;
}

static int write_bytes(const char *path, const unsigned char *bytes, size_t count, int append)
{
    FILE *file = fopen(path, "wb");
    if (file == NULL) return 0;
    int ok = fwrite(bytes, 1, count, file) == count;
    if (append && fputc(0, file) == EOF) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void patch(unsigned char *bytes, size_t offset, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i)
        bytes[offset + i] = (unsigned char)(value >> (8 * i));
}

static int rejection_checks(const char *good, const char *bad, const char *blocked)
{
    TrainingFixture a = {0}, reference = {0};
    DlByteData original = {0}, after = {0};
    unsigned char *changed = NULL;
    int passed = 0, made_directory = 0;
    OK(fixture_create(&a, DL_BACKEND_BLOCKED, 0));
    OK(fixture_create(&reference, DL_BACKEND_BLOCKED, 0));
    DlTrainStats stats = {0};
    for (size_t i = 0; i < 3; ++i) OK(update_once(&a, &stats));
    OK(dl_checkpoint_save(good, &a.optim, &a.state, &a.training, &a.valid, budget));
    OK(dl_checkpoint_load(good, &a.optim, &a.state, &a.training, &a.valid, budget));
    OK(dl_checkpoint_load(good, &reference.optim, &reference.state,
                          &reference.training, &reference.valid, budget));
    /* Live state advances beyond the file: partial writes cannot hide by
     * re-installing values that were already present in the destination. */
    OK(update_once(&a, &stats));
    OK(update_once(&reference, &stats));
    CHECK(fixture_equal(&a, &reference, 1));
    OK(dl_bytes_load(&original, good, (size_t)budget));
    size_t p = a.optim.count;
    size_t moment_offset = 256 + 88 * a.model.param_count + 4 * p;
    size_t data_offset = moment_offset + 8 * p;
    CHECK(original.count == 256 + 88 * a.model.param_count + 12 * p +
                             a.training.count + a.valid.count + 8);
    CHECK(memcmp(original.bytes, "DLCKPT01", 8) == 0 && original.bytes[8] == 1);
    CHECK(original.bytes[9] == 0 && original.bytes[10] == 0 && original.bytes[11] == 0);
    CHECK(memcmp(original.bytes + original.count - 8, "ENDCKP01", 8) == 0);
    changed = malloc(original.count);
    CHECK(changed != NULL);
    const struct { size_t offset, width; uint64_t value; DlStatus status; } defects[] = {
        {0, 1, 'X', DL_SHAPE},
        {8, 8, 2, DL_SHAPE},
        {32, 8, 10, DL_SHAPE},
        {80, 8, UINT64_MAX, DL_SHAPE},
        {128, 8, 161, DL_SHAPE},
        {232, 8, 161, DL_RANGE},
        {240, 8, 49, DL_RANGE},
        {256, 1, 'X', DL_SHAPE},
        {320, 8, 255, DL_SHAPE},
        {moment_offset - 4, 4, UINT32_C(0x7fc00000), DL_NUMERIC},
        {moment_offset, 4, UINT32_C(0x7f800000), DL_NUMERIC},
        {data_offset - 4, 4, UINT32_C(0xbf800000), DL_NUMERIC},
        {data_offset, 1, 'X', DL_SHAPE},
        {original.count - 1, 1, 'X', DL_SHAPE}
    };
    uint64_t revision = a.model.revision;
    for (size_t i = 0; i < sizeof(defects) / sizeof(defects[0]); ++i) {
        memcpy(changed, original.bytes, original.count);
        patch(changed, defects[i].offset, defects[i].value, defects[i].width);
        CHECK(write_bytes(bad, changed, original.count, 0));
        STATUS(dl_checkpoint_load(bad, &a.optim, &a.state, &a.training, &a.valid, budget),
               defects[i].status);
        CHECK(a.model.revision == revision && fixture_equal(&a, &reference, 1));
    }
    const size_t cuts[] = {0, 12, 255, 344, original.count / 2, original.count - 1};
    for (size_t i = 0; i < sizeof(cuts) / sizeof(cuts[0]); ++i) {
        CHECK(write_bytes(bad, original.bytes, cuts[i], 0));
        STATUS(dl_checkpoint_load(bad, &a.optim, &a.state, &a.training, &a.valid, budget), DL_RANGE);
        CHECK(a.model.revision == revision && fixture_equal(&a, &reference, 1));
    }
    CHECK(write_bytes(bad, original.bytes, original.count, 1));
    STATUS(dl_checkpoint_load(bad, &a.optim, &a.state, &a.training, &a.valid, budget), DL_RANGE);
    STATUS(dl_checkpoint_load(good, &a.optim, &a.state, &a.training, &a.valid, 100), DL_RANGE);
    CHECK(a.model.revision == revision && fixture_equal(&a, &reference, 1));
    a.training.bytes[0] ^= 1; /* Same file length, different actual corpus. */
    DlStatus mismatch = dl_checkpoint_load(good, &a.optim, &a.state, &a.training, &a.valid, budget);
    a.training.bytes[0] ^= 1;
    CHECK(mismatch == DL_SHAPE && fixture_equal(&a, &reference, 1));
    DlTrainState other_schedule = a.state;
    other_schedule.schedule.peak *= 0.5;
    STATUS(dl_checkpoint_load(good, &a.optim, &other_schedule, &a.training, &a.valid, budget),
           DL_SHAPE);
    CHECK(fixture_equal(&a, &reference, 1));
    OK(dl_adamw_abort(&reference.optim));
    OK(dl_adamw_begin(&a.optim));
    STATUS(dl_checkpoint_save(good, &a.optim, &a.state, &a.training, &a.valid, budget), DL_STATE);
    STATUS(dl_checkpoint_load(good, &a.optim, &a.state, &a.training, &a.valid, budget), DL_STATE);
    CHECK(a.optim.active && a.optim.tokens == 0);
    OK(dl_adamw_abort(&a.optim));
    CHECK(fixture_equal(&a, &reference, 1));
    /* Cross the header, then fail the streaming byte budget in the temp file. */
    STATUS(dl_checkpoint_save(good, &a.optim, &a.state, &a.training, &a.valid, 300), DL_RANGE);
    CHECK(mkdir(blocked, 0700) == 0);
    made_directory = 1;
    STATUS(dl_checkpoint_save(blocked, &a.optim, &a.state, &a.training, &a.valid, budget), DL_STATE);
    OK(dl_bytes_load(&after, good, (size_t)budget));
    CHECK(after.count == original.count && memcmp(after.bytes, original.bytes, after.count) == 0);
    CHECK(a.model.revision == revision && fixture_equal(&a, &reference, 1));
    puts("checkpoint: 14 corrupt/incompatible fields, six truncations and trailing bytes rejected");
    puts("checkpoint: dataset/schedule/lifecycle/budget checks; failed saves preserve old file");
    passed = 1;
done:
    if (made_directory) (void)rmdir(blocked);
    free(changed);
    dl_bytes_destroy(&after);
    dl_bytes_destroy(&original);
    fixture_destroy(&reference);
    fixture_destroy(&a);
    return passed;
}

int main(void)
{
    char directory[] = ".dl-checkpoint-XXXXXX";
    if (mkdtemp(directory) == NULL) return EXIT_FAILURE;
    char good[128], bad[128], blocked[128];
    (void)snprintf(good, sizeof(good), "%s/state.bin", directory);
    (void)snprintf(bad, sizeof(bad), "%s/bad.bin", directory);
    (void)snprintf(blocked, sizeof(blocked), "%s/directory", directory);
    int passed = continuation(good, DL_BACKEND_REF, 0) &&
                 continuation(good, DL_BACKEND_REF, 1) &&
                 continuation(good, DL_BACKEND_BLOCKED, 0) &&
                 continuation(good, DL_BACKEND_BLOCKED, 1) &&
                 rejection_checks(good, bad, blocked);
    (void)unlink(bad);
    (void)unlink(good);
    if (rmdir(directory) != 0) passed = 0;
    if (!passed) return EXIT_FAILURE;
    puts("all checkpoint checks passed");
    return EXIT_SUCCESS;
}
