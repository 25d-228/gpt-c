# Chapter 15 — Checking continuation and failure recovery

This is Part 4 of [Chapter 15](https://25d-228.github.io/gpt-c/chapter-15.html), not a separate implementation-hour allocation. It completes the tests referenced by Part 3's Makefile. The test programs exercise the same C forward, backward, optimizer, data, and file code as the application. Python only launches processes and compares their outputs and files.

A falling loss is necessary for the fixed-batch experiment, but it does not establish correct restart behavior. A checkpoint can contain perfectly finite weights while referring to the wrong next samples. Conversely, a load test can report the expected failure status after already overwriting half of the live model. We need observable checks at each state boundary, not only a successful return code.

## Share a small fixture without sharing its ownership

Create `tests/training_fixture.h`. This private test header defines a fixture, meaning a reproducible collection of objects used by a test. It is not a new library abstraction. Each executable owns its own zero-initialized fixture objects on the stack. `static` helper functions have internal linkage, so including this header in two executables does not introduce a shared global owner.

Start with the includes and owner layout:

```c title="tests/training_fixture.h"
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
```

The optimizer and both tapes borrow `model` at its final address. We never return this whole structure by value or assign it after construction. The two byte datasets and two batches are separate owners; the evaluation tape is not an alias of the training tape.

Destruction releases borrowers before the model. `fixture_text` sends a small literal through a real binary stream and the actual bounded loader. `tmpfile` returns a stream for a temporary file that is removed when closed; the helper still checks writing, flushing, positioning, and closure. Failure to close is a test failure, not permission to claim a successful load.

```c title="tests/training_fixture.h"
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
```

The fixture uses one block with width 8, two query heads, one key/value head, and a feed-forward width of 12. This keeps repeated complete-model tests small while retaining head sharing, tied weights, both residual paths, and every backward operation. Model initialization and data sampling have different seeds and state objects.

`BUILD` is a local convenience macro: return the first failure immediately, leaving the caller to destroy whatever was constructed. Its `do ... while (0)` wrapper makes an invocation behave syntactically as one statement. The caller always reaches its cleanup label. Creating an optimizer does not consume the model's initial zero gradients; the explicit `abort` establishes the inactive, no-fresh-gradient boundary required by the trainer.

```c title="tests/training_fixture.h"
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
```

The next helper compares semantic continuation state. Parameter values and moment arrays must match bit for bit **within the same build and backend**. A freshly restored owner has a different local model revision from an uninterrupted owner; those revisions must each be coherent with their borrowers, but need not equal one another. They are lifetime guards, not optimizer update counts.

With `scratch=1`, the comparison also includes stale parameter-gradient bytes, the optimizer's double gradient-sum array, and gradient validity. Those bytes are deliberately not checkpoint state, but comparing them after the same next update is useful. We do not `memcmp` an entire C structure: padding bytes, addresses, and local lifetime counters are not portable numerical state. Schedule and geometry are established identically by fixture construction; dedicated compatibility tests exercise mismatches.

```c title="tests/training_fixture.h"
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
```

Complete the header with diagnostics. `#yes` and `#call` stringify the macro argument, so a failing check prints the expression that failed. Every test using these macros has a `done:` label and a `passed` flag initialized to zero. A failure jumps to cleanup and cannot fall through as a success.

```c title="tests/training_fixture.h"
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
```

## Check the schedule against numbers, not against itself

Create `tests/test_training.c`. The first test checks the six rates derived in Part 1, not another implementation of the schedule formula. It also checks that a completed schedule and an invalid warmup leave the caller's output unchanged. The no-warmup and single-update cases exercise branches where an off-by-one error can otherwise produce a division by zero or miss the peak rate.

```c title="tests/test_training.c"
#include "training_fixture.h"

static int schedule_checks(void)
{
    int passed = 0;
    const DlSchedule schedule = {6, 2, 0.01, 0.002};
    const double expected[] = {0.005, 0.01, 0.008, 0.006, 0.004, 0.002};
    double rate = 123.0;
    for (uint64_t i = 0; i < 6; ++i) {
        OK(dl_schedule_rate(schedule, i, &rate));
        CHECK(fabs(rate - expected[i]) < 1e-15);
    }
    rate = 123.0;
    STATUS(dl_schedule_rate(schedule, 6, &rate), DL_RANGE);
    CHECK(rate == 123.0);
    STATUS(dl_schedule_rate((DlSchedule){6, 6, 0.01, 0.002}, 0, &rate), DL_ARGUMENT);
    CHECK(rate == 123.0);
    OK(dl_schedule_rate((DlSchedule){3, 0, 0.01, 0.002}, 0, &rate));
    CHECK(rate == 0.01);
    OK(dl_schedule_rate((DlSchedule){3, 0, 0.01, 0.002}, 2, &rate));
    CHECK(rate == 0.002);
    OK(dl_schedule_rate((DlSchedule){1, 0, 0.01, 0.002}, 0, &rate));
    CHECK(rate == 0.01);
    passed = 1;
done:
    return passed;
}
```

The tolerance $10^{-15}$ applies to these small double schedule values, not to float model gradients. Do not copy it into unrelated numerical comparisons without considering the scale and operation count.

## Learn one fixed batch through the entire graph

The overfit test evaluates the first eight targets, executes 120 complete updates, and evaluates those same eight targets again. Each update accumulates two identical eight-target microbatches. This repetition tests the real accumulation path; it does not create sixteen distinct examples or a sixteen-token attention context.

The pass condition is intentionally loose: final loss must be less than a quarter of initial loss. An exact loss decimal would be a brittle cross-platform test. The RNG must remain unchanged in fixed mode, and 120 updates must account for $120\times2\times8=1920$ processed targets.

```c title="tests/test_training.c"
static int overfit(DlBackend backend)
{
    TrainingFixture f = {0};
    int passed = 0;
    OK(fixture_create(&f, backend, 1));
    DlEvalStats before = {0}, after = {0};
    OK(dl_evaluate(&f.model, &f.evaluation, backend, &f.training, &f.eval_batch, 1, &before));
    for (size_t i = 0; i < 120; ++i) {
        DlTrainStats stats = {0};
        OK(dl_train_update(&f.optim, &f.state, &f.training, &f.batch, &f.tape, &stats));
        CHECK(isfinite(stats.loss) && stats.optim.update == i + 1 && stats.optim.tokens == 16);
    }
    OK(dl_evaluate(&f.model, &f.evaluation, backend, &f.training, &f.eval_batch, 1, &after));
    CHECK(after.loss < 0.25 * before.loss);
    CHECK(f.state.rng.state == 2026 && f.state.tokens_seen == 1920);
    CHECK(f.optim.updates == 120 && !f.optim.active && !f.model.gradients_valid);
    printf("overfit %s: %.9f -> %.9f on %llu fixed targets after 120 updates\n",
           backend == DL_BACKEND_REF ? "ref" : "blocked", before.loss, after.loss,
           (unsigned long long)after.tokens);
    passed = 1;
done:
    fixture_destroy(&f);
    return passed;
}
```

The observed Linux results were 5.692190972 to 0.078423635 nats/target for both backends. A low loss here shows that the integrated derivatives and update path can memorize a small fixed example. It does not measure performance on unseen text. The separate-file pilot in Part 3 serves a different purpose.

## Evaluate while training work is genuinely pending

An isolation test that evaluates only after a completed update could miss an implementation that clears pending gradients. Instead, construct two identical fixtures. In each, begin a group, accumulate one microbatch, and produce a second forward tape without yet differentiating it. Evaluate **only one fixture**, using its separate evaluation owners. Then differentiate the pending forward tape and finish both groups.

This tests three live dependencies simultaneously: the saved training activations, the partial optimizer sum, and the data RNG. The final equality checks include all parameter and moment bytes, not just similar printed losses.

```c title="tests/test_training.c"
static int evaluation_isolation(void)
{
    TrainingFixture a = {0}, b = {0};
    int passed = 0;
    OK(fixture_create(&a, DL_BACKEND_BLOCKED, 0));
    OK(fixture_create(&b, DL_BACKEND_BLOCKED, 0));
    TrainingFixture *fixtures[] = {&a, &b};
    for (size_t i = 0; i < 2; ++i) {
        TrainingFixture *f = fixtures[i];
        OK(dl_adamw_begin(&f->optim));
        OK(dl_byte_batch_sample(&f->training, &f->batch, &f->state.rng));
        double loss = 0.0;
        OK(dl_model_forward(&f->model, &f->tape, f->state.backend,
                             f->batch.ids, f->batch.targets, f->batch.count, &loss));
        OK(dl_model_backward(&f->model, &f->tape, DL_SET));
        OK(dl_adamw_add_mean(&f->optim, 8));
        OK(dl_model_forward(&f->model, &f->tape, f->state.backend,
                             f->batch.ids, f->batch.targets, f->batch.count, &loss));
    }
    CHECK(fixture_equal(&a, &b, 1));
    DlEvalStats evaluated = {0};
    OK(dl_evaluate(&a.model, &a.evaluation, a.state.backend, &a.valid, &a.eval_batch, 0,
                     &evaluated));
    CHECK(evaluated.tokens == ((a.valid.count - 1) / 8) * 8);
    CHECK(evaluated.unscored == a.valid.count - 1 - (size_t)evaluated.tokens);
    CHECK(a.tape.ready && b.tape.ready && fixture_equal(&a, &b, 1));
    for (size_t i = 0; i < 2; ++i) {
        TrainingFixture *f = fixtures[i];
        OK(dl_model_backward(&f->model, &f->tape, DL_SET));
        OK(dl_adamw_add_mean(&f->optim, 8));
        DlOptimStats stats = {0};
        OK(dl_adamw_step(&f->optim, 0.01, &stats));
    }
    CHECK(fixture_equal(&a, &b, 1));
    OK(dl_evaluate(&a.model, &a.evaluation, a.state.backend, &a.valid, &a.eval_batch, 2,
                     &evaluated));
    CHECK(evaluated.tokens == 16 && evaluated.batches == 2);
    CHECK(evaluated.unscored == a.valid.count - 17);
    puts("evaluation: exact scored counts; pending forward/gradient group unchanged");
    passed = 1;
done:
    fixture_destroy(&b);
    fixture_destroy(&a);
    return passed;
}
```

The all-full-batches call scores $8\lfloor(N-1)/8\rfloor$ targets for $N$ validation bytes. The two-batch call scores exactly 16. Its `unscored` count is $N-1-16$, which includes deliberately excluded full batches as well as any short tail. This checks the meaning of the metric's coverage field.

## Fail after sampling, then replay the whole group

The next test temporarily constructs an optimizer with decay $10^6$. That configuration is finite and nonnegative, so construction is valid. At the scheduled positive learning rate, however, `learning_rate * weight_decay > 1`, and the step rejects its negative shrink factor. Forward, backward, and accumulation have already run: this is not merely a bad argument rejected before any data sampling.

After that failure, the trainer must leave the persistent RNG, weights, moments, update count, target count, and caller's statistics unchanged. Its partial gradient work is discarded and its tape is not ready. The test destroys the deliberately unsuitable optimizer and constructs a normal one; editing an attached optimizer's immutable configuration is not the repair interface. The next complete update must match an untouched twin's first update.

```c title="tests/test_training.c"
static int failed_group(void)
{
    TrainingFixture a = {0}, b = {0};
    int passed = 0;
    OK(fixture_create(&a, DL_BACKEND_BLOCKED, 0));
    OK(fixture_create(&b, DL_BACKEND_BLOCKED, 0));
    dl_adamw_destroy(&a.optim);
    OK(dl_adamw_create(&a.optim, &a.model,
                       (DlAdamWConfig){0.9, 0.99, 1e-8, 1e6, 1.0}, 1024 * 1024));
    DlTrainStats stats = {.loss = 123.0};
    STATUS(dl_train_update(&a.optim, &a.state, &a.training, &a.batch, &a.tape, &stats),
           DL_RANGE);
    CHECK(stats.loss == 123.0 && a.state.rng.state == 2026 && !a.tape.ready);
    CHECK(fixture_equal(&a, &b, 1));
    dl_adamw_destroy(&a.optim);
    OK(dl_adamw_create(&a.optim, &a.model,
                       (DlAdamWConfig){0.9, 0.99, 1e-8, 0.01, 1.0}, 1024 * 1024));
    DlTrainStats other = {0};
    OK(dl_train_update(&a.optim, &a.state, &a.training, &a.batch, &a.tape, &stats));
    OK(dl_train_update(&b.optim, &b.state, &b.training, &b.batch, &b.tape, &other));
    CHECK(stats.loss == other.loss && fixture_equal(&a, &b, 1));
    CHECK(memcmp(a.batch.starts, b.batch.starts, sizeof(size_t)) == 0);
    puts("failure: complete group discarded; next attempt replays RNG and update index");
    passed = 1;
done:
    fixture_destroy(&b);
    fixture_destroy(&a);
    return passed;
}
```

The fixture's data RNG starts at 2026. A broken implementation that draws directly from that owner during the failed attempt will generally still train after a restart, but it will train on different windows. This test makes that difference a failure.

Complete the executable:

```c title="tests/test_training.c"
int main(void)
{
    if (!schedule_checks() || !overfit(DL_BACKEND_REF) || !overfit(DL_BACKEND_BLOCKED) ||
        !evaluation_isolation() || !failed_group()) return EXIT_FAILURE;
    puts("all training checks passed");
    return EXIT_SUCCESS;
}
```

## Restore into an already-used model

Create `tests/test_checkpoint.c`. `_POSIX_C_SOURCE` is defined before all includes so the POSIX temporary-directory and file operations are declared. The one-MiB checkpoint budget is ample for this small test model and corpus; it is not the application's larger budget.

The continuation test creates two live models. One saves after three updates. The other runs five updates and holds a pending forward tape. Loading the older checkpoint into the second must preserve its owning addresses, install the saved optimizer time, advance its **local** revision, and make its old tape unusable.

```c title="tests/test_checkpoint.c"
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
```

The three future groups compare losses, gradient norms, learning rates, clipping coefficients, the final microbatch's starts and token arrays, complete weights and moments, gradient scratch, and RNG state. Both fixed and random modes run under both backends, giving **twelve matched future groups per executable**. The checks are not a claim that different backends or machines must be bitwise identical to each other. Each pair has the same build, backend, inputs, and arithmetic order.

Notice the two separate clocks after loading. `optim.updates` becomes 3. `model.revision` becomes its former live value plus one, not the value it once had at update 3. Restoring an old revision would make some old activation tapes appear current. The explicit failed backward call tests this otherwise subtle lifetime error.

## Build malformed files with explicit byte offsets

These helpers create a test file from chosen bytes and patch a little-endian integer at a known offset. The checkpoint format, not a C structure layout, defines those offsets. `width` is at most eight in the fixtures, so `8 * i` never shifts a 64-bit value by 64 or more. Each listed range lies inside the successfully generated original file.

```c title="tests/test_checkpoint.c"
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
```

The rejection test is longer because it verifies a contract over many failure positions. First it saves after update 3, restores both comparison fixtures, and **advances both live models to update 4**. The file now contains older values than the destination. This matters: a loader that writes old weights into the live model before validating a later field must visibly damage the comparison. Testing only an identical saved/live model can miss that bug.

Let $P$ be the number of unique scalar parameters and $K$ the number of parameter-directory entries. The first moment begins at byte $256+88K+4P$: the fixed header, all 88-byte directory entries, and all weight floats precede it. The dataset begins another $8P$ bytes later, after both moment arrays. Entries and weight blocks are interleaved in the file, but these total offsets still locate the first moment and the last weight.

The fourteen defects cover identification, dimensions/counts, schedule/counters, a parameter name and shape, non-finite values, a negative second moment, the actual training bytes, and the trailer. A six-position truncation sweep includes an empty file, a partial header, a boundary before weight data, a middle cut, and a missing final byte. The helper also checks trailing data, incompatible external text with unchanged length, different schedule hyperparameters, and the completed-boundary requirement.

```c title="tests/test_checkpoint.c"
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
```

Three details deserve a second reading. First, the forged parameter count is `UINT64_MAX`, yet no allocation uses it: the loader compares it against the constructed model's known count. Second, after every failed load the revision and complete live numerical state still match the newer reference. A returned error is insufficient without those assertions. Third, failed saves are tested against a pre-existing valid file. A 300-byte budget crosses the 256-byte header before failing, and a directory at the destination causes replacement to fail. The old checkpoint bytes must remain unchanged.

The file is intentionally not an authenticated container. These tests reject the specified malformed/incompatible cases; they cannot detect every corruption that changes a finite weight into another finite weight. Do not infer stronger integrity guarantees from the number of tests.

Complete the test with temporary-directory ownership and cleanup:

```c title="tests/test_checkpoint.c"
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
```

`mkdtemp` substitutes a unique suffix in the writable character array. The program removes its two ordinary files and then the directory. If a failed-save path leaks an unexpected sibling temporary file, the final `rmdir` fails and so does the test. Cleanup errors therefore cannot silently turn into a successful report. No test uses the user's checkpoint path.

## Exercise a process restart and an operating-system write failure

Create `tests/test_train_cli.py`. It invokes the C executable with argument lists, never a shell command containing interpolated paths. The temporary fixture directory belongs to Python's context manager, and every training process gets a separate invocation. Python does not load tensor values or implement any training arithmetic.

The split run stops after update 5 of a planned total of 12. The resumed process finishes updates 6–12. The uninterrupted process uses the same total of 12 from the beginning. Comparing a five-update schedule with a newly extended twelve-update schedule would not test continuation of the same run.

On POSIX platforms with Python's `resource` module, the test also sets the child process's file-size limit to 1024 bytes. It ignores `SIGXFSZ` in that child so an oversized write can return an I/O failure instead of merely terminating the process. A buffered write may fail during `fwrite`, flush, or closure; the C save routine must handle all of these before replacement. The old checkpoint and absence of a leaked sibling are checked after this **actual operating-system failure**, independently of the library's explicit byte-budget test.

```python title="tests/test_train_cli.py"
#!/usr/bin/env python3
"""Exercise the C executable; Python never evaluates a model or gradient."""
from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/byte-check.conf"
    with tempfile.TemporaryDirectory(prefix="dl-train-") as directory:
        work = Path(directory)
        text = work / "text"
        subprocess.run([sys.executable, str(root / "scripts/make_training_fixture.py"),
                        str(text)], check=True, capture_output=True, text=True)
        split, whole = work / "split.bin", work / "whole.bin"

        def run(mode: str, path: Path, stop: int, *, fail: bool = False,
                preexec_fn=None) -> subprocess.CompletedProcess[str]:
            command = [str(binary), mode, str(config), str(text / "train.bin"),
                       str(text / "valid.bin"), str(path), "12", str(stop),
                       "2", "random", "blocked"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    preexec_fn=preexec_fn, timeout=60)
            if (result.returncode != 0) != fail:
                raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
            return result

        run("new", split, 5)
        old = split.read_bytes()
        refused = run("new", split, 5, fail=True)
        if "refuses an existing checkpoint" not in refused.stderr or split.read_bytes() != old:
            raise AssertionError("new replaced an existing checkpoint")
        print("cli: new refuses existing checkpoint")

        # POSIX/macOS/Linux only. SIGXFSZ must be ignored so stdio can report EFBIG.
        try:
            import resource
        except ImportError:
            print("cli: OS short-write injection skipped (resource unavailable)")
        else:
            def limited_file() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

            failed = run("resume", split, 6, fail=True, preexec_fn=limited_file)
            if "dl_checkpoint_save" not in failed.stderr:
                raise AssertionError("failure did not reach checkpoint saving")
            if split.read_bytes() != old or list(work.glob("split.bin.tmp.*")):
                raise AssertionError("short write damaged old file or leaked temporary")
            print("cli: real OS file-size write failure preserves old checkpoint")

        continued = run("resume", split, 12)
        uninterrupted = run("new", whole, 12)
        if split.read_bytes() != whole.read_bytes():
            raise AssertionError("resume and uninterrupted final state differ")
        # Update timing is observational and is not continuation state.
        def updates(text_output: str) -> list[str]:
            return [line.split(" seconds=", 1)[0] for line in text_output.splitlines()
                    if line.startswith("update=")]
        if updates(continued.stdout) != updates(uninterrupted.stdout)[5:]:
            raise AssertionError("resumed losses or optimizer statistics differ")
        print("cli: seven resumed update logs and complete checkpoint bytes match")

        damaged = bytearray((text / "train.bin").read_bytes())
        damaged[0] ^= 1
        (text / "train.bin").write_bytes(damaged)
        previous = split.read_bytes()
        run("resume", split, 12, fail=True)
        if split.read_bytes() != previous:
            raise AssertionError("incompatible data changed checkpoint")
        print("cli: changed same-length corpus rejected")
    print("all training CLI checks passed")


if __name__ == "__main__":
    main()
```

The file-size behavior is specified by Apple's [`setrlimit` manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/setrlimit.2.html). Python's [`subprocess` documentation](https://docs.python.org/3/library/subprocess.html) documents `check`, captured output, timeouts, and the important restriction on `preexec_fn`: it is unsafe in a threaded parent. This harness is a standalone single-threaded command. Do not move its `preexec_fn` into a threaded notebook, web service, or launcher; use a separate single-threaded helper process there. Platforms without `resource` skip this one injection explicitly rather than claiming to have run it.

The 60-second subprocess timeout bounds this tiny test's execution; it is not a throughput guarantee or a restriction on ordinary training. A very slow instrumented environment can raise that test-only limit while preserving every numerical assertion. The executable path is resolved before constructing child commands, and all configuration/data paths are explicit.

Update wall times are deliberately removed from the comparison. Seven resumed update records must otherwise match the uninterrupted suffix exactly. More strongly, the entire final checkpoint must match byte for byte: this format contains no timestamps, temporary filenames, pointer values, or process-local model revisions. Exact file equality here tests its deterministic encoding as well as continuation.

## Run the complete checks

From the Chapter 15 companion root, the ordinary local command is:

```sh
make test
make test-assets
make test-training-cli
```

On a non-Apple host, select a compatible C compiler explicitly:

```sh
make CC=clang test
make test-assets
make CC=clang test-training-cli
```

`make test` builds and runs fourteen C test programs under debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized settings, then checks the existing matrix comparison executable. `test-assets` retains Chapter 13's nine network-free Python tests. `test-training-cli` uses the release C training executable and temporary local files; it does not download a corpus.

The two new C programs report the following check summaries. The loss decimals shown are observations, while the assertions use the contracts described above:

```text
overfit ref: 5.692190972 -> 0.078423635 on 8 fixed targets after 120 updates
overfit blocked: 5.692190972 -> 0.078423635 on 8 fixed targets after 120 updates
evaluation: exact scored counts; pending forward/gradient group unchanged
failure: complete group discarded; next attempt replays RNG and update index
all training checks passed
resume ref random: three next groups bitwise equal; old tape rejected
resume ref fixed: three next groups bitwise equal; old tape rejected
resume blocked random: three next groups bitwise equal; old tape rejected
resume blocked fixed: three next groups bitwise equal; old tape rejected
checkpoint: 14 corrupt/incompatible fields, six truncations and trailing bytes rejected
checkpoint: dataset/schedule/lifecycle/budget checks; failed saves preserve old file
all checkpoint checks passed
```

The CLI test additionally reports rejection of an existing `new` checkpoint, preservation after the OS file-size failure, equality of seven resumed records and final checkpoint bytes, and rejection of a changed same-length corpus. All these paths were executed on the stated Linux host. Leak-enabled sanitizer runs of the two new C tests reported no leaks. Leak detection support varies by platform; an unsupported detector is not evidence that the application leaked or that it was checked successfully.

No test executes native M4 instructions, downloads TinyStories, samples generated text, or constructs an incremental key/value cache. The actual training results in Part 3 are local diagnostic runs on the original fixture text. Keep those boundaries attached to any quoted result.

## Break state handling deliberately

Use a disposable copy of the source and restore it between experiments. Rebuild the affected sanitizer executable, then run it. These are logical defects: a clean sanitizer run does not excuse an incorrect continuation.

**Commit the RNG too early.** In `dl_train_update`, immediately after a successful random batch sample, add `s->rng = next_rng;` instead of waiting for the successful optimizer step. The deliberately failed group now consumes random state. `test_training` fails its unchanged-RNG assertion. Repair by keeping all draws in `next_rng` until commit.

**Read but discard the checkpoint RNG.** In `contents`, read the serialized RNG into a temporary `uint64_t` instead of into the candidate state's `rng.state`. The file is syntactically consumed, but the destination continues with its old data stream. The random continuation case in `test_checkpoint` fails persistent-state equality after load. Repair the destination, not the file length.

**Reset moments on load.** Just before `dl_adamw_restore` in `dl_checkpoint_load`, zero the two candidate moment arrays. Restored weights can still produce the expected immediate evaluation loss, but Adam's next update is different. `test_checkpoint` fails the moment-state comparison. Repair by preserving both serialized moments and the successful-update count.

**Write weights before validating the rest of the file.** In the read branch of `contents`, copy each successfully decoded candidate weight block into its live parameter immediately. A later malformed moment, dataset byte, or trailer now returns an error after changing the live model. The newer-live/older-file rejection test fails. Repair by keeping all decoded values private until parsing, compatibility checks, exact EOF, and stream closure have succeeded.

All four wrong implementations were run in isolated copies. Each failed an explicit state or numerical assertion without an AddressSanitizer/UndefinedBehaviorSanitizer report. Memory validity and application-state validity are different properties.

A useful valid variation is already exercised by `evaluation_isolation`: score two validation batches rather than every full batch, and verify 16 scored targets plus the correct excluded count. The trainer's next state must not change merely because evaluation covers fewer examples. Do not compare the resulting loss to a different prefix as though it were the same measurement.

## Diagnose a real run

A fixed-batch loss that stays near its initial value should be investigated before increasing model size. Confirm that targets are shifted once, backward succeeds, the optimizer consumes every intended microbatch mean, and weights actually change. Inspect clipping statistics: a consistently tiny coefficient may explain slow changes but is not itself proof of a bad gradient.

A resumed run that immediately evaluates correctly but diverges on its next update suggests missing moments, wrong update time, or different next samples. Check schedule total versus invocation stop, RNG restoration, fixed/random mode, geometry, backend, and exact data bytes. A failed load that changes the current model indicates premature mutation; do not repair it by ignoring the failure.

An apparent validation improvement is meaningful only with the same scored prefix or traversal. Print `scored` and `unscored`; distinguish training loss measured before a step from validation loss measured after it. The current evaluator uses full training-tape storage, and its prefix limit is explicit. Neither choice is an inference optimization.

At this point, the C program trains, evaluates, saves a validated continuation state, and resumes reproducibly on the same build/backend. Keep your own native pilot's configuration, scored coverage, and elapsed-update observations with your working notes. The next chapter separates inference memory from training activations and implements sampling; no future implementation is required to run this chapter.
