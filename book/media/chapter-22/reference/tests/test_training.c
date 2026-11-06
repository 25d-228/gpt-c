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

int main(void)
{
    if (!schedule_checks() || !overfit(DL_BACKEND_REF) || !overfit(DL_BACKEND_BLOCKED) ||
        !evaluation_isolation() || !failed_group()) return EXIT_FAILURE;
    puts("all training checks passed");
    return EXIT_SUCCESS;
}
