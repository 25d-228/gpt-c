# Chapter 15 — Train, evaluate, checkpoint, and resume

The decoder can calculate a loss and every derivative, and the optimizer can transform those derivatives into one update. A training run connects those operations across time. Which examples belong to one update? Does evaluation disturb an unfinished gradient group? After a failed write, which file remains usable? After restarting, do we receive the same next examples or merely a plausible-looking loss?

This chapter builds the complete C training application. We first learn a fixed batch, then sample windows from a bounded text file and measure a separate validation file. We save the state required to continue, not just the state required to run a forward pass.

**Prerequisites:** the complete [Chapter 14 companion](https://25d-228.github.io/gpt-c/media/chapter-14/reference/Makefile), including Chapter 12's model and training tape, Chapter 13's byte batches and explicit random state, and Chapter 14's optimizer. `dl_model_backward(..., DL_SET)` computes a fresh microbatch mean. `dl_adamw_add_mean` consumes it with its actual target count. A successful optimizer step changes the model revision and consumes the whole group.

**Implementation estimate:** **6.00 hours** for this entire chapter: 2.00 for the loop and evaluation, 2.00 for checkpoint/resume, and 2.00 for active debugging and tests. Reading and unattended training are separate. The four linked parts are one chapter: this part implements the trainer; [Part 2](https://25d-228.github.io/gpt-c/chapter-15-part-2.html) implements checkpoints; [Part 3](https://25d-228.github.io/gpt-c/chapter-15-part-3.html) supplies the application and actual training experiments; [Part 4](https://25d-228.github.io/gpt-c/chapter-15-part-4.html) supplies the complete tests and diagnostic variations.

The [57-file companion](https://25d-228.github.io/gpt-c/media/chapter-15/reference/Makefile) contains all prerequisites. Fourteen files are new. Only `include/dl/optim.h`, `src/optim.c`, and the Makefile change among the 43 prerequisite files. The model's forward/backward code does not change. Except for the two explicitly described optimizer insertions, filename-labeled blocks for a file concatenate in reading order, with one blank line between blocks, to form that complete file. The two source insertions for `src/optim.c` appear in this part and Part 2; both go before its existing final `#undef RETURN_IF_ERROR`.

> **Learning objectives**
> Execute complete update groups; keep validation independent of training state; derive a schedule from completed updates; distinguish parameter state from continuation state; serialize fields instead of C object layouts; reject a damaged checkpoint without replacing live state; and demonstrate identical next updates after a restart.

## The state that crosses an update boundary

Let $B$ be the number of sequences in a microbatch, $T$ the number of scored targets per sequence, and $A$ the number of microbatches per optimizer update. One microbatch scores $n=BT$ targets and one completed group scores

$$
N=ABT.
$$

For the byte model's default $B=2$, $T=128$, and $A=4$, this is $N=1024$. Each sequence has its own causal context and positions starting at zero. Accumulation does not splice the four microbatches into a single 1024-token attention sequence.

At a completed boundary, the model owns weights, the optimizer owns moments, and the trainer owns the random-number generator's state and the number of processed targets. The gradient sum is not pending. Its old bytes can remain in memory, but the next `begin` clears them before they are used.

| State | Owner and lifetime | Needed for continuation? |
| --- | --- | --- |
| Unique weights and fixed model configuration | `DlModel` at a stable address | Yes |
| First/second moments and completed updates | `DlAdamW`, borrowing that model | Yes |
| Learning-rate schedule and batching policy | `DlTrainState`, a pointer-free value | Yes |
| Training random state and processed targets | `DlTrainState` | Yes |
| Exact training and validation bytes | Two immutable `DlByteData` owners | Yes, bound to the checkpoint |
| A partial gradient group | Optimizer sum plus tokens | Not allowed at a checkpoint boundary |
| Activations and copied batch IDs | Training or evaluation tape | No; recompute after restart |
| The model revision | A live-object lifetime guard | Do not import it from the file |

Our random sampler chooses windows with replacement. Therefore it has no epoch cursor or shuffled index array to save. Its next choice is determined by its current generator state and the immutable data bytes. The fixed-batch diagnostic always repeats the first full batch, so it does not advance the generator at all. A different loader that shuffled examples would need to save that additional state; recording a seed alone would not replace it.

## A schedule indexed by successful updates

Let $u$ be the number of **completed** optimizer updates. The next attempt uses index $t=u+1$. Let $U$ be the intended total updates, $w$ the number of warmup updates, $\eta_p$ the peak rate, and $\eta_f$ the final positive rate. Require $0\le w<U$, $0<\eta_f\le\eta_p$, and $U\le2^{53}$, matching the optimizer's exact-integer count bound.

During nonempty warmup, increase linearly:

$$
\eta_t=\eta_p\frac{t}{w},\qquad 1\le t\le w.
$$

Set the peak anchor $a=\max(w,1)$. When $w=0$, update one is the peak. After the anchor, interpolate down to the floor:

$$
\eta_t=\eta_f+(\eta_p-\eta_f)\frac{U-t}{U-a},\qquad t>a.
$$

The second formula is evaluated only when $t>a$, so its denominator cannot be zero. A one-update schedule has no decay interval and uses the peak once. The floor is the decay endpoint, not a lower bound on early warmup values.

For $U=6$, $w=2$, $\eta_p=0.01$, and $\eta_f=0.002$, the rates are

$$
[0.005,\ 0.010,\ 0.008,\ 0.006,\ 0.004,\ 0.002].
$$

Suppose the attempt at update three fails. The optimizer still reports $u=2$, so retrying that same group uses $0.008$ again. Neither the count of attempts nor the count of microbatches is an appropriate schedule clock.

Create the public trainer header. The comments distinguish independent output objects from borrowed owners. `DlTrainState` itself owns no heap allocation, so making a speculative value copy of its generator is legitimate; copying a model or tape owner would not be.

```c title="include/dl/train.h"
#ifndef DL_TRAIN_H
#define DL_TRAIN_H

#include "dl/data.h"
#include "dl/optim.h"

typedef struct {
    uint64_t total, warmup;
    double peak, floor;
} DlSchedule;

/* Value state, no owned pointers. Only the trainer/checkpoint API mutates it.
 * Fixed mode repeats the FIRST full batch; random mode samples with replacement.
 * A boundary has optim.active==0, tokens==0, no fresh parameter gradients.
 * All dataset bytes/configuration stay immutable for a run. */
typedef struct {
    DlSchedule schedule;
    DlRng rng;
    size_t batch, time, microbatches;
    DlBackend backend;
    int fixed;
    uint64_t tokens_seen;
} DlTrainState;

typedef struct {
    double loss;
    DlOptimStats optim;
} DlTrainStats;

typedef struct {
    double loss;
    uint64_t tokens;
    size_t batches, unscored;
} DlEvalStats;

/* LR for NEXT update; completed==total is end (DL_RANGE). Output untouched
 * on failure. Positive finite floor<=peak, 0<=warmup<total<=2^53. */
DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out);
/* Check boundary, schedule, counters and exact bytes-per-update arithmetic.
 * No mutation. Data owners need at least one full B*T target batch each. */
DlStatus dl_train_boundary(const DlAdamW *optim, const DlTrainState *state,
                           const DlByteData *training, const DlByteData *valid);
/* Train tape/batch match state B,T. All outputs/storage disjoint, constructed
 * owners unmodified, no concurrent calls. One complete group or no update:
 * after an operational failure, RNG/counters/weights/moments unchanged and
 * partial gradients discarded. Tape/batch scratch may have changed. */
DlStatus dl_train_update(DlAdamW *optim, DlTrainState *state,
                         const DlByteData *data, DlByteBatch *batch,
                         DlTrainTape *tape, DlTrainStats *stats);
/* Separate evaluation tape AND batch (never pending training work). Limit 0
 * means all full batches; otherwise stop after limit. Reset cursor locally.
 * No optimizer/RNG/parameter-gradient writes. out untouched on failure.
 * unscored includes trailing targets AND any remainder excluded by limit. */
DlStatus dl_evaluate(DlModel *model, DlTrainTape *evaluation_tape,
                      DlBackend backend, const DlByteData *data,
                      DlByteBatch *evaluation_batch, size_t limit,
                      DlEvalStats *out);

#endif
```

Create `src/train.c`. The existing status-return macro returns immediately on failure; none of these early returns should be mistaken for a successful partial update. `dl_schedule_rate` computes into a local scalar and commits its output only after the result is positive and finite. Integer division would destroy the interpolation, so both operands are explicitly converted to `double`.

```c title="src/train.c"
#include "dl/train.h"

#include <math.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out)
{
    if (out == NULL || !isfinite(s.peak) || !isfinite(s.floor) ||
        s.floor <= 0.0 || s.peak < s.floor || s.total == 0 ||
        s.total > DL_OPT_MAX_COUNT || s.warmup >= s.total) return DL_ARGUMENT;
    if (completed >= s.total) return DL_RANGE;
    uint64_t next = completed + 1;
    uint64_t anchor = s.warmup == 0 ? 1 : s.warmup;
    double rate = s.peak;
    if (next <= anchor) {
        if (s.warmup != 0) rate *= (double)next / (double)s.warmup;
    } else {
        double remaining = (double)(s.total - next) /
                           (double)(s.total - anchor);
        rate = s.floor + (s.peak - s.floor) * remaining;
    }
    if (!isfinite(rate) || rate <= 0.0) return DL_NUMERIC;
    *out = rate;
    return DL_OK;
}
```

The schedule is an explicit policy for this program, not an assertion that a linear schedule is optimal for every model. Do not extend `TOTAL` while resuming the same experiment: that changes the intended decay trajectory. Run the same total budget with different stopping points, or deliberately start a new experiment.

## Recognize a real boundary before touching state

The optimizer's `active` and `tokens` fields distinguish an incomplete group from stale scratch bytes. We also reject a fresh, unconsumed model gradient. Silently discarding that gradient during checkpoint saving would turn a programming error into a lost update.

Because this trainer keeps $B$, $T$, and $A$ fixed for a run, its processed-target counter must satisfy

$$
\text{tokens\_seen}=uABT.
$$

This counts **uses** of target tokens. Repeated windows count repeatedly; it does not mean that many unique bytes have been read. Check the multiplication before evaluating it. `UINT64_MAX / targets` is the division-based bound from the earlier size-arithmetic chapters.

The trainer deliberately requires both files to contain at least one full $BT$-target batch, even though random sampling alone could draw several rows from a smaller file. This keeps random training, fixed training, and deterministic evaluation under one simple data contract. A complete $BT$ targets require at least $BT+1$ source bytes.

```c title="src/train.c"
static DlStatus state_check(const DlAdamW *o, const DlTrainState *s,
                             const DlByteData *d, uint64_t *group)
{
    if (o == NULL || s == NULL || d == NULL || group == NULL ||
        o->model == NULL || o->model->params == NULL || o->moments == NULL ||
        o->sum == NULL || d->bytes == NULL) return DL_ARGUMENT;
    if (o->model->config.vocab != 256 || s->batch == 0 || s->time == 0 ||
        s->microbatches == 0 || (s->fixed != 0 && s->fixed != 1) ||
        (s->backend != DL_BACKEND_REF && s->backend != DL_BACKEND_BLOCKED))
        return DL_ARGUMENT;
    if (o->revision != o->model->revision || o->count != o->model->value_count ||
        o->active || o->tokens != 0 || o->model->gradients_valid) return DL_STATE;
    double ignored = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, 0, &ignored));
    size_t rows = 0, targets = 0;
    RETURN_IF_ERROR(dl_size_mul(s->batch, s->time, &rows));
    RETURN_IF_ERROR(dl_size_mul(rows, s->microbatches, &targets));
    if (targets > DL_OPT_MAX_COUNT || d->count > d->capacity ||
        d->count <= rows || s->time - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    if (o->updates > s->schedule.total ||
        o->updates > UINT64_MAX / (uint64_t)targets) return DL_RANGE;
    if (s->tokens_seen != o->updates * (uint64_t)targets) return DL_STATE;
    *group = (uint64_t)targets;
    return DL_OK;
}

DlStatus dl_train_boundary(const DlAdamW *o, const DlTrainState *s,
                           const DlByteData *training, const DlByteData *valid)
{
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, training, &group));
    return state_check(o, s, valid, &group);
}
```

`state_check` checks only legitimate, constructed owners and immutable metadata; it is not a way to validate an arbitrary forged pointer. The low-level owners still carry the contracts from Chapters 2, 12, and 13. The aggregate limit $2^{53}$ ensures an exact conversion of the group's integer target count to `double`, not exact floating-point gradient accumulation.

## Explicitly discard a failed group

Chapter 14 provides `begin`, `add_mean`, and `step`. We add an explicit `abort`: clear the current group and leave it inactive, without changing weights, moments, update count, or revision. This is also how a newly constructed model enters the trainer's boundary state. The model constructor marks its initialized zero gradients as valid for ordinary model operations; those zeros are not a fresh training microbatch.

Insert the following declarations immediately **before the final `#endif`** in `include/dl/optim.h`. Do not remove any existing declarations. The restore operation is implemented in Part 2.

```c title="include/dl/optim.h"
/* Discard a partial group (or clear stale boundary scratch), retaining weights,
 * moments, update count and revision. Does not rewind any external RNG. */
DlStatus dl_adamw_abort(DlAdamW *optim);
/* Coordinated restore into a STABLE model/optimizer pair at a boundary.
 * Three independent flat arrays, each count floats in parameter-directory
 * order. They must not overlap any live model/optimizer storage or each other.
 * Checks all values (second moments >=0) and revision capacity before commit.
 * On success advances the LIVE revision, installs weights/moments/update count,
 * clears gradient/sum scratch, and invalidates all old tapes by revision.
 * Does not change optimizer config; the checkpoint must match it beforehand.
 * On failure all live state unchanged. Not safe for concurrent readers. */
DlStatus dl_adamw_restore(DlAdamW *optim, const float *weights,
                          const float *first, const float *second,
                          size_t count, uint64_t updates);
```

Insert this complete function immediately **before the existing final `#undef RETURN_IF_ERROR`** in `src/optim.c`. Its call to `dl_adamw_begin` uses the existing attachment check and gradient clearing. That function sets `tokens=0`; we then set `active=0`. The caller, not the optimizer, decides what to do with data position or random state.

```c title="src/optim.c"
DlStatus dl_adamw_abort(DlAdamW *o)
{
    RETURN_IF_ERROR(dl_adamw_begin(o));
    o->active = 0;
    return DL_OK;
}
```

An abort is not an undo of a successful optimizer step. A step preflights every prospective weight and moment before committing; after it succeeds, the update belongs to history. The trainer calls `abort` only when no step has committed. Recovering from an operating-system process failure instead requires the last completed checkpoint.

## Sample speculatively, commit once

One group follows this sequence:

```text
boundary -> begin -> sample/forward/backward/add_mean repeated A times
         -> step -> commit generator and processed-target count -> boundary
```

Copy the training generator into `next_rng` before sampling. Every random microbatch advances only that copy. If any operation fails, discard the partial gradients and return without writing `state.rng`. A retry starts from the same next windows. This differs from saving the initial seed: the current state may already be thousands of draws into the stream, and rejection sampling can consume a varying number of words.

Keep a target-weighted loss sum as well as a target-weighted gradient sum. For microbatch means $L_i$ with counts $n_i$,

$$
L_{\mathrm{group}}=\frac{\sum_i n_iL_i}{\sum_i n_i}.
$$

For two targets at mean loss $1.2$ and six at mean loss $0.4$, the combined loss is $(2\cdot1.2+6\cdot0.4)/8=0.6$, not the unweighted average $0.8$. Our current microbatches are equal-sized, but the code still expresses the count weighting directly.

Continue `src/train.c` with the complete group operation:

```c title="src/train.c"
DlStatus dl_train_update(DlAdamW *o, DlTrainState *s, const DlByteData *data,
                         DlByteBatch *batch, DlTrainTape *tape, DlTrainStats *stats)
{
    if (batch == NULL || tape == NULL || stats == NULL) return DL_ARGUMENT;
    uint64_t group = 0;
    RETURN_IF_ERROR(state_check(o, s, data, &group));
    if (batch->storage == NULL || batch->batch != s->batch ||
        batch->time != s->time || tape->model != o->model ||
        tape->arena.data == NULL || tape->rows != batch->count ||
        tape->attention_config.batch != s->batch ||
        tape->attention_config.time != s->time) return DL_SHAPE;
    if (s->tokens_seen > UINT64_MAX - group) return DL_OVERFLOW;
    double rate = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate(s->schedule, o->updates, &rate));
    DlRng next_rng = s->rng; /* Speculative stream; commit only after step. */
    DlStatus status = dl_adamw_begin(o);
    if (status != DL_OK) return status;
    double weighted_loss = 0.0;
    for (size_t i = 0; i < s->microbatches; ++i) {
        if (s->fixed) {
            size_t cursor = 0;
            int has = 0;
            status = dl_byte_batch_next(data, batch, &cursor, &has);
            if (status == DL_OK && !has) status = DL_RANGE;
        } else status = dl_byte_batch_sample(data, batch, &next_rng);
        double loss = 0.0;
        if (status == DL_OK)
            status = dl_model_forward(o->model, tape, s->backend,
                                      batch->ids, batch->targets, batch->count, &loss);
        if (status == DL_OK) status = dl_model_backward(o->model, tape, DL_SET);
        if (status == DL_OK) status = dl_adamw_add_mean(o, (uint64_t)batch->count);
        if (status != DL_OK) break;
        weighted_loss += loss * (double)batch->count;
        if (!isfinite(weighted_loss)) { status = DL_NUMERIC; break; }
    }
    DlTrainStats result = {.loss = weighted_loss / (double)group};
    if (status == DL_OK) status = dl_adamw_step(o, rate, &result.optim);
    if (status != DL_OK) {
        tape->ready = 0;
        /* No update has committed; abort cannot fail for this attached owner. */
        DlStatus cleanup = dl_adamw_abort(o);
        return cleanup == DL_OK ? status : cleanup;
    }
    s->rng = next_rng;
    s->tokens_seen += group;
    *stats = result;
    return DL_OK;
}
```

The order of calls matters. `backward(..., DL_SET)` replaces the model's temporary gradient for each microbatch. `add_mean` consumes that mean into the optimizer's independent accumulator. Calling backward with `DL_ADD` here would count previous contributions again. No weights change between microbatches.

`result.loss` describes the examples at the **pre-update** weights. `result.optim` describes the step that follows. They are returned together only after the step succeeds. A numerical error can leave tape or batch scratch partly written, but those are not continuation state. The error path clears the gradient group, invalidates the training tape, and leaves the caller's statistics untouched. Structural errors detected before `begin` do not even start a group.

Consider a failure in the final optimizer preflight after all four microbatches. The speculative generator has advanced, but the stored generator has not. Both moment arrays and the model revision are unchanged because no candidate was committed. Aborting discards all four gradients. It would be incorrect to keep three earlier microbatches, skip the fourth, and label that as the planned 1024-target update.

## Validation is forward-only, not state-free by accident

Evaluation uses a **different tape and a different batch owner**. The model weights are shared for read access. Calling forward on the training tape would destroy its pending activations, even if the forward routine never touched a parameter. Reusing the training batch can similarly obscure which data is pending. Distinct storage makes the separation visible.

For a validation file containing $D_v$ bytes, a full batch scores $BT$ targets. Without a batch limit, the scored count is

$$
Q=BT\left\lfloor\frac{D_v-1}{BT}\right\rfloor.
$$

The unscored remainder is $D_v-1-Q$. With a positive limit $K$ on full batches, replace the floor above by

$$
\min\left(K,\left\lfloor\frac{D_v-1}{BT}\right\rfloor\right).
$$

The minus one is not a special-token convention: the final source byte has no following byte inside this file. Each row receives only its own length-$T$ prefix. The measurement is teacher-forced next-byte cross-entropy with context reset at every row, not a single uninterrupted pass with all earlier file bytes available.

Continue and finish `src/train.c`:

```c title="src/train.c"
DlStatus dl_evaluate(DlModel *model, DlTrainTape *tape, DlBackend backend,
                      const DlByteData *data, DlByteBatch *batch,
                      size_t limit, DlEvalStats *out)
{
    if (model == NULL || tape == NULL || data == NULL || batch == NULL ||
        out == NULL || model->params == NULL || batch->storage == NULL)
        return DL_ARGUMENT;
    if (model->config.vocab != 256 || tape->model != model ||
        tape->rows != batch->count || tape->attention_config.batch != batch->batch ||
        tape->attention_config.time != batch->time) return DL_SHAPE;
    DlEvalStats result = {0};
    size_t cursor = 0;
    double total_loss = 0.0;
    for (;;) {
        if (limit != 0 && result.batches == limit) break;
        int has = 0;
        RETURN_IF_ERROR(dl_byte_batch_next(data, batch, &cursor, &has));
        if (!has) break;
        double loss = 0.0;
        RETURN_IF_ERROR(dl_model_forward(model, tape, backend, batch->ids,
                                         batch->targets, batch->count, &loss));
        total_loss += (double)batch->count * loss;
        if (!isfinite(total_loss)) return DL_NUMERIC;
        ++result.batches;
    }
    if (cursor == 0) return DL_RANGE;
    result.tokens = (uint64_t)cursor;
    result.unscored = data->count - 1 - cursor;
    result.loss = total_loss / (double)cursor;
    *out = result;
    return DL_OK;
}

#undef RETURN_IF_ERROR
```

The local `cursor` starts at zero on every call. The sequential data iterator has no random-state argument, so validation cannot consume training draws. `dl_model_forward` writes this evaluation tape but leaves parameter gradients unchanged. There is no call to `zero_grad`, `begin`, `backward`, or `step` in the evaluation routine.

If there is not even one complete batch, evaluation returns `DL_RANGE` rather than a fabricated zero loss. The output record is committed only on success. A limited pass reports all unscored targets, including those excluded by the limit—not merely the short tail. The application will label that measurement a validation **prefix**, not a full-file score.

A separate evaluation tape currently reserves the same full training workspace as the training tape, even though backward is not called. This is intentional reuse of already-understood code. Chapter 16 will build the smaller inference workspace; we do not hide one inside an unexplained evaluation helper now.

## Before writing a large training loop

Predict two checks from these interfaces. First, the fixed-batch mode should leave the data generator's state unchanged through hundreds of updates. Second, evaluation inserted between a pending training forward and backward should not change that backward's gradients. Part 4 supplies complete programs for both checks, including a real overfit experiment and a failed optimizer step followed by an exact replay.

The chapter's runnable commands and configuration files are in Part 3. Continue with [Part 2 — Versioned checkpoints and coherent restoration](https://25d-228.github.io/gpt-c/chapter-15-part-2.html), where the update boundary becomes a file format.
