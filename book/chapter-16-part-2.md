# Chapter 16 · Part 2 — Weight-only loading and categorical selection

[Part 1](https://25d-228.github.io/gpt-c/chapter-16.html) supplied a forward path whose owners contain no gradients or moments. Now load it without constructing training state, then turn a row of logits into one token ID. Both operations have a commit boundary: an incomplete file must not create ready weights, and a failed selection must not consume the caller's random state.

This is the second part of the same **3.00-hour** chapter. The existing Chapter 15 checkpoint format is retained; no conversion utility or new model-computation dependency is introduced.

## Reading weights is not resuming training

A training restart must match the optimizer, schedule, data geometry, and exact datasets. An inference process needs the model configuration and weights. It should not need the original corpus files merely because the training checkpoint embeds them.

We will read the same `DLCKPT01` file. The caller supplies the expected `DlModelConfig`, normally read from its local configuration file. That already-validated configuration determines every allocation and parameter shape. File fields are compared with it, not accepted as unchecked allocation sizes.

The format's total length is

$$
256+88N_p+12P+N_{\mathrm{train}}+N_{\mathrm{valid}}+8\quad\text{bytes},
$$

where $N_p=2+9L$ is the parameter-record count, $P$ the unique value count, and $N_{\mathrm{train}},N_{\mathrm{valid}}$ the embedded dataset byte lengths. Each parameter has 64 name bytes and three eight-byte integers before its values. The $12P$ term covers four bytes of weights and eight bytes of optimizer moments per parameter value.

Inference retains only $4P$ of that payload. It still consumes and checks the rest: finite first moments, finite nonnegative second moments, the declared data bytes, the trailer, and exact end of file. The embedded datasets have no restricted byte alphabet, so they are consumed without retaining or comparing them. This is not continuation validation: it cannot establish that the file contains the corpus originally intended by the caller.

Replace `include/dl/checkpoint.h` with this complete version. The old training entry points are unchanged.

```c title="include/dl/checkpoint.h"
#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"
#include "dl/inference.h"

/* Version 1: explicit little-endian integers and IEEE binary32/binary64.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, backend, schedule and fixed mode.
 * Dataset bytes are included and compared exactly (not merely file lengths).
 * No untrusted file field controls an allocation. max_file_bytes also bounds
 * the temporary 3*P*sizeof(float) load payload; metadata/stdio are excluded.
 * Ordinary local files in a caller-controlled directory; no concurrent edits,
 * writers, adversarial paths, live-owner moves, or aliasing of output state.
 * DL_SHAPE: incompatibility; DL_RANGE: truncation/budget/trailing bytes;
 * DL_NUMERIC: bad weights/moments; DL_STATE: lifecycle or I/O failure.
 * SAVE: write sibling temporary file, flush/sync/close, then rename. Until the
 * successful rename, an existing destination remains untouched. No automatic
 * retries or power-loss durability guarantee (no parent-directory/full sync).
 * LOAD: read/check/close candidate first, then coordinated in-place restore.
 * A failed load preserves all live state. Success increments the LIVE model
 * revision (never imports a saved revision), invalidating every old tape. */
DlStatus dl_checkpoint_save(const char *path, const DlAdamW *optim,
                            const DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);
DlStatus dl_checkpoint_load(const char *path, DlAdamW *optim,
                            DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);

/* Inference-only read of the SAME version-1 training file. expected config is
 * supplied by the caller, never used unchecked from the file. Empty out only.
 * Allocates P weight floats, not a DlModel, gradients, moments or datasets.
 * Validates header/directory/finite weights/moments/trailer/exact EOF; consumes
 * embedded data in bounded chunks without requiring original dataset files.
 * Unlike resume, does NOT compare dataset identity or import trainer RNG/state.
 * max_file_bytes bounds parsed bytes (plus one EOF probe); max_float_bytes
 * bounds weights.
 * Only after checked close does out become immutable ready weights. Failure
 * frees the entire candidate and leaves out unchanged. Not an integrity or
 * authenticity check: structurally valid changed finite values can pass. */
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes);

#endif
```

The loading contract is local and bounded, not a hostile-filesystem security interface. It assumes an ordinary file in a caller-controlled directory, no concurrent writer, and truthful live C owners. There is no authenticated-integrity claim: a different finite weight value that preserves the format can pass. The purpose of these checks is to prevent malformed, incompatible, truncated, or numerically invalid state from becoming a ready model.

## Reuse the existing byte transport

Keep all of Chapter 15's `src/checkpoint.c`, including its platform checks, explicit little-endian integer decoder, save path, and coordinated training restore. Replace **only** its existing `float_values` function with the following complete function. The sole behavior change is that a null read destination means “decode and validate, but do not retain.” Existing training load calls still supply destinations.

```c title="src/checkpoint.c"
static DlStatus float_values(Io *io, const float *source, float *destination,
                              size_t count, int nonnegative)
{
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        float value = 0.0f;
        if (!io->reading) memcpy(&bits, &source[i], sizeof(bits));
        uint64_t word = bits;
        RETURN_IF_ERROR(integer(io, &word, 4));
        bits = (uint32_t)word;
        memcpy(&value, &bits, sizeof(value));
        if (!isfinite(value) || (nonnegative && value < 0.0f)) return DL_NUMERIC;
        if (io->reading && destination != NULL) destination[i] = value;
    }
    return DL_OK;
}
```

Notice the order: `integer` consumes four explicit file bytes, `memcpy` transfers their decoded bit pattern into a float without pointer-punning, and finiteness/sign checks happen **before** the optional store. A null destination must not skip these checks. This is how the inference loader avoids allocating moments without silently trusting their byte region.

Insert the following four pieces, in order, **immediately before the existing final `#undef RETURN_IF_ERROR`** in `src/checkpoint.c`. They add helpers and `dl_checkpoint_load_weights`; do not replace the previous `contents`, `dl_checkpoint_save`, or `dl_checkpoint_load` functions.

First decode finite doubles and consume arbitrary bytes with a fixed 4096-byte stack buffer:

```c title="src/checkpoint.c"
static DlStatus read_double(Io *io, double *out)
{
    uint64_t bits = 0;
    RETURN_IF_ERROR(integer(io, &bits, 8));
    memcpy(out, &bits, sizeof(*out));
    return isfinite(*out) ? DL_OK : DL_NUMERIC;
}

static DlStatus discard_bytes(Io *io, uint64_t count)
{
    unsigned char scratch[4096];
    while (count != 0) {
        size_t chunk = count < sizeof(scratch) ? (size_t)count : sizeof(scratch);
        RETURN_IF_ERROR(transfer(io, scratch, chunk));
        count -= (uint64_t)chunk;
    }
    return DL_OK;
}
```

`discard_bytes` reads through the existing budgeted transport. It does not seek over an unchecked region or allocate the length stated in the file. Its fixed buffer is reused until all declared bytes have been consumed or a short read, budget, or I/O error occurs.

The first ten integer fields are immutable model/format information. The next nine describe the old training run. The latter do not need to match the inference prefix length, but they must be valid according to the saved format. In particular, batch, time, and accumulation cannot be zero, their product must fit the exact optimizer-count bound, and processed tokens must agree with completed updates.

```c title="src/checkpoint.c"
static DlStatus weight_header(Io *io, const DlWeights *w,
                               uint64_t *training_bytes, uint64_t *valid_bytes)
{
    RETURN_IF_ERROR(expect_bytes(io, "DLCKPT01", 8));
    DlModelConfig c = w->config;
    const uint64_t fixed[] = {1, c.vocab, c.layers, c.width, c.query_heads,
        c.kv_heads, c.ffn_width, c.seed, w->param_count, w->arena.count};
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); ++i)
        RETURN_IF_ERROR(expect_integer(io, fixed[i], 8));
    /* B,T,A,backend,fixed,total,warmup,training bytes,validation bytes. */
    uint64_t h[9] = {0};
    for (size_t i = 0; i < 9; ++i) RETURN_IF_ERROR(integer(io, &h[i], 8));
    if (c.vocab != 256 || h[0] == 0 || h[1] == 0 || h[2] == 0 ||
        h[0] > SIZE_MAX || h[1] > SIZE_MAX || h[2] > SIZE_MAX ||
        h[1] - 1 > DL_ROPE_MAX_POSITION || h[3] > 1 || h[4] > 1 ||
        h[7] > SIZE_MAX || h[8] > SIZE_MAX) return DL_RANGE;
    size_t rows = 0, group = 0;
    RETURN_IF_ERROR(dl_size_mul((size_t)h[0], (size_t)h[1], &rows));
    RETURN_IF_ERROR(dl_size_mul(rows, (size_t)h[2], &group));
    if ((uint64_t)group > DL_OPT_MAX_COUNT || h[7] <= rows || h[8] <= rows)
        return DL_RANGE;
    RETURN_IF_ERROR(expect_float(io, c.epsilon));
    RETURN_IF_ERROR(expect_float(io, c.init_std));
    RETURN_IF_ERROR(expect_double(io, c.rope_theta));
    double r[7] = {0}; /* beta1,beta2,epsilon,decay,clip,peak,floor */
    for (size_t i = 0; i < 7; ++i) RETURN_IF_ERROR(read_double(io, &r[i]));
    if (r[0] < 0.0 || r[0] >= 1.0 || r[1] < 0.0 || r[1] >= 1.0 ||
        r[2] <= 0.0 || r[3] < 0.0 || r[4] < 0.0) return DL_RANGE;
    double unused_rate = 0.0;
    RETURN_IF_ERROR(dl_schedule_rate((DlSchedule){h[5], h[6], r[5], r[6]},
                                     0, &unused_rate));
    uint64_t updates = 0, tokens = 0, unused_rng = 0;
    RETURN_IF_ERROR(integer(io, &updates, 8));
    RETURN_IF_ERROR(integer(io, &tokens, 8));
    RETURN_IF_ERROR(integer(io, &unused_rng, 8));
    if (updates > h[5] || updates > UINT64_MAX / (uint64_t)group ||
        tokens != updates * (uint64_t)group) return DL_RANGE;
    *training_bytes = h[7];
    *valid_bytes = h[8];
    return DL_OK;
}
```

The two dataset lengths are the only values passed out of this helper. The saved sampling geometry does not become an inference batch, the update count does not become a generation count, and `unused_rng` does not become the generator's seed. A model's weight shapes are independent of the trainer's batch size and the current prefix length.

The schedule helper is reused only to validate the serialized schedule. No optimizer object is created. The saved initialization seed remains part of the expected model configuration because version 1 already defines it that way, even though reading weights does not perform initialization.

The directory reader now consumes each expected name, shape, and decay flag, storing values directly into the private candidate arena. It then checks and discards the two moment arrays and the data payloads:

```c title="src/checkpoint.c"
static DlStatus weight_contents(Io *io, DlWeights *w)
{
    uint64_t training_bytes = 0, valid_bytes = 0;
    RETURN_IF_ERROR(weight_header(io, w, &training_bytes, &valid_bytes));
    for (size_t p = 0; p < w->param_count; ++p) {
        DlWeight *param = &w->params[p];
        DlMatrix v = param->value;
        RETURN_IF_ERROR(expect_bytes(io, param->name, sizeof(param->name)));
        RETURN_IF_ERROR(expect_integer(io, v.rows, 8));
        RETURN_IF_ERROR(expect_integer(io, v.cols, 8));
        RETURN_IF_ERROR(expect_integer(io, (uint64_t)param->decay, 8));
        RETURN_IF_ERROR(float_values(io, NULL, v.base + v.offset, v.rows * v.cols, 0));
    }
    /* Decode/check but retain neither moment array. No seek past unchecked data. */
    RETURN_IF_ERROR(float_values(io, NULL, NULL, w->arena.count, 0));
    RETURN_IF_ERROR(float_values(io, NULL, NULL, w->arena.count, 1));
    RETURN_IF_ERROR(discard_bytes(io, training_bytes));
    RETURN_IF_ERROR(discard_bytes(io, valid_bytes));
    RETURN_IF_ERROR(expect_bytes(io, "ENDCKP01", 8));
    if (fgetc(io->file) != EOF) return DL_RANGE;
    return ferror(io->file) ? DL_STATE : DL_OK;
}
```

A successful read of the last weight is not enough to commit. A damaged moment, missing data suffix, wrong trailer, extra byte, or close error still rejects the candidate. The final `fgetc` is one EOF probe beyond the parsed format bytes; this is why the header describes the file budget as parsed bytes plus that probe.

Finally, expose the loader. All potentially failing work, including `fclose`, precedes the public ownership transfer:

```c title="src/checkpoint.c"
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes)
{
    if (path == NULL || path[0] == '\0' || out == NULL || max_file_bytes == 0)
        return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    RETURN_IF_ERROR(platform_check());
    DlWeights next = {0};
    RETURN_IF_ERROR(dl_weights_create(&next, expected, max_float_bytes));
    FILE *file = fopen(path, "rb");
    if (file == NULL) { dl_weights_destroy(&next); return DL_STATE; }
    Io io = {file, max_file_bytes, 1};
    DlStatus status = weight_contents(&io, &next);
    if (fclose(file) != 0) status = DL_STATE;
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    next.ready = 1;
    *out = next;
    return DL_OK;
}
```

On success, the new owner is immutable and ready. On failure, its arena and directory are destroyed and the caller's destination remains empty. Loading into a live owner is rejected rather than invalidating a workspace's borrow. Replacing weights requires destroying the workspace, destroying the old weight owner, then loading and constructing a new pair.

This path allocates no `DlModel`, parameter gradients, optimizer moments, gradient sum, training tape, or corpus owner. Linking the existing training module into the same executable does not allocate its runtime objects. The loader's read time still includes the complete checkpoint, which is a separate cost from retained memory.

## Temperature changes relative odds

Let a finite logit row be $z_0,\ldots,z_{V-1}$. For a finite positive temperature $\tau$, define

$$
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_{j=0}^{V-1}\exp(z_j/\tau)}.
$$

These probabilities describe a categorical distribution: exactly one ID is selected. They are not independent yes/no choices for each token. The [Transformers `TemperatureLogitsWarper` documentation](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TemperatureLogitsWarper) describes temperature as a positive logit-scaling parameter used with sampling. We implement the scaling and selection directly.

For two positive-probability outcomes,

$$
\frac{p_i}{p_j}=\exp\left(\frac{z_i-z_j}{\tau}\right).
$$

Reducing $\tau$ increases the odds ratio favoring the larger logit. It does not require changing weights or computing another decoder forward pass. With logits $[0,\log2,\log4]$:

| Temperature | Relative weights | Probabilities |
| --- | --- | --- |
| $1$ | $[1,2,4]$ | $[1,2,4]/7$ |
| $1/2$ | $[1,4,16]$ | $[1,4,16]/21$ |
| $2$ | $[1,\sqrt2,2]$ | $[1,\sqrt2,2]/(3+\sqrt2)$ |

The C logits are float approximations to these logarithms, so the tests use a small tolerance rather than exact equality with the rational formulas.

Greedy decoding is a separate operation. We choose the lowest ID among equal largest logits, and it draws no random number. Temperature zero is rejected, not treated as a magic greedy flag. In particular, as positive temperature approaches zero, **tied** maxima can retain equal mass; that limit does not implement our deterministic tie rule.

## Subtract before dividing, and widen before subtracting

Set $m=\max_i z_i$ and calculate

$$
a_i=\exp\left(\frac{z_i-m}{\tau}\right),\qquad
S=\sum_i a_i,\qquad p_i=a_i/S.
$$

The common factor $\exp(m/\tau)$ cancels, so the exact distribution is unchanged. Each exponent is nonpositive. At least one maximal logit yields $a_i=1$, so the sum cannot be zero. Very small weights may underflow to zero; that removes numerically unrepresentable mass without inventing a uniform fallback.

The order of operations matters. Calculating `z / temperature` first can overflow to positive infinity for several finite positive logits; subtracting a separately overflowed maximum then gives an invalid `infinity - infinity`. Calculating the difference in **float** first can overflow when subtracting opposite extremes. Instead, widen each logit to double, subtract the double maximum, and only then divide.

For input `[-FLT_MAX, FLT_MAX, FLT_MAX]` and an extremely small positive temperature, the negative scaled difference can become negative infinity on the target's IEEE arithmetic. Its exponential is zero. The two maximal entries compute `0 / temperature`, retain weight one, and normalize to `[0, 0.5, 0.5]`. An intermediate negative infinity here is not the same as accepting a non-finite input logit. Every input logit, including an outcome that would lose, is required to be finite.

Create the complete selection header:

```c title="include/dl/sample.h"
#ifndef DL_SAMPLE_H
#define DL_SAMPLE_H

#include "dl/common.h"
#include "dl/rng.h"

/* Independent, live, disjoint arrays/outputs of stated lengths; count>0.
 * All logits finite (even entries that would lose). Ties choose lowest ID.
 * Output ID unchanged on failure; no allocation or RNG in these three calls. */
DlStatus dl_greedy(const float *logits, size_t count, size_t *token);
/* Finite temperature>0; zero is NOT a greedy mode. Double probabilities.
 * Preflight failure leaves probabilities unchanged. Extreme negative scaled
 * differences may become -infinity; their exp is zero, not an invalid input.
 * No top-k/p filtering. At least one maximal logit has unnormalized weight 1. */
DlStatus dl_sample_probabilities(const float *logits, size_t count,
                                 double temperature, double *probabilities);
/* Finite 0<=p<=1, |sum(p)-1|<=1e-10; finite 0<=u<1. Renormalize the accepted
 * rounding residue using target=u*sum(p). Strict '<' implements half-open
 * intervals. Skip zero mass. A rounded terminal endpoint uses the LAST
 * positive entry, never a trailing zero. Output unchanged on failure. */
DlStatus dl_categorical(const double *probabilities, size_t count,
                        double uniform, size_t *token);
/* scratch[count] doubles may be overwritten. RNG/token commit only on success;
 * exactly ONE uniform draw per successful sampled token, even if mass is
 * concentrated at one ID. Does not consume a training or model RNG. */
DlStatus dl_sample_next(const float *logits, size_t count, double temperature,
                        DlRng *rng, double *scratch, size_t *token);

#endif
```

Create `src/sample.c`. The first two routines implement greedy selection and probabilities. They validate every input before changing a public output. `dl_greedy` uses `>` rather than `>=`, so a later tie cannot replace the first maximum.

```c title="src/sample.c"
#include "dl/sample.h"

#include <math.h>

DlStatus dl_greedy(const float *logits, size_t count, size_t *token)
{
    if (logits == NULL || token == NULL || count == 0) return DL_ARGUMENT;
    size_t best = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(logits[i])) return DL_NUMERIC;
        if (logits[i] > logits[best]) best = i;
    }
    *token = best;
    return DL_OK;
}

DlStatus dl_sample_probabilities(const float *logits, size_t count,
                                 double temperature, double *probabilities)
{
    if (logits == NULL || probabilities == NULL || count == 0 ||
        !isfinite(temperature) || temperature <= 0.0) return DL_ARGUMENT;
    double maximum = -(double)INFINITY;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(logits[i])) return DL_NUMERIC;
        if ((double)logits[i] > maximum) maximum = logits[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        /* Widen BEFORE subtraction; subtract BEFORE dividing by temperature. */
        probabilities[i] = exp(((double)logits[i] - maximum) / temperature);
        sum += probabilities[i];
    }
    for (size_t i = 0; i < count; ++i) probabilities[i] /= sum;
    return DL_OK;
}
```

For finite float inputs and positive finite temperature, the widened differences are finite before division, every exponential lies in $[0,1]$, and the sum is positive. The practical array lengths supported by the host cannot overflow a double sum of terms at most one. The routine therefore needs no arbitrary “all logits invalid” fallback after its preflight. It uses the same no-fast-math build policy as the numerical chapters.

The probability array is caller-owned scratch. The generator can reuse it for each token. It is not retained in the model or checkpoint, and no gradient is computed through sampling.

## Select a half-open interval

Define cumulative probabilities $c_i=\sum_{j=0}^{i}p_j$, with $c_{-1}=0$. A uniform draw $u\in[0,1)$ selects ID $i$ when

$$
c_{i-1}\le u<c_i.
$$

For `[0, 0.25, 0.5, 0.25, 0]`, ID zero has no interval. ID one owns $[0,0.25)$, ID two owns $[0.25,0.75)$, and ID three owns $[0.75,1)$. At the exact boundary $u=0.25$, choose ID two, not ID one. At $u=0$, skip the zero-mass entry. A draw of one is invalid.

Floating-point probabilities may sum to a number very slightly different from one. The public categorical routine accepts only finite entries in $[0,1]$, with positive total and an absolute total error no greater than $10^{-10}$. It selects against `u * sum`, using the actual accepted total. This is a narrowly specified rounding allowance, not permission to supply arbitrary unnormalized weights.

If multiplication rounds the target up to the final cumulative endpoint, use the **last positive** entry. Returning `count - 1` unconditionally could select a trailing zero-probability token. A zero-mass array or materially wrong total is rejected instead.

```c title="src/sample.c"
DlStatus dl_categorical(const double *probabilities, size_t count,
                        double uniform, size_t *token)
{
    if (probabilities == NULL || token == NULL || count == 0 ||
        !isfinite(uniform) || uniform < 0.0 || uniform >= 1.0) return DL_ARGUMENT;
    double sum = 0.0;
    size_t last = count;
    for (size_t i = 0; i < count; ++i) {
        double p = probabilities[i];
        if (!isfinite(p) || p < 0.0 || p > 1.0) return DL_NUMERIC;
        sum += p;
        if (p > 0.0) last = i;
    }
    if (last == count || !isfinite(sum) || fabs(sum - 1.0) > 1e-10) return DL_NUMERIC;
    double target = uniform * sum, cumulative = 0.0;
    for (size_t i = 0; i < count; ++i) {
        cumulative += probabilities[i];
        if (probabilities[i] > 0.0 && target < cumulative) {
            *token = i;
            return DL_OK;
        }
    }
    *token = last; /* Only a rounded endpoint can reach this fallback. */
    return DL_OK;
}

DlStatus dl_sample_next(const float *logits, size_t count, double temperature,
                        DlRng *rng, double *scratch, size_t *token)
{
    if (rng == NULL || token == NULL) return DL_ARGUMENT;
    DlStatus status = dl_sample_probabilities(logits, count, temperature, scratch);
    if (status != DL_OK) return status;
    DlRng next = *rng;
    size_t selected = 0;
    status = dl_categorical(scratch, count, dl_rng_uniform(&next), &selected);
    if (status != DL_OK) return status;
    *rng = next;
    *token = selected;
    return DL_OK;
}
```

`dl_sample_next` first constructs valid probabilities, copies the small value-type RNG, draws once from that copy, and selects. Only then does it commit the RNG and token. Failed temperature, logit, or categorical validation cannot consume the caller's state. Probability scratch is disposable on failure.

A successful sampled token consumes exactly one draw even if numerical rounding concentrates all probability on one ID. A successful greedy token consumes none. The RNG is the existing explicit SplitMix64-based teaching generator, not a source of cryptographic randomness. Its uniform output uses 53 bits, so “uniform in $[0,1)$” means a finite grid, not an infinitely precise real draw. Extremely tiny intervals can contain no representable draw or disappear in cumulative rounding. This is a numerical limitation, not evidence that their exact mathematical probability is zero.

## Check boundaries without statistical luck

The six boundary probes below use probabilities exactly representable in binary. A second test takes 1000 midpoint draws on a regular grid and expects counts 250, 500, and 250. It checks interval arithmetic deterministically; it is not a claim about the statistical quality of a random generator.

Create `tests/test_sample.c`. Its `CHECK` macro returns failure explicitly, so optimized builds still execute every check.

```c title="tests/test_sample.c"
#include "dl/sample.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(yes) do { if (!(yes)) { \
    fprintf(stderr, "FAILED: %s (line %d)\n", #yes, __LINE__); return 0; \
} } while (0)

static int boundaries(void)
{
    const double p[] = {0.0, 0.25, 0.5, 0.25, 0.0};
    const double draws[] = {0.0, nextafter(0.25, 0.0), 0.25,
                            nextafter(0.75, 0.0), 0.75, nextafter(1.0, 0.0)};
    const size_t expected[] = {1, 1, 2, 2, 3, 3};
    size_t token = 99;
    for (size_t i = 0; i < 6; ++i) {
        CHECK(dl_categorical(p, 5, draws[i], &token) == DL_OK);
        CHECK(token == expected[i]);
    }
    /* Deterministic midpoint coverage, not a statistical/flaky RNG assertion. */
    size_t counts[5] = {0};
    for (size_t i = 0; i < 1000; ++i) {
        CHECK(dl_categorical(p, 5, ((double)i + 0.5) / 1000.0, &token) == DL_OK);
        ++counts[token];
    }
    CHECK(counts[0] == 0 && counts[1] == 250 && counts[2] == 500 &&
          counts[3] == 250 && counts[4] == 0);
    const double rounded[] = {0.1, 0.2, 0.6999999999999998, 0.0};
    CHECK(dl_categorical(rounded, 4, nextafter(1.0, 0.0), &token) == DL_OK && token == 2);
    puts("sampling: six half-open boundaries; 1000 midpoint draws give 250/500/250");
    return 1;
}
```

Next check the three hand distributions, an exactly representable common logit shift, a one-token vocabulary, extreme finite logits, the smallest positive double temperature, and the independent greedy tie rule. The shift fixture uses integer float logits so the shift itself does not lose information before entering the sampler.

```c title="tests/test_sample.c"
static int temperatures(void)
{
    const float logits[] = {0.0f, 0.6931471805599453f, 1.3862943611198906f};
    const double temperatures[] = {1.0, 0.5, 2.0};
    const double sqrt2 = sqrt(2.0);
    const double expected[3][3] = {
        {1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0},
        {1.0 / 21.0, 4.0 / 21.0, 16.0 / 21.0},
        {1.0 / (3.0 + sqrt2), sqrt2 / (3.0 + sqrt2), 2.0 / (3.0 + sqrt2)}
    };
    double p[3] = {0}, shifted[3] = {0};
    for (size_t t = 0; t < 3; ++t) {
        CHECK(dl_sample_probabilities(logits, 3, temperatures[t], p) == DL_OK);
        for (size_t i = 0; i < 3; ++i) CHECK(fabs(p[i] - expected[t][i]) < 3e-8);
    }
    const float x[] = {1.0f, 2.0f, 3.0f}, y[] = {101.0f, 102.0f, 103.0f};
    CHECK(dl_sample_probabilities(x, 3, 0.7, p) == DL_OK);
    CHECK(dl_sample_probabilities(y, 3, 0.7, shifted) == DL_OK);
    for (size_t i = 0; i < 3; ++i) CHECK(p[i] == shifted[i]);
    const float extremes[] = {-FLT_MAX, FLT_MAX, FLT_MAX};
    CHECK(dl_sample_probabilities(extremes, 3, DBL_MIN, p) == DL_OK);
    CHECK(p[0] == 0.0 && p[1] == 0.5 && p[2] == 0.5);
    CHECK(dl_sample_probabilities(extremes, 3, DBL_MAX, p) == DL_OK);
    for (size_t i = 0; i < 3; ++i) CHECK(fabs(p[i] - 1.0 / 3.0) < 1e-15);
    size_t token = 99;
    CHECK(dl_greedy(extremes, 3, &token) == DL_OK && token == 1);
    CHECK(dl_sample_probabilities(extremes, 3, DBL_TRUE_MIN, p) == DL_OK);
    CHECK(dl_categorical(p, 3, 0.75, &token) == DL_OK && token == 2);
    const float one[] = {-123.0f};
    CHECK(dl_sample_probabilities(one, 1, 1.0, p) == DL_OK && p[0] == 1.0);
    CHECK(dl_greedy(one, 1, &token) == DL_OK && token == 0);
    puts("sampling: three hand distributions, shift invariance, extreme temperatures, greedy ties");
    return 1;
}
```

Invalid-temperature and invalid-logit tests preserve both the token sentinel and the RNG state. Invalid categorical inputs must not silently normalize a bad distribution into something plausible. The replay test independently obtains a uniform value from a copy of the RNG and compares it with the wrapper's result for 64 calls.

```c title="tests/test_sample.c"
static int failures_and_rng(void)
{
    const float good[] = {0.0f, 1.0f, 2.0f};
    const double bad_temperatures[] = {0.0, -1.0, NAN, INFINITY};
    double scratch[3] = {7.0, 8.0, 9.0};
    size_t token = 99;
    DlRng rng = {42};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(dl_sample_next(good, 3, bad_temperatures[i], &rng, scratch, &token) == DL_ARGUMENT);
        CHECK(rng.state == 42 && token == 99 && scratch[0] == 7.0);
    }
    const float bad[] = {100.0f, NAN, -INFINITY};
    CHECK(dl_greedy(bad, 3, &token) == DL_NUMERIC && token == 99);
    CHECK(dl_sample_next(bad, 3, 1.0, &rng, scratch, &token) == DL_NUMERIC);
    CHECK(rng.state == 42 && token == 99 && scratch[0] == 7.0);
    const double invalid[5][3] = {{0, 0, 0}, {0.2, 0.2, 0.2},
        {-0.1, 0.5, 0.6}, {NAN, 0, 1}, {INFINITY, 0, 0}};
    for (size_t i = 0; i < 5; ++i)
        CHECK(dl_categorical(invalid[i], 3, 0.5, &token) == DL_NUMERIC && token == 99);
    const double p[] = {0.25, 0.5, 0.25};
    const double invalid_draws[] = {-0.1, 1.0, NAN, INFINITY};
    for (size_t i = 0; i < 4; ++i)
        CHECK(dl_categorical(p, 3, invalid_draws[i], &token) == DL_ARGUMENT && token == 99);
    CHECK(dl_greedy(NULL, 3, &token) == DL_ARGUMENT);
    CHECK(dl_greedy(good, 0, &token) == DL_ARGUMENT);
    CHECK(dl_sample_probabilities(good, 3, 1.0, NULL) == DL_ARGUMENT);
    CHECK(dl_categorical(p, 0, 0.0, &token) == DL_ARGUMENT);
    DlRng expected = rng, replay = rng;
    for (size_t i = 0; i < 64; ++i) {
        double draw = dl_rng_uniform(&expected);
        size_t want = 99, again = 99;
        CHECK(dl_sample_probabilities(good, 3, 0.7, scratch) == DL_OK);
        CHECK(dl_categorical(scratch, 3, draw, &want) == DL_OK);
        CHECK(dl_sample_next(good, 3, 0.7, &rng, scratch, &token) == DL_OK);
        CHECK(dl_sample_next(good, 3, 0.7, &replay, scratch, &again) == DL_OK);
        CHECK(token == want && again == token && rng.state == expected.state && replay.state == rng.state);
    }
    puts("sampling: invalid inputs preserve RNG/output; 64 sampled draws replay exactly");
    return 1;
}
```

Finish the test entry point:

```c title="tests/test_sample.c"
int main(void)
{
    if (!boundaries() || !temperatures() || !failures_and_rng()) return EXIT_FAILURE;
    puts("all sampling checks passed");
    return EXIT_SUCCESS;
}

#undef CHECK
```

A reproducible draw is not a model-quality guarantee. It establishes that, on unchanged logits and policy, the same explicit random state takes the same branch. Tiny changes to logits near an interval boundary can select a different ID and then change every later prefix. That is why inference correctness is checked on logits before interpreting long generated strings.

Continue with [Part 3 — Full-prefix generation and executable checks](https://25d-228.github.io/gpt-c/chapter-16-part-3.html). It supplies the complete application and Makefile, the independent training-forward comparisons, checkpoint damage tests, binary-output tests, and an actual trained-model continuation.
