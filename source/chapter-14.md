# Chapter 14 — AdamW, accumulation, and gradient clipping

A derivative answers how the loss changes near the current parameters. It does not decide how several batches become one update, how past derivatives influence that update, or when a failed update is allowed to change state. Those are explicit parts of the training algorithm. A plausible falling loss would not tell us whether we accidentally decayed the shared embedding twice or advanced Adam's clock once per microbatch.

We will implement one optimizer over Chapter 12's **unique** parameter directory. It owns two moment arrays and a token-weighted gradient accumulator. The model continues to perform every forward and backward operation in C. No Python optimizer, automatic differentiation, or external tensor library is introduced.

**Prerequisites:** the complete [Chapter 13 companion snapshot](https://25d-228.github.io/gpt-c/media/chapter-13/reference/Makefile), including Chapter 12's model and Chapter 13's batches. `dl_model_backward(..., DL_SET)` produces a mean derivative for one batch; `DL_ADD` adds another mean without applying a token weight. `DlParameter.decay` is already one for matrix weights and zero for normalization scales. The tied embedding/output table has one owner and one directory entry.

**Implementation estimate:** **4.00 hours** for both parts: 2.50 for state and update logic, and 1.50 for normalization, clipping, scalar references, failure exercises, and model integration. Reading is separate. This part derives and implements the optimizer. [Part 2](https://25d-228.github.io/gpt-c/chapter-14-part-2.html) supplies every test function, the updated build, executable checks, and diagnostic experiments. Both parts are one chapter.

Create `include/dl/optim.h`, `src/optim.c`, and `tests/test_optim.c`; replace the root Makefile with the Part 2 listing. The other **39 files** in Chapter 13 remain unchanged. Filename-labeled blocks for a file concatenate in reading order with **one blank line between blocks**. The [matching 43-file companion snapshot](https://25d-228.github.io/gpt-c/media/chapter-14/reference/Makefile) includes all prerequisite code. We do not change the model's derivatives to implement this chapter.

> **Learning objectives**
> Derive a token-weighted accumulation rule; distinguish update count from microbatch count; implement Adam's two moment recurrences and bias correction; separate decay from the loss gradient; clip one global norm; preserve state when an update fails; and compare a combined batch with equivalent microbatches before and after updating real decoder parameters.

## One update has one objective

Let $\theta\in\mathbb R^P$ denote the vector formed by visiting each unique parameter tensor in directory order and each tensor's dense elements in order. $P$ is the number of scalar parameter values, **not** the number of tensors. Our implementation keeps the tensors where they already live; this vector is a mathematical view and an indexing convention for optimizer storage.

Suppose microbatch $i$ scores $n_i$ target tokens. A microbatch is a batch whose saved activations fit in the available workspace; several such batches can contribute to one parameter update. Define its mean loss and derivative by

$$
L_i(\theta)=\frac{1}{n_i}\sum_{r=0}^{n_i-1}\ell_{i,r}(\theta),
\qquad g_i=\nabla_\theta L_i.
$$

Here $\ell_{i,r}$ is the next-token cross-entropy for target $r$. Chapter 5 already performs the division by the number of targets. Chapter 12 propagates that derivative throughout the model. Repeating another division inside each backward call would be wrong.

The objective for the combined set of targets is

$$
N=\sum_i n_i,\qquad
L=\frac{\sum_i n_iL_i}{N},\qquad
\bar g=\nabla_\theta L=\frac{\sum_i n_i g_i}{N}.
$$

The final equality follows from linearity of differentiation while the weights and the individual examples remain fixed. Thus we store a **token-weighted sum** $S=\sum_i n_i g_i$ and a separate integer $N$. Only the completed group is divided by $N$.

Consider two coordinates spread across different parameter tensors. The first microbatch has two targets and mean derivative $(3,0)$. The second has six targets and mean derivative $(0,4)$. The accumulator and final mean are

$$
S=2(3,0)+6(0,4)=(6,24),\qquad
\bar g=(6,24)/8=(0.75,3).
$$

Averaging the two means equally gives $(1.5,2)$, a different direction. Adding them without dividing gives $(3,4)$, a different magnitude and direction. Neither represents the eight-target mean. The same token weights must also be used when reporting the combined scalar loss.

For the later default setting, each microbatch has $B=2$ independent sequences of $T=128$ targets, so $n_i=256$. Four microbatches produce $N=1024$ targets and $\bar g=(g_1+g_2+g_3+g_4)/4$. That equal-size shortcut is a consequence of the general formula, not the interface's definition.

## Equal targets must also have equal contexts

A combined batch can be split along its **batch dimension** without changing the examples. Part 2 uses three independent length-three sequences: a combined $B=3,T=3$ call is compared with $B=1,T=3$ followed by $B=2,T=3$. These calls score three and six targets respectively. Every sequence still begins with RoPE position zero, uses its original causal prefix, and has the same targets.

Splitting one length-nine sequence into independent length-three pieces is not equivalent. The pieces lose earlier context and reset positions. Keeping nine targets alone does not preserve the objective. Likewise, do not update the weights between contributing microbatches. The derivatives would then be evaluated at different $\theta$ values.

The current decoder has neither dropout nor batch-dependent normalization. Its separate batch items do not attend to one another. These properties make the comparison meaningful. Different float reduction orders still mean that the two implementations should agree within a numerical tolerance, not necessarily bit for bit.

We will use `DL_SET` for **every** model backward in this path. The model gradient arrays are temporary storage for the latest mean. `dl_adamw_add_mean` consumes that mean into an independent double accumulator. Using `DL_ADD` here would feed earlier means into the accumulator again. This does not change Chapter 12's valid `DL_ADD` semantics; it defines a different, explicit accumulation path.

## Clip the completed vector, not its pieces

The Euclidean, or $L_2$, norm of the completed mean is

$$
r=\|\bar g\|_2=\sqrt{\sum_{j=0}^{P-1}\bar g_j^2}.
$$

A positive clipping threshold $c$ defines one coefficient for **all** coordinates:

$$
a=\begin{cases}
1,&r\le c,\\
c/r,&r>c,
\end{cases}
\qquad \widetilde g_j=a\bar g_j.
$$

We additionally define `clip_norm == 0` to disable clipping. It does not mean replace every gradient with zero. The zero-gradient case has $r=0$ and $a=1$, so no division by zero is needed. The norm-clipping rule is described in [Pascanu, Mikolov, and Bengio, Section 3.2][clipping]; the [PyTorch norm-clipping interface][clip-doc] likewise treats all supplied parameter gradients as one vector. We implement the loops ourselves and state our exact threshold rule rather than claiming bitwise compatibility with a library's epsilon choices.

For coordinates $(3,4)$, $r=5$. With $c=2$, $a=0.4$ and $\widetilde g=(1.2,1.6)$. Clipping the two tensors separately would yield $(2,2)$ instead. Clipping each scalar to an interval is also different from scaling a vector. Both mistakes can pass a test that only examines one nonzero coordinate.

Clipping is nonlinear, so it must follow accumulation. Take two equal-sized microbatches with means $(100,0)$ and $(-100,2)$. Their combined mean is $(0,1)$ and needs no clipping at threshold two. Clipping the individual means first and then averaging instead produces approximately $(0.000200,0.019996)$. Large opposing components should have cancelled before the norm decision.

The shared embedding/output table appears **once** in this norm, but its gradient already contains both paths from Chapter 12. Counting its tensor twice changes the clipping coefficient for every parameter, even before we reach weight decay.

## Two memories of the clipped gradient

Let $t$ count completed optimizer updates, beginning with $t=1$ for the first successful step. For each coordinate $j$, Adam maintains a first moment $m_{t,j}$ and a second **raw** moment $v_{t,j}$:

$$
m_{t,j}=\beta_1m_{t-1,j}+(1-\beta_1)\widetilde g_{t,j},
$$

$$
v_{t,j}=\beta_2v_{t-1,j}+(1-\beta_2)\widetilde g_{t,j}^2.
$$

Both arrays initially contain zero. The coefficients satisfy $0\le\beta_1,\beta_2<1$. The first array smooths signed derivatives; the second smooths their squared magnitudes. $v$ is not a centered variance: the algorithm does not subtract the square of a mean.

Unroll the first recurrence from zero:

$$
m_t=(1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}\widetilde g_k.
$$

For a constant gradient $q$, the geometric sum gives $m_t=q(1-\beta_1^t)$. Its small initial magnitude partly reflects the artificial zero initialization. Dividing by $1-\beta_1^t$ removes that factor. Applying the same argument to the squared gradient gives

$$
\widehat m_{t,j}=\frac{m_{t,j}}{1-\beta_1^t},\qquad
\widehat v_{t,j}=\frac{v_{t,j}}{1-\beta_2^t}.
$$

This is an initialization correction, not a claim that a nonstationary training gradient is perfectly estimated. The bias-correction discussion in [Kingma and Ba, Section 3][adam] supplies the original motivation; [AdamW Algorithm 2][adamw] gives the constant-beta recurrences used here. We do not implement a decaying beta, AMSGrad, or an optimizer family-selection framework.

The adaptive contribution to the parameter step is

$$
\eta_t\frac{\widehat m_{t,j}}{\sqrt{\widehat v_{t,j}}+\epsilon}.
$$

Here $\eta_t$ is the caller-supplied learning rate and $\epsilon>0$ is an optimizer denominator constant. It is outside the square root and is unrelated to the model's RMSNorm epsilon. Moving it inside the root changes the algorithm.

If a new gradient is zero but an old moment is nonzero, the recurrences still age that history. A zero gradient does **not** imply a zero adaptive step. Likewise, clipping the loss gradient does not guarantee that the parameter displacement has norm at most $c$: the moment history, coordinatewise denominator, learning rate, and separate decay all intervene. On a first update with negligible epsilon, rescaling all gradients can even leave the adaptive weight step almost unchanged. Tests must inspect moments and clipping statistics, not only the new weights.

## Keep weight decay out of both moments

Let $d_j$ be one for a matrix parameter and zero for a normalization scale. With decay coefficient $\lambda\ge0$, this implementation uses

$$
\theta_{t,j}=(1-\eta_t\lambda d_j)\theta_{t-1,j}
-\eta_t\frac{\widehat m_{t,j}}{\sqrt{\widehat v_{t,j}}+\epsilon}.
$$

Only the clipped **loss** gradient enters the moment recurrences. We never replace it by $\widetilde g+\lambda\theta$. That alternative is an $L_2$ penalty differentiated through the adaptive optimizer and is not generally the same update. [Loshchilov and Hutter, Section 2][adamw] explain this distinction.

The learning-rate convention matters when reproducing formulas. We use shrink factor `1 - learning_rate * weight_decay`, matching the convention displayed in the [AdamW API equations][adamw-doc]. The original paper writes a base rate $\alpha$, a schedule multiplier $s_t$, and decay $\lambda_{\rm paper}$, with gradient rate $s_t\alpha$ but shrink $1-s_t\lambda_{\rm paper}$. To obtain our rule when $\eta_t=s_t\alpha$, choose $\lambda_{\rm paper}=\alpha\lambda$. Copying the same numerical decay coefficient between those notations without this mapping changes the shrinkage.

The directory already decides decay membership. The table, query/key/value/output matrices, and gate/up/down matrices decay. All RMSNorm scale tensors do not. We make this a directory property rather than infer it from a tensor's shape: a small test matrix could have one row, which does not make it a normalization scale.

For this laboratory, require $0\le\eta_t\lambda\le1$. This prevents a negative shrink factor. It is an explicit interface restriction, not a universal definition of AdamW. There is one learning rate per call, no per-tensor rate group, and no separate output-table update. Tying changes the derivative contribution, not the number of optimizer visits.

## Calculate two updates before coding them

Use one matrix coordinate with initial weight two, gradient sequence $2,-1$, $\beta_1=1/2$, $\beta_2=1/4$, $\epsilon=0.1$, learning rate $0.2$, and decay $\lambda=0.5$. Disable clipping for this calculation. The shrink factor is $1-0.2\times0.5=0.9$.

| Quantity | Update 1 | Update 2 |
|---|---:|---:|
| Incoming mean gradient | 2 | -1 |
| First moment | $1$ | $(1/2)1+(1/2)(-1)=0$ |
| Second raw moment | $(3/4)4=3$ | $(1/4)3+(3/4)1=1.5$ |
| First correction denominator | $1-1/2=0.5$ | $1-1/4=0.75$ |
| Second correction denominator | $1-1/4=0.75$ | $1-1/16=0.9375$ |
| Corrected first moment | 2 | 0 |
| Corrected second moment | 4 | 1.6 |

For update one,

$$
\theta_1=0.9(2)-0.2\frac{2}{2+0.1}
=1.6095238095\ldots
$$

For update two the signed moment cancels, so the adaptive term is zero, but decay still applies:

$$
\theta_2=0.9\theta_1=1.4485714286\ldots
$$

A norm-scale coordinate starting at two with the same gradients has the same moments, but its first weight becomes $1.8095238095\ldots$ and stays there on update two. These are mathematical decimal expectations; float storage rounds them. Part 2 checks both kinds of parameter, extends the sequence to four updates with changing learning rates, and compares against a separately expressed scalar recurrence.

## The update clock is not the data clock

The optimizer stores `updates`, the number of completed weight updates. The next successful step uses `updates + 1` in its correction denominators. `tokens` counts only the current accumulation group. Ten microbatches could still constitute one update; ten rejected step attempts constitute zero updates.

A caller should select its learning rate for the next update from `updates + 1`. This chapter passes explicit rates, including a changing sequence in the tests. Chapter 15 will implement the actual warmup/decay schedule. There is no hidden schedule counter here and no counter increment when a batch is merely read or differentiated.

A failed step leaves `updates` unchanged. Retrying with a corrected positive rate must use the same next-update index. A successful step invalidates old model tapes by advancing the model revision. That revision can already include initialization-time test edits, so it is **not** an alternative Adam update counter.

## Lay out optimizer state without duplicating parameters

For each directory tensor, let its first flat coordinate be the sum of preceding tensor sizes. If the tied table has $VD$ elements, its moments occupy indices zero through $VD-1$. The first block's attention-norm moments immediately follow. The optimizer uses the same traversal in accumulation, candidate checks, and committing writes.

`moments` owns $2P$ floats. Coordinates `[0,P)` hold $m$ and `[P,2P)` hold $v$. `sum` separately owns $P$ doubles. `model` is borrowed: it must retain its address and outlive the optimizer. No weight or parameter-gradient allocation is duplicated. The optimizer is not allowed to outlive, move, or silently reattach to its model.

On the target's four-byte floats and eight-byte doubles, this is $16P$ bytes of optimizer array payload. For the planned byte model, $P=2{,}885{,}888$, giving 46,174,208 bytes, about **44.04 MiB**. Weights and model gradients add another $8P$ bytes, making those arrays plus optimizer state about **66.05 MiB**, excluding the training tape, file data, metadata, allocator overhead, and process runtime. These are layout calculations, not measured resident memory. Accumulation costs one reusable parameter-sized double array instead of retaining four activation tapes.

Float moments are an intentional storage choice. Recurrences and candidate weights use double arithmetic, but each new moment is rounded to float **before** bias correction. Consequently the update uses the state that will actually be saved, rather than silently using extra bits that disappear before the next update. This is our specified reference behavior, not a promise of bitwise equivalence to a fused optimizer kernel.

Create the types first:


```c title="include/dl/optim.h"
#ifndef DL_OPTIM_H
#define DL_OPTIM_H

#include "dl/model.h"

/* Counts up to 2^53 are exactly representable as binary64 integers. */
#define DL_OPT_MAX_COUNT UINT64_C(9007199254740992)

typedef struct {
    double beta1, beta2, epsilon, weight_decay, clip_norm;
} DlAdamWConfig;

typedef struct {
    DlModel *model;                  /* Borrow: stable address, outlives us. */
    DlAdamWConfig config;
    float *moments;                  /* Owner: [first moment P][second P]. */
    double *sum;                    /* Owner: P token-weighted gradients. */
    size_t count;
    uint64_t updates, tokens, revision;
    int active;
} DlAdamW;

typedef struct {
    uint64_t update, tokens;
    double learning_rate, grad_norm, clip_scale;
} DlOptimStats;
```

`uint64_t` and `UINT64_C` are available through the model's RNG header. `DL_OPT_MAX_COUNT` is $2^{53}$, the largest integer bound up to which binary64 represents every integer exactly. We cap both a group's token count and the update count used in double expressions there. This does not make gradient summation exact; it only avoids introducing a rounded integer denominator or exponent by accepting an arbitrarily large count.

The pointers are owners, unlike the borrowed matrix views used by kernels. `dl_adamw_destroy` frees only these two allocations. Do not copy a live `DlAdamW` by assignment: that would copy owner pointers, not clone their allocations. The candidate assignment in the constructor is safe because ownership is being transferred from a local candidate into an empty caller object.

The public calls specify the lifecycle and failure boundaries:


```c title="include/dl/optim.h"
/* Config: finite 0<=beta<1, epsilon>0, decay>=0, clip>=0. clip==0 disables
 * clipping. Empty {0} owner only. Budget covers 2*P floats + P doubles;
 * metadata/allocator overhead excluded. Failure leaves out unchanged.
 * Never copy a live owner, edit metadata/config, or move its borrowed model.
 * Constructed model metadata remains immutable. No concurrent access. */
DlStatus dl_adamw_create(DlAdamW *out, DlModel *model, DlAdamWConfig config,
                         size_t max_state_bytes);
void dl_adamw_destroy(DlAdamW *optim);

/* Start/restart one update group: zero sum and model gradients, tokens=0,
 * gradients_valid=0. Retain moments and completed update count. No weight
 * change. A previous partial group is deliberately discarded. An outside
 * weight edit invalidates this optimizer; destroy/recreate rather than
 * silently attaching old moments to externally changed parameters. */
DlStatus dl_adamw_begin(DlAdamW *optim);

/* Consume a FRESH single-microbatch mean from model.grad, normally produced
 * by backward(..., DL_SET); tokens is its actual number of scored targets.
 * Accumulate tokens*grad in double. Never feed a sum of batch means (ADD).
 * Success marks model gradients consumed (gradients_valid=0); bytes remain.
 * Failure changes neither sum/count nor model gradients/validity. Combined
 * tokens <= 2^53. A failed backward requires begin/recompute the whole group. */
DlStatus dl_adamw_add_mean(DlAdamW *optim, uint64_t tokens);

/* Needs an active nonempty group and no unconsumed model gradients.
 * finite lr>0; 0<=lr*decay<=1 (nonnegative shrink factor).
 * Form the token mean, clip its ONE global norm, then update unique params.
 * Moment storage is float; recurrences and candidate weights use double.
 * Every candidate is checked before ANY weight/moment/count/revision write.
 * Failure leaves the complete call state and stats unchanged. A successful
 * call increments updates/revision, consumes the group, invalidates old tapes
 * and model gradients; sum/grad bytes are cleared by the NEXT begin, not here.
 * stats is a separate caller-owned object, disjoint from all optimizer/model
 * storage. No use of model SGD or external weight edits while attached. */
DlStatus dl_adamw_step(DlAdamW *optim, double learning_rate, DlOptimStats *stats);

#endif
```

There are three useful states. Immediately after creation or a successful step, the optimizer is inactive. `begin` makes it active with zero accepted tokens. Successful `add_mean` calls make it active and nonempty. Only that last state may step.

| Operation | Accumulator | Model gradient validity | Moments and weights |
|---|---|---|---|
| `create` | Allocated, zero, inactive | Unchanged | Moments zero; weights unchanged |
| `begin` | Zeroed, active, count zero | Cleared bytes, marked not fresh | Retained |
| `backward(..., DL_SET)` | Unchanged | One fresh microbatch mean | Unchanged |
| `add_mean` | Add weighted mean and count | Marked consumed; bytes remain | Unchanged |
| Successful `step` | Inactive, count zero; bytes remain | Invalid; old tapes are stale | Commit once |
| Rejected `add_mean` or `step` | Unchanged by that call | Unchanged by that call | Unchanged by that call |

`begin` deliberately discards a partial group but does not reset the moment history. Clearing moment history every update reduces the algorithm to repeated first-step behavior. Conversely, failing to clear `sum` at the next group mixes derivatives evaluated at different weights.

The model's `gradients_valid` flag guards accidental reuse: a successful add marks its input consumed, so calling add twice without another valid derivative is rejected. It cannot prove that a caller supplied the correct target count or used `DL_SET` rather than `DL_ADD`. Those are mathematical preconditions. The model is still ordinary C, not a provenance-tracking automatic differentiation system.

## Validate ownership before beginning arithmetic

Start `src/optim.c` with imports, the existing early-return idiom, configuration validation, and an attachment check:


```c title="src/optim.c"
#include "dl/optim.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

static int config_valid(DlAdamWConfig c)
{
    return isfinite(c.beta1) && c.beta1 >= 0.0 && c.beta1 < 1.0 &&
           isfinite(c.beta2) && c.beta2 >= 0.0 && c.beta2 < 1.0 &&
           isfinite(c.epsilon) && c.epsilon > 0.0 &&
           isfinite(c.weight_decay) && c.weight_decay >= 0.0 &&
           isfinite(c.clip_norm) && c.clip_norm >= 0.0;
}

static DlStatus attached(const DlAdamW *o)
{
    if (o == NULL || o->model == NULL || o->model->params == NULL ||
        o->moments == NULL || o->sum == NULL || o->count == 0)
        return DL_ARGUMENT;
    if (o->revision != o->model->revision || o->count != o->model->value_count)
        return DL_STATE;
    return DL_OK;
}
```

The configuration check excludes NaN before an inequality could accidentally let it pass. The model's directory and tensor geometry are already validated by its constructor and remain immutable. `attached` checks the owner relationship and revision, not whether arbitrary forged pointers secretly reference live allocations.

An external legal weight edit must call `dl_model_mark_changed` first. That makes this optimizer's saved revision stale. We reject further use rather than silently apply old moments to unrelated weights. For an intentional restart from edited weights, destroy and recreate the optimizer; that explicitly resets its history. A later checkpoint loader must restore a coherent model and optimizer together, not bypass this rule with raw edits.

Creation checks the full allocation budget before allocating either array. It builds a candidate, frees whichever allocation succeeded if the other fails, and only then exposes a complete owner:


```c title="src/optim.c"
void dl_adamw_destroy(DlAdamW *o)
{
    if (o == NULL) return;
    free(o->sum);
    free(o->moments);
    *o = (DlAdamW){0};
}

DlStatus dl_adamw_create(DlAdamW *out, DlModel *model, DlAdamWConfig config,
                         size_t max_state_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL ||
        model->value_count == 0 || !config_valid(config) || max_state_bytes == 0)
        return DL_ARGUMENT;
    if (out->model != NULL || out->moments != NULL || out->sum != NULL ||
        out->count != 0 || out->updates != 0 || out->tokens != 0 || out->active)
        return DL_STATE;
    size_t two_p = 0, moment_bytes = 0, sum_bytes = 0, total = 0;
    RETURN_IF_ERROR(dl_size_mul(model->value_count, 2, &two_p));
    RETURN_IF_ERROR(dl_size_mul(two_p, sizeof(float), &moment_bytes));
    RETURN_IF_ERROR(dl_size_mul(model->value_count, sizeof(double), &sum_bytes));
    RETURN_IF_ERROR(dl_size_add(moment_bytes, sum_bytes, &total));
    if (total > max_state_bytes) return DL_RANGE;
    DlAdamW candidate = {0};
    candidate.model = model;
    candidate.config = config;
    candidate.count = model->value_count;
    candidate.revision = model->revision;
    candidate.moments = calloc(two_p, sizeof(float));
    candidate.sum = calloc(candidate.count, sizeof(double));
    if (candidate.moments == NULL || candidate.sum == NULL) {
        dl_adamw_destroy(&candidate);
        return DL_ALLOC;
    }
    *out = candidate;
    return DL_OK;
}
```

`two_p` counts floats, while `moment_bytes`, `sum_bytes`, and `total` count bytes. Keeping those names separate prevents a budget check in elements followed by an allocation in bytes. Every multiplication and addition uses Chapter 2's checked size helpers. The budget excludes small struct and allocator overhead; it is not a process-memory limiter.

The two `calloc` calls produce zero initial moments and a zero initial accumulator on the supported floating-point targets. Destroying a partly constructed local candidate is valid: `free(NULL)` does nothing. The caller remains unchanged until `*out = candidate`. No globals, optimizer singleton, or hidden model pointer is created.

A new accumulation boundary physically clears the model's temporary gradients as well as the double sum:


```c title="src/optim.c"
DlStatus dl_adamw_begin(DlAdamW *o)
{
    RETURN_IF_ERROR(attached(o));
    RETURN_IF_ERROR(dl_model_zero_grad(o->model));
    for (size_t i = 0; i < o->count; ++i) o->sum[i] = 0.0;
    o->tokens = 0;
    o->active = 1;
    o->model->gradients_valid = 0; /* Cleared storage is not a fresh backward. */
    return DL_OK;
}
```

`dl_model_zero_grad` normally marks zero gradients valid for an ADD-based model path. We immediately mark them not fresh for **this** consumption protocol. The next complete `DL_SET` backward overwrites all of them and restores validity. `begin` does not touch a moment, update counter, or model weight. It is also the recovery operation when we deliberately abandon a partial accumulation group.

## Multiply each mean by its own token count

The add operation has two passes. The first calculates every prospective double sum and validates it without storing. The second repeats that same arithmetic and commits. This is an in-memory all-or-nothing operation under the single-threaded ownership contract.


```c title="src/optim.c"
DlStatus dl_adamw_add_mean(DlAdamW *o, uint64_t tokens)
{
    RETURN_IF_ERROR(attached(o));
    if (!o->active || !o->model->gradients_valid) return DL_STATE;
    if (tokens == 0) return DL_ARGUMENT;
    if (o->tokens > DL_OPT_MAX_COUNT || tokens > DL_OPT_MAX_COUNT - o->tokens)
        return DL_OVERFLOW;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlTensor *g = &o->model->params[p].grad;
        for (size_t i = 0; i < g->count; ++i, ++at) {
            double next = o->sum[at] + (double)tokens * (double)g->data[i];
            if (!isfinite(g->data[i]) || !isfinite(o->sum[at]) || !isfinite(next))
                return DL_NUMERIC;
        }
    }
    at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlTensor *g = &o->model->params[p].grad;
        for (size_t i = 0; i < g->count; ++i, ++at)
            o->sum[at] += (double)tokens * (double)g->data[i];
    }
    o->tokens += tokens;
    o->model->gradients_valid = 0;
    return DL_OK;
}
```

In the unequal example, `tokens` is two for the first call and six for the second; the appropriate coordinates finish at six and twenty-four. `at` advances across all tensors without resetting at each tensor boundary. Resetting it inside the outer loop would make unrelated parameter families share accumulator cells.

The two casts before multiplication are important. Token counts and derivatives enter double arithmetic before the product is formed. We do not multiply a float by a token count in float and then widen an already overflowed result. The subtraction-based count check prevents unsigned wrap before `o->tokens += tokens`.

An invalid gradient at the very last coordinate must not add earlier coordinates or consume its token count. That is why finite checking cannot be interleaved with committed stores. The second pass is safe because no call between the passes changes the gradients or sum, and the interface forbids concurrent mutation.

A failed **model backward** has a different contract: it may have partly written the model gradients. Do not call add on that invalid result, and do not step using only the earlier accepted microbatches as though the full planned objective had succeeded. Restart with `begin` and recompute the intended group. Restarting this accumulator does not rewind Chapter 13's data RNG; retain/replay the intended batch IDs or restore the caller's saved sampling state when that is the desired recovery behavior.

## Accumulate a norm without squaring in float

A finite float derivative can have a square outside float range. For example, `FLT_MAX * FLT_MAX` evaluated as float overflows before an assignment to double could help. We operate on double means and use a scaled sum of squares, which also avoids needlessly squaring very small unscaled values.

Maintain nonnegative `scale` and `ssq` such that the sum of squares seen so far equals $\text{scale}^2\text{ssq}$. For a new absolute coordinate $x$ larger than `scale`, rewrite the old sum in units of $x^2$:

$$
\text{ssq}'=1+\text{ssq}(\text{scale}/x)^2,
\qquad \text{scale}'=x.
$$

Otherwise add $(x/\text{scale})^2$ to `ssq`. The ratios are at most one. A zero coordinate contributes nothing. With inputs three then four, the state is first `(scale,ssq)=(3,1)`, then `(4,1+9/16)`, and the final norm is $4\sqrt{25/16}=5$.


```c title="src/optim.c"
/* Scaled sum of squares: norm^2 = scale^2 * ssq. No float square. */
static DlStatus mean_norm(const DlAdamW *o, double *out)
{
    double scale = 0.0, ssq = 1.0;
    for (size_t i = 0; i < o->count; ++i) {
        double a = fabs(o->sum[i] / (double)o->tokens);
        if (!isfinite(a)) return DL_NUMERIC;
        if (a == 0.0) continue;
        if (scale < a) {
            double ratio = scale / a;
            ssq = 1.0 + ssq * ratio * ratio;
            scale = a;
        } else {
            double ratio = a / scale;
            ssq += ratio * ratio;
        }
    }
    double norm = scale * sqrt(ssq);
    if (!isfinite(norm)) return DL_NUMERIC;
    *out = norm;
    return DL_OK;
}
```

The entire vector uses a single `scale` and `ssq`; they are not restarted for each parameter tensor. The final norm is returned as double and may legitimately exceed `FLT_MAX`. A non-finite mean or final norm is rejected before clipping. In particular, do not take an infinite norm, obtain a zero scale, and assume multiplying an infinite gradient by zero has repaired it.

For two `FLT_MAX` gradients with threshold one, the finite double norm is approximately $\sqrt2\,\mathrm{FLT\_MAX}$. Each clipped coordinate is about $1/\sqrt2$. The small positive clipping coefficient stays in double through multiplication. A later moment can then fit in float. Without clipping, this same finite input must be rejected if its second moment cannot be represented in the chosen float storage.

## Compute a candidate without changing live state

For beta close to one and a small update count, `1 - pow(beta, update)` subtracts two nearby numbers. We instead evaluate the same expression through

$$
1-\beta^t=-\operatorname{expm1}(t\log\beta),
$$

where `expm1(x)` computes $e^x-1$ without first rounding $e^x$ and then subtracting one. Apple's archived [math-library reference][expm1] documents this small-argument behavior; it is a longstanding C function, not a new macOS-version requirement. Beta zero is handled directly, without calling `log(0)`. We calculate each denominator once per step, not per scalar parameter. The small `Candidate` struct stores just one coordinate's prospective weight and moments on the stack.


```c title="src/optim.c"
/* Unlike 1-pow(beta,t), this does not subtract two nearly equal numbers. */
static double correction(double beta, uint64_t update)
{
    return beta == 0.0 ? 1.0 : -expm1((double)update * log(beta));
}

typedef struct {
    float weight, first, second;
} Candidate;
```

Now create that candidate. The recurrence uses the clipped token mean, not the sum and not the previous microbatch's temporary gradient. `at` addresses the first moment, while `count + at` addresses its matching second moment.


```c title="src/optim.c"
static DlStatus candidate_at(const DlAdamW *o, size_t at, double weight,
                              int decay, double lr, double clip,
                              double correction1, double correction2,
                              Candidate *out)
{
    double old_m = o->moments[at];
    double old_v = o->moments[o->count + at];
    double g = (o->sum[at] / (double)o->tokens) * clip;
    if (!isfinite(weight) || !isfinite(old_m) || !isfinite(old_v) ||
        old_v < 0.0 || !isfinite(g)) return DL_NUMERIC;
    double m = o->config.beta1 * old_m + (1.0 - o->config.beta1) * g;
    double v = o->config.beta2 * old_v + (1.0 - o->config.beta2) * g * g;
    if (!isfinite(m) || fabs(m) > FLT_MAX || !isfinite(v) || v > FLT_MAX)
        return DL_NUMERIC;
    /* Bias-correct the moments that will actually be stored, not extra bits. */
    float stored_m = (float)m, stored_v = (float)v;
    double mhat = (double)stored_m / correction1;
    double vhat = (double)stored_v / correction2;
    double adaptive = mhat / (sqrt(vhat) + o->config.epsilon);
    double shrink = decay ? 1.0 - lr * o->config.weight_decay : 1.0;
    double next = weight * shrink - lr * adaptive;
    if (!isfinite(mhat) || !isfinite(vhat) || !isfinite(adaptive) ||
        !isfinite(next) || fabs(next) > FLT_MAX) return DL_NUMERIC;
    *out = (Candidate){(float)next, stored_m, stored_v};
    return DL_OK;
}
```

The order is deliberate. First validate the old weight and moments. A second moment cannot be negative; accepting a corrupt negative value would later send an invalid argument to `sqrt`. Next compute new moments in double and check their float range. Then round them, bias-correct those stored values, calculate the adaptive term, and finally combine it with decay from the **old** weight.

Using `weight * shrink - learning_rate * adaptive` avoids first storing a rounded decayed weight and then rounding again after the adaptive subtraction. It also makes it obvious that decay is absent from both moment recurrences. The only output assignment occurs after all checks for this coordinate succeed.

Float underflow is not automatically treated as failure. Very small moments can round to zero; this is one limitation of float optimizer state. Positive epsilon protects a zero denominator, but an extremely small epsilon can still make a prospective weight exceed float range. The explicit finite/range checks reject that candidate rather than clamp it or silently reset the optimizer. A numerically valid zero step is not proof that a chosen epsilon or learning rate is useful for training.

## Commit only after checking the last coordinate

The step first validates state, count limits, rate, and the allowed decay factor. It forms the mean norm and one clipping coefficient, then checks **every** candidate. No parameter, moment, clock, revision, or statistics object has changed yet.

Only after that complete pass does it invalidate old tapes and perform the same per-coordinate calculations again for storage. Recomputing avoids allocating three additional model-sized candidate arrays. This is a compute-versus-memory choice at the update boundary, not a performance claim.


```c title="src/optim.c"
DlStatus dl_adamw_step(DlAdamW *o, double learning_rate, DlOptimStats *stats)
{
    RETURN_IF_ERROR(attached(o));
    if (stats == NULL || !isfinite(learning_rate) || learning_rate <= 0.0)
        return DL_ARGUMENT;
    if (!o->active || o->tokens == 0 || o->model->gradients_valid)
        return DL_STATE;
    if (o->updates >= DL_OPT_MAX_COUNT || o->tokens > DL_OPT_MAX_COUNT ||
        o->model->revision == UINT64_MAX) return DL_OVERFLOW;
    double decay_step = learning_rate * o->config.weight_decay;
    if (!isfinite(decay_step) || decay_step > 1.0) return DL_RANGE;
    uint64_t update = o->updates + 1;
    double c1 = correction(o->config.beta1, update);
    double c2 = correction(o->config.beta2, update);
    if (!isfinite(c1) || c1 <= 0.0 || !isfinite(c2) || c2 <= 0.0)
        return DL_NUMERIC;
    double norm = 0.0;
    RETURN_IF_ERROR(mean_norm(o, &norm));
    double clip = 1.0;
    if (o->config.clip_norm > 0.0 && norm > o->config.clip_norm)
        clip = o->config.clip_norm / norm;
    size_t at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        const DlParameter *a = &o->model->params[p];
        for (size_t i = 0; i < a->value.count; ++i, ++at) {
            Candidate trial;
            RETURN_IF_ERROR(candidate_at(o, at, a->value.data[i], a->decay,
                learning_rate, clip, c1, c2, &trial));
        }
    }
    /* Single-threaded: repeating unchanged inputs cannot acquire a new error.
     * Recompute instead of allocating a second model-sized candidate buffer. */
    RETURN_IF_ERROR(dl_model_mark_changed(o->model));
    at = 0;
    for (size_t p = 0; p < o->model->param_count; ++p) {
        DlParameter *a = &o->model->params[p];
        for (size_t i = 0; i < a->value.count; ++i, ++at) {
            Candidate next = {0};
            (void)candidate_at(o, at, a->value.data[i], a->decay,
                              learning_rate, clip, c1, c2, &next);
            a->value.data[i] = next.weight;
            o->moments[at] = next.first;
            o->moments[o->count + at] = next.second;
        }
    }
    *stats = (DlOptimStats){update, o->tokens, learning_rate, norm, clip};
    o->updates = update;
    o->revision = o->model->revision;
    o->tokens = 0;
    o->active = 0;
    return DL_OK;
}

#undef RETURN_IF_ERROR
```

Why is the ignored result in the commit pass justified? `candidate_at` depends only on its own old weight and two old moment coordinates, the unchanged sum, configuration, and locally computed constants. Earlier committed coordinates are not among those inputs. There are no allocations, user callbacks, or concurrent changes between the passes, and the caller does not change floating-point behavior during a call. The second evaluation therefore cannot introduce an input that failed to pass the first evaluation. If this were a concurrent optimizer or one with cross-coordinate updates, that reasoning would no longer be sufficient.

`dl_model_mark_changed` is called before the first weight write and after the exhaustion check. Under the validated contract, it succeeds, increments the revision, and invalidates model gradients. Tapes from the old revision cannot then be differentiated against new weights. This is an ordinary function-call guarantee, not crash recovery: abrupt process termination halfway through memory stores is not recoverable without the later checkpoint mechanism.

Statistics commit only at the end. `grad_norm` reports the norm **before** clipping; `clip_scale` reports the coefficient actually used. `tokens` reports the whole consumed group even though the live accumulator count is reset to zero. These numbers let a future trainer distinguish a large raw derivative from a small clipped derivative without recomputing either.

We do not clear the large sum and gradient arrays again at success. They are logically invalid/inactive, and the next `begin` clears them. In particular, a second `step` cannot accidentally reuse their stale bytes. Checkpointing at completed update boundaries will not require serializing those discarded partial-gradient buffers; it will require weights, moments, the completed-update counter, configuration, and the relevant caller-owned schedule/data state.

## Follow the ownership of one actual group

After `begin`, no mean is pending. A successful forward and `DL_SET` backward place one fresh mean in the model gradient tensors. Add consumes it with that tape's actual number of rows. The tape and model gradients can then be reused for the next microbatch while the double sum retains the accepted contribution. Weights and moments remain fixed throughout that process.

After the last add, the model gradient flag is false, and `step` can consume the group. Calling step while a fresh model gradient is still unconsumed is rejected: it would otherwise silently omit the most recent backward result. This flag is not a complete transaction coordinator; callers must still propagate each returned status and stop on a failed forward/backward.

The complete integration function in [Part 2](https://25d-228.github.io/gpt-c/chapter-14-part-2.html) implements that order twice, once with a combined batch and once with microbatches. It compares losses and all gradient coordinates before updating, then compares all weights and both moment arrays afterward. A deliberately pending forward is also checked after the update to prove that the model revision protection still works.

## Technical references

[AdamW, Section 2 and Algorithm 2][adamw] defines decoupling and the moment/correction structure. [Adam, Section 3][adam] explains the initialization correction. [The clipping paper, Section 3.2 and Algorithm 1][clipping] defines vector norm scaling. The [AdamW API equations][adamw-doc] and [global norm API][clip-doc] make useful convention comparisons; they are documentation references, not runtime dependencies. All model and optimizer computation here remains the C code shown in the chapter.

[adamw]: https://arxiv.org/html/1711.05101v3
[adam]: https://arxiv.org/html/1412.6980v5#S3
[clipping]: https://arxiv.org/html/1211.5063v2#S3.SS2
[adamw-doc]: https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html
[clip-doc]: https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html

[expm1]: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/expm1.3.html
