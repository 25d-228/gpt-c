# Chapter 19 · Part 2 — A linear training tape and paired dispatch

This continues [the tiled backward derivation](https://25d-228.github.io/gpt-c/chapter-19.html). Start from the complete Chapter 18 implementation and the replacements in Part 1. The materialized path remains available and numerically unchanged.

## Do not confuse inference reuse with a training tape

An inference workspace can overwrite the first block's intermediates while evaluating the second. Training cannot: the backward pass visits blocks in reverse order and needs each block's successful forward state. We therefore add `DlTiledWorkspace *tiled`, an owned array of **one owner per model block**, to `DlTrainTape`. Each workspace contains that block's three statistic planes and bounded forward scratch. Keeping bounded forward scratch per block is simpler than introducing additional borrowed-workspace types; it is counted rather than hidden.

The one `tiled_backward` owner is shared across reverse visits. The existing float views retain Q0, K0, V, rotated Q/K, and context. The square score/probability views are left as zero-initialized empty records. The same omission applies to the square gradient scratch views. Dispatch never passes those empty views into a materialized operator.

The model still borrows one fixed tape, copies token IDs and targets into owned metadata, marks a successful forward ready, consumes it once in backward, and rejects stale model revisions. A failed tiled backward invalidates **all** model gradients, just like a failed materialized backward. Calling the plain update or optimizer on those invalid gradients remains prohibited.

## Count the arrays before creating them

Let $L$ be decoder blocks, $M=BT$ token rows, $D$ model width, $K=H_{kv}d$ key/value width, $F$ feed-forward width, and $V$ vocabulary size. In this model $D=H_qd$ even though the isolated sublayer permits another input width.

The tiled float arena contains

$$
A_{\mathrm{linear}}=(L+1)M(9D+3K+3F)+2MV
$$

floats. To verify rather than memorize the expression, count hidden states: $(L+1)MD$. Each block saves five model-width views, three query-width views, three key/value-width views, and three feed-forward-width views: $8MD+3MK+3MF$. Outside blocks there is final normalization, four model-gradient work views, the six linear attention-gradient views, three feed-forward-gradient views, and two vocabulary-width matrices. `embedding_grad` is an alias to the completed model gradient, not another allocation.

The materialized arena has the same linear payload plus

$$
A_{\mathrm{square}}=2(L+1)BH_qT^2.
$$

There are two square forward arrays per block and two shared square backward arrays. The tiled constructor must skip those placements entirely; allocating them and merely declining to use them would not implement the intended memory tradeoff.

Let $N=BH_qT$ be the number of query-head statistic rows. The tiled double payload is

$$
D_{\mathrm{tiled}}=L(3N+RS+Rd+2R)+(2RS+Rd+2Sd+R).
$$

The first term owns all per-block forward statistics and scratch; the second is the one backward workspace from Part 1. On a binary32/binary64 machine the numerical payload is $4A_{\mathrm{linear}}+8D_{\mathrm{tiled}}$ bytes. This count excludes parameters, optimizer state, ID/target/position arrays, frequency arrays, view records, allocator metadata, stack, and process/runtime overhead. Those are not zero merely because they are absent from this particular budget.

At fixed model and tile dimensions, every sequence-length-dependent term is linear in $T$. Increasing $R$ and $S$ with sequence length can defeat that claim; these constructor arguments are deliberately fixed during the comparisons.

## Extend the model's owner records

Replace `include/dl/model.h`. The old constructor remains materialized and keeps its previous float-arena budget. The new constructor's budget covers **float plus double numerical payload**. `dl_train_tape_payload_bytes` reports that same quantity for an already constructed tape; it does not query resident memory.

```c title="include/dl/model.h"
#ifndef DL_MODEL_H
#define DL_MODEL_H

#include "dl/attention_tiled.h"
#include "dl/ffn.h"
#include "dl/loss.h"
#include "dl/norm.h"
#include "dl/rng.h"

typedef struct {
    size_t vocab, layers, width, query_heads, kv_heads, ffn_width;
    float epsilon, init_std;
    double rope_theta;
    uint64_t seed;
} DlModelConfig;

/* Nine independent parameters per block; table first, final scale last. */
enum {
    DL_P_ATTN_NORM, DL_P_QUERY, DL_P_KEY, DL_P_VALUE, DL_P_OUTPUT,
    DL_P_FFN_NORM, DL_P_GATE, DL_P_UP, DL_P_DOWN, DL_BLOCK_PARAMS
};

typedef struct {
    char name[64];
    DlTensor value, grad;           /* Unique owners, both dense rank two. */
    int decay;                      /* Matrices: 1. Norm scales: 0. */
} DlParameter;

typedef struct {
    DlModelConfig config;
    DlParameter *params;            /* Owns param_count initialized records. */
    size_t param_count, value_count;
    uint64_t revision;
    int gradients_valid;
} DlModel;

typedef struct {
    DlMatrix norm1, attention_out, middle, norm2, ffn_out;
    DlAttentionCache attention;
    DlFfnCache ffn;
} DlBlockTape;

typedef struct {
    const DlModel *model;            /* Borrow: model must outlive this tape. */
    DlAttentionConfig attention_config;
    size_t rows;
    DlTensor arena;                 /* Sole float owner for all views below. */
    DlMatrix *hidden;               /* Owns layers+1 view records, NOT floats. */
    DlBlockTape *blocks;            /* Owns layers view records. */
    size_t *metadata;               /* Owns 3*rows size_t: IDs/targets/positions. */
    size_t *ids, *targets, *positions;
    double *inv_freq;               /* Owns head_dim/2 frequencies. */
    DlMatrix final_norm, logits, dlogits;
    DlMatrix g, other, du, dn, embedding_grad;
    DlAttentionScratch attention_scratch;
    DlFfnScratch ffn_scratch;
    DlAttentionMode attention_mode; /* Fixed at construction, not a matrix backend. */
    DlTiledWorkspace *tiled;        /* Owns one forward workspace per block, or NULL. */
    DlTiledBackwardWorkspace tiled_backward; /* Shared across reverse block visits. */
    DlBackend backend;
    uint64_t revision;
    int ready;
} DlTrainTape;

/* Config: positive dimensions, V>=2, D%Hq==0, Hq%Hkv==0, even D/Hq;
 * finite epsilon/init_std > 0, finite theta > 1; checked parameter sizes.
 * Parameter count is 2+9*layers. count outputs unchanged on failure. */
DlStatus dl_model_config_validate(DlModelConfig config,
                                  size_t *parameter_count, size_t *value_count);
/* Strict 12-line name=value file in the documented order, C numeric locale.
 * No spaces/comments/extra lines; LF or CRLF, final newline required.
 * DL_ARGUMENT: syntax; DL_RANGE: numeric range; DL_STATE: file I/O.
 * All three outputs unchanged on failure and disjoint from one another. */
DlStatus dl_model_config_load(const char *path, DlModelConfig *config,
                              size_t *batch, size_t *time);

/* Constructors take initialized empty owners ({0}); never copy live owners
 * or mutate metadata. Failure leaves destination unchanged. Destroy tape
 * BEFORE its model; destroy is repeatable on an empty/partly built owner.
 * No concurrent calls, edits, or destruction. No hidden global state. */
DlStatus dl_model_create(DlModel *out, DlModelConfig config);
void dl_model_destroy(DlModel *model);
/* Float-arena budget excludes parameter arrays and small metadata records.
 * Positive B,T, positions 0..T-1 per sequence. Budget 0 is invalid. */
DlStatus dl_train_tape_create(DlTrainTape *out, const DlModel *model,
                             size_t batch, size_t time, size_t max_float_bytes);
/* Tiled mode removes scores/probs/dP/dS views entirely. Budget includes float
 * arena AND all forward/backward double payload, but excludes view metadata,
 * IDs/targets/positions, frequencies, parameters and optimizer. Legacy create
 * remains materialized with its unchanged float-only budget. Tile sizes are
 * positive, fixed, and part of reproducible training state. */
DlStatus dl_train_tape_create_tiled(DlTrainTape *out, const DlModel *model,
    size_t batch, size_t time, size_t query_tile, size_t key_tile, size_t max_bytes);
DlStatus dl_train_tape_payload_bytes(const DlTrainTape *tape, size_t *out);
void dl_train_tape_destroy(DlTrainTape *tape);

/* Accessors borrow the constructed model's arrays. Index must be in range.
 * Values may be read freely. Before any external value edit, call
 * dl_model_mark_changed and proceed ONLY on success. Never edit a weight
 * between matching forward/backward. Metadata must not be changed. */
DlMatrix dl_parameter_value(DlModel *model, size_t index);
DlMatrix dl_parameter_grad(DlModel *model, size_t index);
DlStatus dl_model_mark_changed(DlModel *model);
DlStatus dl_model_zero_grad(DlModel *model);

/* Dense full-sequence training; ids/targets each contain rows valid IDs.
 * Separate caller-owned input storage and separate double loss output.
 * Copies IDs/targets into tape: caller may reuse inputs after return.
 * Any new forward attempt invalidates the old ready state. Success commits
 * loss/logits and a ready tape; failure leaves loss unchanged, invalidates
 * all tape results, and leaves parameters/parameter gradients unchanged.
 * No gradient clearing, weight update, dropout, or dataset shifting here. */
DlStatus dl_model_forward(DlModel *model, DlTrainTape *tape, DlBackend backend,
                          const size_t *ids, const size_t *targets,
                          size_t count, double *loss);
/* Consume ONE matching ready tape. SET replaces every unique gradient;
 * ADD needs valid finite old gradients and adds one batch's mean derivative.
 * No extra scaling. On arithmetic failure all gradients are INVALID, even
 * earlier successful ADDs; zero/recompute from a known accumulation boundary.
 * Success leaves tape.embedding_grad as the final lookup-input derivative.
 * No second backward without another forward. */
DlStatus dl_model_backward(DlModel *model, DlTrainTape *tape, DlWriteMode mode);
/* Plain-gradient smoke update, NOT AdamW. Requires valid gradients, eta>0.
 * First checks EVERY candidate in double; then updates each unique value once.
 * Failure changes nothing. Success increments revision and invalidates grads
 * and hence any older forward tapes. Call zero_grad before new accumulation. */
DlStatus dl_model_sgd(DlModel *model, double eta);

#endif
```

## Replace the model implementation as a connected file

The blocks below concatenate to the complete `src/model.c`. Parameter enumeration, initialization, strict configuration parsing, residual scheduling, tied-table accumulation, and the plain-gradient update retain their earlier behavior. The new logic is in tape ownership/layout and paired attention dispatch; it must not bypass the existing state guards.

### Retain checked architecture and configuration parsing

The parameter count does not depend on attention strategy. Tiling changes intermediates, not the number or order of trainable tensors. Keep all checked shape arithmetic and named settings so checkpoint compatibility continues to refer to one architecture.

```c title="src/model.c"
#include "dl/model.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RETURN_IF_ERROR(call) do {       \
    DlStatus status_ = (call);           \
    if (status_ != DL_OK) return status_;\
} while (0)

static DlStatus add_product(size_t *sum, size_t a, size_t b)
{
    size_t product = 0;
    RETURN_IF_ERROR(dl_size_mul(a, b, &product));
    return dl_size_add(*sum, product, sum);
}

DlStatus dl_model_config_validate(DlModelConfig c,
                                  size_t *parameter_count, size_t *value_count)
{
    if (parameter_count == NULL || value_count == NULL ||
        parameter_count == value_count) return DL_ARGUMENT;
    if (c.vocab < 2 || c.layers == 0 || c.width == 0 ||
        c.query_heads == 0 || c.kv_heads == 0 || c.ffn_width == 0)
        return DL_SHAPE;
    if (c.width % c.query_heads != 0 || c.query_heads % c.kv_heads != 0 ||
        (c.width / c.query_heads) % 2 != 0) return DL_SHAPE;
    if (!isfinite(c.epsilon) || c.epsilon <= 0.0f ||
        !isfinite(c.init_std) || c.init_std <= 0.0f ||
        !isfinite(c.rope_theta) || c.rope_theta <= 1.0) return DL_RANGE;
    size_t kv = 0, per_block = 0, values = 0, params = 0, bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(c.kv_heads, c.width / c.query_heads, &kv));
    /* Two D*D, two K*D, three F*D, two D scales per block. */
    for (size_t i = 0; i < 2; ++i) {
        RETURN_IF_ERROR(add_product(&per_block, c.width, c.width));
        RETURN_IF_ERROR(add_product(&per_block, kv, c.width));
        RETURN_IF_ERROR(dl_size_add(per_block, c.width, &per_block));
    }
    for (size_t i = 0; i < 3; ++i)
        RETURN_IF_ERROR(add_product(&per_block, c.ffn_width, c.width));
    RETURN_IF_ERROR(add_product(&values, c.vocab, c.width));
    RETURN_IF_ERROR(add_product(&values, c.layers, per_block));
    RETURN_IF_ERROR(dl_size_add(values, c.width, &values));
    RETURN_IF_ERROR(dl_size_mul(c.layers, DL_BLOCK_PARAMS, &params));
    RETURN_IF_ERROR(dl_size_add(params, 2, &params));
    RETURN_IF_ERROR(dl_size_mul(params, sizeof(DlParameter), &bytes));
    RETURN_IF_ERROR(dl_size_mul(values, 2 * sizeof(float), &bytes));
    *parameter_count = params;
    *value_count = values;
    return DL_OK;
}
```

The existing parser still reads the twelve documented configuration lines in order. Tile policy is not silently appended to this architecture file: training run policy will be a separate field of `DlTrainState`. This also preserves every existing configuration fixture.

```c title="src/model.c"
static DlStatus read_setting(FILE *file, const char *key, char value[128])
{
    char line[128];
    size_t used = 0;
    int ch = 0;
    while ((ch = fgetc(file)) != '\n') {
        if (ch == EOF) return ferror(file) ? DL_STATE : DL_ARGUMENT;
        if (ch == 0 || used == sizeof(line) - 1) return DL_ARGUMENT;
        line[used++] = (char)ch;
    }
    if (used != 0 && line[used - 1] == '\r') --used;
    line[used] = '\0';
    size_t key_size = strlen(key);
    if (used <= key_size + 1 || strncmp(line, key, key_size) != 0 ||
        line[key_size] != '=') return DL_ARGUMENT;
    for (size_t i = key_size + 1; i < used; ++i)
        if (isspace((unsigned char)line[i])) return DL_ARGUMENT;
    memcpy(value, line + key_size + 1, used - key_size);
    return DL_OK;
}

static DlStatus decimal_u64(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return DL_ARGUMENT;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return DL_ARGUMENT;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return DL_RANGE;
        value = value * 10 + digit;
    }
    *out = value;
    return DL_OK;
}

static DlStatus parse_settings(FILE *file, DlModelConfig *c,
                                size_t *batch, size_t *time)
{
    const char *keys[] = {
        "vocab", "layers", "width", "query_heads", "kv_heads", "ffn_width",
        "batch", "time", "epsilon", "rope_theta", "init_std", "seed"
    };
    size_t *sizes[] = {&c->vocab, &c->layers, &c->width, &c->query_heads,
                       &c->kv_heads, &c->ffn_width, batch, time};
    char value[128];
    for (size_t i = 0; i < 12; ++i) {
        RETURN_IF_ERROR(read_setting(file, keys[i], value));
        if (i < 8 || i == 11) {
            uint64_t integer = 0;
            RETURN_IF_ERROR(decimal_u64(value, &integer));
            if (i == 11) c->seed = integer;
            else {
                if (integer > SIZE_MAX) return DL_RANGE;
                *sizes[i] = (size_t)integer;
            }
        } else {
            char *end = NULL;
            errno = 0;
            double number = strtod(value, &end);
            if (end == value || *end != '\0') return DL_ARGUMENT;
            if (errno == ERANGE || !isfinite(number) || number <= 0.0)
                return DL_RANGE;
            if (i == 9) c->rope_theta = number;
            else {
                if (number > FLT_MAX || number < FLT_MIN) return DL_RANGE;
                if (i == 8) c->epsilon = (float)number;
                else c->init_std = (float)number;
            }
        }
    }
    if (fgetc(file) != EOF) return DL_ARGUMENT;
    if (ferror(file)) return DL_STATE;
    size_t parameters = 0, values = 0, rows = 0;
    RETURN_IF_ERROR(dl_model_config_validate(*c, &parameters, &values));
    if (*batch == 0 || *time == 0 || *time - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    return dl_size_mul(*batch, *time, &rows);
}

DlStatus dl_model_config_load(const char *path, DlModelConfig *config,
                              size_t *batch, size_t *time)
{
    if (path == NULL || config == NULL || batch == NULL || time == NULL ||
        batch == time) return DL_ARGUMENT;
    FILE *file = fopen(path, "rb");
    if (file == NULL) return DL_STATE;
    DlModelConfig candidate = {0};
    size_t b = 0, t = 0;
    DlStatus status = parse_settings(file, &candidate, &b, &t);
    if (fclose(file) != 0 && status == DL_OK) status = DL_STATE;
    if (status == DL_OK) {
        *config = candidate;
        *batch = b;
        *time = t;
    }
    return status;
}
```

### Retain the unique parameter directory

The table remains one owner despite its lookup and output uses. Scale parameters are initialized to one; matrix initialization uses explicit RNG state and checked names. The model destructor can release a partially constructed directory. A tiled tape borrows this model exactly as a materialized tape does.

```c title="src/model.c"
static DlMatrix dense_view(DlTensor *tensor)
{
    return (DlMatrix){tensor->data, tensor->count, 0,
                      tensor->shape[0], tensor->shape[1], tensor->shape[1]};
}

DlMatrix dl_parameter_value(DlModel *m, size_t index)
{
    return dense_view(&m->params[index].value);
}

DlMatrix dl_parameter_grad(DlModel *m, size_t index)
{
    return dense_view(&m->params[index].grad);
}

static DlConstMatrix weight(DlModel *m, size_t index)
{
    return dl_matrix_read(dl_parameter_value(m, index));
}

static DlStatus create_parameter(DlParameter *p, const char *name,
                                  size_t rows, size_t cols, int scale,
                                  double stddev, DlRng *rng)
{
    int n = snprintf(p->name, sizeof(p->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(p->name)) return DL_RANGE;
    const size_t shape[] = {rows, cols};
    RETURN_IF_ERROR(dl_tensor_create(&p->value, 2, shape));
    RETURN_IF_ERROR(dl_tensor_create(&p->grad, 2, shape));
    p->decay = !scale;
    double limit = sqrt(3.0) * stddev;
    if (!isfinite(limit) || limit > FLT_MAX) return DL_RANGE;
    for (size_t i = 0; i < p->value.count; ++i) {
        p->value.data[i] = scale ? 1.0f
            : (float)((2.0 * dl_rng_uniform(rng) - 1.0) * limit);
        p->grad.data[i] = 0.0f;
    }
    return DL_OK;
}

static DlStatus allocate_parameters(DlModel *m)
{
    size_t bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(m->param_count, sizeof(*m->params), &bytes));
    m->params = malloc(bytes);
    if (m->params == NULL) return DL_ALLOC;
    for (size_t i = 0; i < m->param_count; ++i)
        m->params[i] = (DlParameter){0};
    DlModelConfig c = m->config;
    DlRng rng = {c.seed};
    RETURN_IF_ERROR(create_parameter(&m->params[0], "token_embedding",
                                      c.vocab, c.width, 0, c.init_std, &rng));
    size_t kv = c.kv_heads * (c.width / c.query_heads);
    const char *names[] = {"attn_norm", "query", "key", "value", "output",
                           "ffn_norm", "gate", "up", "down"};
    const size_t rows[] = {1, c.width, kv, kv, c.width, 1,
                           c.ffn_width, c.ffn_width, c.width};
    const size_t cols[] = {c.width, c.width, c.width, c.width, c.width,
                           c.width, c.width, c.width, c.ffn_width};
    for (size_t layer = 0; layer < c.layers; ++layer) {
        for (size_t slot = 0; slot < DL_BLOCK_PARAMS; ++slot) {
            char name[64];
            int n = snprintf(name, sizeof(name), "blocks.%zu.%s", layer, names[slot]);
            if (n < 0 || (size_t)n >= sizeof(name)) return DL_RANGE;
            int scale = slot == DL_P_ATTN_NORM || slot == DL_P_FFN_NORM;
            double stddev = c.init_std;
            if (slot == DL_P_OUTPUT || slot == DL_P_DOWN)
                stddev /= sqrt(2.0 * (double)c.layers);
            size_t index = 1 + layer * DL_BLOCK_PARAMS + slot;
            RETURN_IF_ERROR(create_parameter(&m->params[index], name,
                rows[slot], cols[slot], scale, stddev, &rng));
        }
    }
    return create_parameter(&m->params[m->param_count - 1], "final_norm",
                             1, c.width, 1, c.init_std, &rng);
}

void dl_model_destroy(DlModel *m)
{
    if (m == NULL) return;
    if (m->params != NULL) {
        for (size_t i = 0; i < m->param_count; ++i) {
            dl_tensor_destroy(&m->params[i].value);
            dl_tensor_destroy(&m->params[i].grad);
        }
    }
    free(m->params);
    *m = (DlModel){0};
}

DlStatus dl_model_create(DlModel *out, DlModelConfig config)
{
    if (out == NULL) return DL_ARGUMENT;
    if (out->params != NULL || out->param_count != 0) return DL_STATE;
    DlModel candidate = {0};
    candidate.config = config;
    RETURN_IF_ERROR(dl_model_config_validate(config, &candidate.param_count,
                                              &candidate.value_count));
    DlStatus status = allocate_parameters(&candidate);
    if (status != DL_OK) {
        dl_model_destroy(&candidate);
        return status;
    }
    candidate.revision = 1;
    candidate.gradients_valid = 1;   /* Constructor initialized all to zero. */
    *out = candidate;               /* Sole ownership transfer. */
    return DL_OK;
}
```

## Omit square placements, not just square computation

`place` advances a checked arena cursor. On the first layout pass, the arena is absent and only counts are computed. On the second pass, actual views are constructed. `place_attention` places the six linear views in either mode, but places scores/probabilities only for materialized attention. Its square gradient callers use the same function and therefore receive the same omission.

The `scores`/`probs` records were initialized to `{0}` before layout. Leaving them empty is intentional, not a missing assignment. The checked `score_rows` product is linear and is still harmless; only the multiplication by time inside the skipped placements would be quadratic.

```c title="src/model.c"
static DlStatus place(DlTrainTape *t, size_t *cursor, size_t rows, size_t cols,
                       DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (t->arena.data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(&t->arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus place_attention(DlTrainTape *t, size_t *cursor,
                                 DlMatrix *q0, DlMatrix *k0, DlMatrix *v,
                                 DlMatrix *q, DlMatrix *k, DlMatrix *scores,
                                 DlMatrix *probs, DlMatrix *context)
{
    size_t m = t->rows, d = t->model->config.width;
    size_t kv = t->attention_config.kv_heads * t->attention_config.head_dim;
    size_t score_rows = 0;
    RETURN_IF_ERROR(dl_size_mul(m, t->attention_config.query_heads, &score_rows));
    RETURN_IF_ERROR(place(t, cursor, m, d, q0));
    RETURN_IF_ERROR(place(t, cursor, m, kv, k0));
    RETURN_IF_ERROR(place(t, cursor, m, kv, v));
    RETURN_IF_ERROR(place(t, cursor, m, d, q));
    RETURN_IF_ERROR(place(t, cursor, m, kv, k));
    if (t->attention_mode == DL_ATTENTION_MATERIALIZED) {
        RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, scores));
        RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, probs));
    } /* Tiled views stay {0}; no square count is even computed. */
    return place(t, cursor, m, d, context);
}
```

### Keep the existing saved activations and reverse scratch

The layout still records every block's forward values and one shared reverse workspace. It does not discard the normalized input needed for projection weight gradients. Query/key gradients before and after RoPE remain separate views, avoiding illegal in-place transforms. The vocabulary logits and their gradients also remain materialized; this chapter optimizes attention storage, not every allocation in the model.

```c title="src/model.c"
static DlStatus layout_tape(DlTrainTape *t, size_t *count)
{
    DlModelConfig c = t->model->config;
    size_t cursor = 0, m = t->rows;
    for (size_t i = 0; i <= c.layers; ++i)
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->hidden[i]));
    for (size_t i = 0; i < c.layers; ++i) {
        DlBlockTape *b = &t->blocks[i];
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->norm1));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->attention_out));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->middle));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->norm2));
        RETURN_IF_ERROR(place(t, &cursor, m, c.width, &b->ffn_out));
        DlAttentionCache *a = &b->attention;
        RETURN_IF_ERROR(place_attention(t, &cursor, &a->q0, &a->k0, &a->v,
            &a->q, &a->k, &a->scores, &a->probs, &a->context));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.gate));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.value));
        RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &b->ffn.hidden));
    }
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->final_norm));
    RETURN_IF_ERROR(place(t, &cursor, m, c.vocab, &t->logits));
    RETURN_IF_ERROR(place(t, &cursor, m, c.vocab, &t->dlogits));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->g));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->other));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->du));
    RETURN_IF_ERROR(place(t, &cursor, m, c.width, &t->dn));
    DlAttentionScratch *a = &t->attention_scratch;
    RETURN_IF_ERROR(place_attention(t, &cursor, &a->q0, &a->k0, &a->v,
        &a->q, &a->k, &a->scores, &a->probs, &a->context));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.hidden));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.gate));
    RETURN_IF_ERROR(place(t, &cursor, m, c.ffn_width, &t->ffn_scratch.value));
    *count = cursor;
    return DL_OK;
}
```

## Construct a candidate under one combined budget

Allocate the array of forward workspace records, initialize **every record** to empty before constructing any one of them, and then build them sequentially. Subtract each successful numerical payload from the remaining budget. A late failure can safely walk and destroy every record because the untouched ones are still empty.

First reserve the complete float count, then use the remaining budget for all forward/backward doubles. No caller-owned output changes until the candidate is complete. The arena allocation may fail after double owners exist; destruction must therefore release both kinds of storage.

```c title="src/model.c"
static DlStatus allocate_tiled_state(DlTrainTape *t, size_t qr, size_t kc,
                                      size_t *remaining)
{
    size_t layers = t->model->config.layers, bytes = 0, n = 0;
    RETURN_IF_ERROR(dl_size_mul(layers, sizeof(*t->tiled), &bytes));
    t->tiled = malloc(bytes);
    if (t->tiled == NULL) return DL_ALLOC;
    for (size_t i = 0; i < layers; ++i) t->tiled[i] = (DlTiledWorkspace){0};
    RETURN_IF_ERROR(dl_size_mul(t->rows, t->attention_config.query_heads, &n));
    for (size_t i = 0; i < layers; ++i) {
        if (*remaining == 0) return DL_RANGE;
        RETURN_IF_ERROR(dl_tiled_workspace_create(&t->tiled[i], n,
            t->attention_config.head_dim, qr, kc, *remaining));
        *remaining -= t->tiled[i].count * sizeof(double);
    }
    if (*remaining == 0) return DL_RANGE;
    RETURN_IF_ERROR(dl_tiled_backward_workspace_create(&t->tiled_backward,
        t->attention_config.head_dim, qr, kc, *remaining));
    *remaining -= t->tiled_backward.count * sizeof(double);
    return DL_OK;
}

static DlStatus allocate_tape(DlTrainTape *t, size_t max_bytes, size_t qr, size_t kc)
{
    size_t layers = t->model->config.layers, bytes = 0, entries = 0;
    RETURN_IF_ERROR(dl_size_add(layers, 1, &entries));
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->hidden), &bytes));
    t->hidden = malloc(bytes);
    if (t->hidden == NULL) return DL_ALLOC;
    for (size_t i = 0; i < entries; ++i) t->hidden[i] = (DlMatrix){0};
    RETURN_IF_ERROR(dl_size_mul(layers, sizeof(*t->blocks), &bytes));
    t->blocks = malloc(bytes);
    if (t->blocks == NULL) return DL_ALLOC;
    for (size_t i = 0; i < layers; ++i) t->blocks[i] = (DlBlockTape){0};
    size_t count = 0;
    RETURN_IF_ERROR(layout_tape(t, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    if (bytes > max_bytes) return DL_RANGE;
    size_t remaining = max_bytes - bytes;
    if (t->attention_mode == DL_ATTENTION_TILED)
        RETURN_IF_ERROR(allocate_tiled_state(t, qr, kc, &remaining));
    RETURN_IF_ERROR(dl_tensor_create(&t->arena, 1, &count));
    RETURN_IF_ERROR(layout_tape(t, &count));
    RETURN_IF_ERROR(dl_size_mul(t->rows, 3, &entries));
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->metadata), &bytes));
    t->metadata = malloc(bytes);
    if (t->metadata == NULL) return DL_ALLOC;
    t->ids = t->metadata;
    t->targets = t->metadata + t->rows;
    t->positions = t->metadata + 2 * t->rows;
    for (size_t i = 0; i < t->rows; ++i)
        t->positions[i] = i % t->attention_config.time;
    entries = t->attention_config.head_dim / 2;
    RETURN_IF_ERROR(dl_size_mul(entries, sizeof(*t->inv_freq), &bytes));
    t->inv_freq = malloc(bytes);
    if (t->inv_freq == NULL) return DL_ALLOC;
    return dl_rope_frequencies(t->attention_config.head_dim,
                               t->model->config.rope_theta, t->inv_freq, entries);
}
```

### Destroy borrowers before their model

Cleanup reads the model's layer count to destroy per-block owners, so the tape must still be destroyed before the model. The internal constructor sets mode once and commits the candidate only after all allocations and frequencies succeed. Public wrappers offer materialized or tiled construction without allowing a live tape's allocation policy to be edited.

```c title="src/model.c"
void dl_train_tape_destroy(DlTrainTape *t)
{
    if (t == NULL) return;
    dl_tiled_backward_workspace_destroy(&t->tiled_backward);
    if (t->tiled != NULL) {
        for (size_t i = 0; i < t->model->config.layers; ++i)
            dl_tiled_workspace_destroy(&t->tiled[i]);
    }
    free(t->tiled);
    free(t->inv_freq);
    free(t->metadata);
    free(t->blocks);
    free(t->hidden);
    dl_tensor_destroy(&t->arena);
    *t = (DlTrainTape){0};
}

static DlStatus tape_create(DlTrainTape *out, const DlModel *model,
    size_t batch, size_t time, DlAttentionMode mode, size_t qr, size_t kc, size_t max_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL ||
        batch == 0 || time == 0 || max_bytes == 0) return DL_ARGUMENT;
    if (out->model != NULL || out->arena.data != NULL || out->hidden != NULL ||
        out->blocks != NULL || out->metadata != NULL || out->inv_freq != NULL ||
        out->tiled != NULL || out->tiled_backward.data != NULL)
        return DL_STATE;
    if (time - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlTrainTape candidate = {0};
    candidate.model = model;
    candidate.attention_mode = mode;
    candidate.attention_config = (DlAttentionConfig){batch, time,
        model->config.query_heads, model->config.kv_heads,
        model->config.width / model->config.query_heads};
    RETURN_IF_ERROR(dl_size_mul(batch, time, &candidate.rows));
    DlStatus status = allocate_tape(&candidate, max_bytes, qr, kc);
    if (status != DL_OK) {
        dl_train_tape_destroy(&candidate);
        return status;
    }
    *out = candidate;
    return DL_OK;
}
```

### Make mode and payload visible

The legacy constructor requests no tiled owners. The tiled constructor rejects zero tile capacities. The byte accessor sums every retained double owner; it does not count only the backward tile while forgetting saved statistics. The two small conversion helpers copy **borrowed view records**, not owning buffers: this is safe and leaves lifetimes unchanged.

```c title="src/model.c"
DlStatus dl_train_tape_create(DlTrainTape *out, const DlModel *model,
                             size_t batch, size_t time, size_t max_float_bytes)
{
    return tape_create(out, model, batch, time, DL_ATTENTION_MATERIALIZED,
                       0, 0, max_float_bytes);
}

DlStatus dl_train_tape_create_tiled(DlTrainTape *out, const DlModel *model,
    size_t batch, size_t time, size_t qr, size_t kc, size_t max_bytes)
{
    if (qr == 0 || kc == 0) return DL_SHAPE;
    return tape_create(out, model, batch, time, DL_ATTENTION_TILED, qr, kc, max_bytes);
}

DlStatus dl_train_tape_payload_bytes(const DlTrainTape *t, size_t *out)
{
    if (t == NULL || out == NULL || t->model == NULL || t->arena.data == NULL)
        return DL_ARGUMENT;
    size_t total = 0, doubles = t->tiled_backward.count, bytes = 0;
    RETURN_IF_ERROR(dl_size_mul(t->arena.count, sizeof(float), &total));
    if (t->tiled != NULL)
        for (size_t i = 0; i < t->model->config.layers; ++i)
            RETURN_IF_ERROR(dl_size_add(doubles, t->tiled[i].count, &doubles));
    RETURN_IF_ERROR(dl_size_mul(doubles, sizeof(double), &bytes));
    RETURN_IF_ERROR(dl_size_add(total, bytes, &total));
    *out = total;
    return DL_OK;
}

static DlTiledAttentionCache tiled_cache(DlAttentionCache a)
{
    return (DlTiledAttentionCache){a.q0, a.k0, a.v, a.q, a.k, a.context};
}

static DlTiledAttentionScratch tiled_scratch(DlAttentionScratch a)
{
    return (DlTiledAttentionScratch){a.context, a.q, a.k, a.v, a.q0, a.k0};
}
```

### Reuse the existing parameter access and revision protocol

These helpers select the same unique parameter and gradient records for both strategies. `dl_model_mark_changed` advances the model revision and invalidates old gradients before external edits. `dl_model_zero_grad` is still the explicit recovery/start boundary for gradient accumulation.

```c title="src/model.c"
static DlAttentionWeights attention_weights(DlModel *m, size_t base)
{
    return (DlAttentionWeights){weight(m, base + DL_P_QUERY),
        weight(m, base + DL_P_KEY), weight(m, base + DL_P_VALUE),
        weight(m, base + DL_P_OUTPUT)};
}

static DlAttentionWeightGrads attention_grads(DlModel *m, size_t base)
{
    return (DlAttentionWeightGrads){dl_parameter_grad(m, base + DL_P_QUERY),
        dl_parameter_grad(m, base + DL_P_KEY), dl_parameter_grad(m, base + DL_P_VALUE),
        dl_parameter_grad(m, base + DL_P_OUTPUT)};
}

static DlFfnWeights ffn_weights(DlModel *m, size_t base)
{
    return (DlFfnWeights){weight(m, base + DL_P_GATE),
                          weight(m, base + DL_P_UP), weight(m, base + DL_P_DOWN)};
}

static DlFfnGrads ffn_grads(DlModel *m, size_t base)
{
    return (DlFfnGrads){dl_parameter_grad(m, base + DL_P_GATE),
        dl_parameter_grad(m, base + DL_P_UP), dl_parameter_grad(m, base + DL_P_DOWN)};
}

static DlStatus finite_parameters(DlModel *m, int gradients)
{
    for (size_t p = 0; p < m->param_count; ++p) {
        DlTensor *a = gradients ? &m->params[p].grad : &m->params[p].value;
        for (size_t i = 0; i < a->count; ++i)
            if (!isfinite(a->data[i])) return DL_NUMERIC;
    }
    return DL_OK;
}

DlStatus dl_model_mark_changed(DlModel *m)
{
    if (m == NULL || m->params == NULL) return DL_ARGUMENT;
    if (m->revision == UINT64_MAX) return DL_OVERFLOW;
    ++m->revision;
    m->gradients_valid = 0;
    return DL_OK;
}

DlStatus dl_model_zero_grad(DlModel *m)
{
    if (m == NULL || m->params == NULL) return DL_ARGUMENT;
    for (size_t p = 0; p < m->param_count; ++p)
        for (size_t i = 0; i < m->params[p].grad.count; ++i)
            m->params[p].grad.data[i] = 0.0f;
    m->gradients_valid = 1;
    return DL_OK;
}
```

## Forward and backward are a matched pair

After each block's first normalization, dispatch to the selected attention forward. The tiled call receives **`&t->tiled[layer]`**, never a single workspace overwritten for all blocks. Later normalization, feed-forward, residual, vocabulary projection, and loss computation are unchanged. Ready state commits only after the full model forward succeeds.

```c title="src/model.c"
DlStatus dl_model_forward(DlModel *m, DlTrainTape *t, DlBackend backend,
                          const size_t *ids, const size_t *targets,
                          size_t count, double *loss)
{
    if (t == NULL) return DL_ARGUMENT;
    t->ready = 0;
    t->embedding_grad = (DlMatrix){0};
    if (m == NULL || m->params == NULL || t->model != m ||
        t->arena.data == NULL || ids == NULL || targets == NULL || loss == NULL)
        return DL_ARGUMENT;
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED)
        return DL_ARGUMENT;
    if (count != t->rows) return DL_SHAPE;
    for (size_t i = 0; i < count; ++i)
        if (ids[i] >= m->config.vocab || targets[i] >= m->config.vocab)
            return DL_RANGE;
    RETURN_IF_ERROR(finite_parameters(m, 0));
    for (size_t i = 0; i < count; ++i) {
        t->ids[i] = ids[i];
        t->targets[i] = targets[i];
    }
    RETURN_IF_ERROR(dl_embedding_forward(weight(m, 0), t->ids, count, t->hidden[0]));
    for (size_t layer = 0; layer < m->config.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        DlBlockTape *b = &t->blocks[layer];
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(t->hidden[layer]),
            weight(m, base + DL_P_ATTN_NORM), m->config.epsilon, b->norm1));
        if (t->attention_mode == DL_ATTENTION_TILED) {
            RETURN_IF_ERROR(dl_attention_tiled_forward(backend, t->attention_config,
                dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
                count, t->inv_freq, t->attention_config.head_dim / 2,
                tiled_cache(b->attention), &t->tiled[layer], b->attention_out));
        } else {
        RETURN_IF_ERROR(dl_attention_forward(backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            count, t->inv_freq, t->attention_config.head_dim / 2,
            b->attention, b->attention_out));
        }
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(t->hidden[layer]),
            dl_matrix_read(b->attention_out), b->middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(b->middle),
            weight(m, base + DL_P_FFN_NORM), m->config.epsilon, b->norm2));
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(b->norm2),
            ffn_weights(m, base), b->ffn, b->ffn_out));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(b->middle),
            dl_matrix_read(b->ffn_out), t->hidden[layer + 1]));
    }
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(t->hidden[m->config.layers]),
        weight(m, m->param_count - 1), m->config.epsilon, t->final_norm));
    RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(t->final_norm), DL_NOTRANS,
        weight(m, 0), DL_TRANS, t->logits, DL_SET));
    double candidate_loss = 0.0;
    RETURN_IF_ERROR(dl_cross_entropy_mean(dl_matrix_read(t->logits), t->targets,
        count, &candidate_loss, NULL, DL_SET));
    t->backend = backend;
    t->revision = m->revision;
    t->ready = 1;
    *loss = candidate_loss;
    return DL_OK;
}
```

### Reverse the same graph and preserve all branches

Backward uses the selected tape mode, the saved matrix backend, and the same block index's statistics. It does not choose a different attention algorithm based on available memory halfway through a call. The residual identity path and learned branch still merge through SET/ADD as derived in Chapters 7 and 12.

The output use of the tied embedding table is written at the beginning; the lookup use is added at the end. Tiling does not change that obligation. The tape is consumed and global gradient validity is cleared **before** any reverse operation, so a failure in an earlier block cannot accidentally leave later-block gradients marked usable.

```c title="src/model.c"
DlStatus dl_model_backward(DlModel *m, DlTrainTape *t, DlWriteMode mode)
{
    if (m == NULL || m->params == NULL || t == NULL || t->model != m)
        return DL_ARGUMENT;
    if (mode != DL_SET && mode != DL_ADD) return DL_ARGUMENT;
    if (!t->ready || t->revision != m->revision) return DL_STATE;
    if (mode == DL_ADD) {
        if (!m->gradients_valid) return DL_STATE;
        RETURN_IF_ERROR(finite_parameters(m, 1));
    }
    t->ready = 0;                    /* Consume before any gradient writes. */
    t->embedding_grad = (DlMatrix){0};
    m->gradients_valid = 0;
    double ignored_loss = 0.0;
    RETURN_IF_ERROR(dl_cross_entropy_mean(dl_matrix_read(t->logits), t->targets,
        t->rows, &ignored_loss, &t->dlogits, DL_SET));
    RETURN_IF_ERROR(dl_matmul(t->backend, dl_matrix_read(t->dlogits), DL_TRANS,
        dl_matrix_read(t->final_norm), DL_NOTRANS, dl_parameter_grad(m, 0), mode));
    RETURN_IF_ERROR(dl_matmul(t->backend, dl_matrix_read(t->dlogits), DL_NOTRANS,
        weight(m, 0), DL_NOTRANS, t->other, DL_SET));
    RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(t->hidden[m->config.layers]),
        weight(m, m->param_count - 1), dl_matrix_read(t->other), m->config.epsilon,
        t->g, DL_SET, dl_parameter_grad(m, m->param_count - 1), mode));
    DlMatrix g = t->g, dx = t->other; /* Swap borrowed views, never owners. */
    for (size_t remaining = m->config.layers; remaining != 0; --remaining) {
        size_t layer = remaining - 1, base = 1 + layer * DL_BLOCK_PARAMS;
        DlBlockTape *b = &t->blocks[layer];
        RETURN_IF_ERROR(dl_residual_backward(dl_matrix_read(g), t->du, DL_SET));
        RETURN_IF_ERROR(dl_ffn_backward(t->backend, dl_matrix_read(b->norm2),
            ffn_weights(m, base), b->ffn, dl_matrix_read(g), t->ffn_scratch,
            t->dn, DL_SET, ffn_grads(m, base), mode));
        RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(b->middle),
            weight(m, base + DL_P_FFN_NORM), dl_matrix_read(t->dn), m->config.epsilon,
            t->du, DL_ADD, dl_parameter_grad(m, base + DL_P_FFN_NORM), mode));
        RETURN_IF_ERROR(dl_residual_backward(dl_matrix_read(t->du), dx, DL_SET));
        if (t->attention_mode == DL_ATTENTION_TILED) {
            RETURN_IF_ERROR(dl_attention_tiled_backward(t->backend, t->attention_config,
                dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
                t->rows, t->inv_freq, t->attention_config.head_dim / 2,
                tiled_cache(b->attention), &t->tiled[layer], dl_matrix_read(t->du),
                tiled_scratch(t->attention_scratch), &t->tiled_backward,
                t->dn, DL_SET, attention_grads(m, base), mode));
        } else {
        RETURN_IF_ERROR(dl_attention_backward(t->backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            t->rows, t->inv_freq, t->attention_config.head_dim / 2, b->attention,
            dl_matrix_read(t->du), t->attention_scratch, t->dn, DL_SET,
            attention_grads(m, base), mode));
        }
        RETURN_IF_ERROR(dl_rmsnorm_backward(dl_matrix_read(t->hidden[layer]),
            weight(m, base + DL_P_ATTN_NORM), dl_matrix_read(t->dn), m->config.epsilon,
            dx, DL_ADD, dl_parameter_grad(m, base + DL_P_ATTN_NORM), mode));
        DlMatrix temporary = g;
        g = dx;
        dx = temporary;
    }
    RETURN_IF_ERROR(dl_embedding_backward(dl_matrix_read(g), t->ids, t->rows,
                                           dl_parameter_grad(m, 0), DL_ADD));
    t->embedding_grad = g;
    m->gradients_valid = 1;
    return DL_OK;
}
```

### Retain the small plain-update diagnostic

The smoke update still preflights every candidate value and consumes valid gradients once. It is useful for diagnosing a complete graph but is not a replacement for the existing AdamW trainer. Its validity checks also make failed-backward recovery observable in Part 5.

```c title="src/model.c"
DlStatus dl_model_sgd(DlModel *m, double eta)
{
    if (m == NULL || m->params == NULL || !isfinite(eta) || eta <= 0.0)
        return DL_ARGUMENT;
    if (!m->gradients_valid) return DL_STATE;
    if (m->revision == UINT64_MAX) return DL_OVERFLOW;
    for (size_t p = 0; p < m->param_count; ++p) {
        DlParameter *a = &m->params[p];
        for (size_t i = 0; i < a->value.count; ++i) {
            double w = a->value.data[i], g = a->grad.data[i];
            double next = w - eta * g;
            if (!isfinite(w) || !isfinite(g) || !isfinite(next) || fabs(next) > FLT_MAX)
                return DL_NUMERIC;
        }
    }
    for (size_t p = 0; p < m->param_count; ++p) {
        DlParameter *a = &m->params[p];
        for (size_t i = 0; i < a->value.count; ++i)
            a->value.data[i] = (float)((double)a->value.data[i] - eta * a->grad.data[i]);
    }
    return dl_model_mark_changed(m);
}

#undef RETURN_IF_ERROR
```

## Trace a two-block backward failure

Suppose forward succeeds for blocks zero and one. Backward consumes the ready tape, differentiates block one, and then discovers an invalid scaled sum in block zero. Some block-one parameter gradients already exist, but the entire model's `gradients_valid` flag is false. An update returns `DL_STATE`. A fresh successful forward followed by SET backward overwrites the invalid gradients and reestablishes a usable result.

Do not repair the one statistic and retry the consumed tape with ADD: this can count previously written gradients twice. The outer trainer will abort the current accumulation group and replay from its unchanged data RNG boundary, as in Chapter 15.

Continue with [Part 3 — Training policy and versioned continuation](https://25d-228.github.io/gpt-c/chapter-19-part-3.html).
