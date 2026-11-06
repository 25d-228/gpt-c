# Chapter 18 · Part 2 — Integrating bounded attention into inference

The core now computes context without square storage. An inference integration is only memory-bounded if its owner also stops allocating the unused square arrays. This part changes the existing workspace layout and dispatch while retaining the same decoder graph, immutable weight owner, checkpoint reader, and generation loop.

This is the second part of the **5.00-hour Chapter 18**. Begin with [Part 1](https://25d-228.github.io/gpt-c/chapter-18.html). The complete checks and their observed results are in [Part 3](https://25d-228.github.io/gpt-c/chapter-18-part-3.html).

## Choose storage at construction, not after allocating everything

`DlInference` now contains one `DlTiledWorkspace`. A null `tiled.data` means the original materialized path; a live tiled owner means tiled full-prefix inference. Strategy is fixed when the workspace is constructed. There is no supported raw field edit that switches a live workspace between layouts.

The original `dl_inference_create` remains source-compatible and preserves its float-payload budget. The new `dl_inference_create_tiled` takes query and key tile dimensions plus a **combined float-and-double payload budget**. Giving the double owner a second independent budget would hide its contribution from the caller's bound.

Replace `include/dl/inference.h` in full:

```c title="include/dl/inference.h"
#ifndef DL_INFERENCE_H
#define DL_INFERENCE_H

#include "dl/model.h"
#include "dl/attention_tiled.h"

typedef struct {
    char name[64];
    DlMatrix value;                  /* Borrow into the single weight arena. */
    int decay;                      /* Format metadata, not an optimizer. */
} DlWeight;

typedef struct {
    DlModelConfig config;
    DlWeight *params;               /* Owns param_count view records. */
    DlTensor arena;                 /* Owns ONLY P weight floats. */
    size_t param_count;
    int ready;                     /* Set only by a successful copy/load. */
} DlWeights;

typedef struct {
    const DlWeights *weights;      /* Stable-address immutable borrow. */
    size_t capacity;                /* Maximum ONE-sequence prefix length. */
    DlTiledWorkspace tiled;         /* data==NULL: materialized; otherwise tiled. */
    DlTensor arena;                 /* One reusable layer, no backward data. */
    size_t *positions;              /* Owns capacity absolute positions. */
    double *inv_freq;               /* Owns head_dim/2 frequencies. */
} DlInference;

/* Empty {0} owners only. No live owner copies, metadata edits or concurrency.
 * Budgets bound float payloads, not metadata/allocator/stack/process memory.
 * Constructors leave out unchanged on failure. Destroy is repeatable.
 * create allocates an UNREADY directory for the checkpoint loader; it zeros
 * the float payload but does not initialize a usable model. from_model copies
 * values only, never gradients, and commits ready=1 after checking finiteness.
 * After copy/load, weights are immutable. Destroy all borrowing workspaces
 * before their weights. Do not reload, move, or mutate a borrowed weight owner. */
DlStatus dl_weights_create(DlWeights *out, DlModelConfig config,
                           size_t max_float_bytes);
DlStatus dl_weights_from_model(DlWeights *out, const DlModel *model,
                               size_t max_float_bytes);
void dl_weights_destroy(DlWeights *weights);
DlStatus dl_inference_create(DlInference *out, const DlWeights *weights,
                             size_t capacity, size_t max_float_bytes);
/* Same immutable-weight borrow, but no square scores/probabilities. This budget
 * includes BOTH the float arena AND all double tile/statistics payload. The
 * positions/frequencies/metadata and called-kernel stack are counted separately.
 * query_tile=16, key_tile=32 are starting choices, not architectural constants.
 * Attention strategy is fixed at creation; backend still selects matmul only. */
DlStatus dl_inference_create_tiled(DlInference *out, const DlWeights *weights,
                                   size_t capacity, size_t query_tile,
                                   size_t key_tile, size_t max_payload_bytes);
void dl_inference_destroy(DlInference *workspace);

/* Full-prefix B=1 baseline, no persistent KV state and no allocations here.
 * ids[length] is a separate live array, 1<=length<=capacity, each ID<V.
 * Compute all hidden prefix rows, but final norm/projection only for rows
 * [first_row, first_row+logits.rows). logits.cols==V; nonempty valid view.
 * Caller owns logits, with canonical base/count metadata. Its ENTIRE owner
 * must differ from both arenas; cross-type metadata/IDs are also disjoint.
 * No targets, loss, gradients, moments, dropout, RNG, or hidden append counter.
 * Invalid arguments/IDs/geometry leave logits unchanged. Later numeric errors
 * can partially write logits; discard the WHOLE result on any failure.
 * Scratch is disposable on failure; a fresh full call safely overwrites it.
 * Logits remain caller-owned and survive the next workspace call. */
DlStatus dl_inference_forward(DlInference *workspace, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits);

#endif
```

`DlWeights` is unchanged. It owns parameter values once, including the tied embedding/output table, without gradients or moments. Both inference variants borrow its stable address. Destroy every borrowing workspace before destroying its weights; never mutate or reload borrowed weights.

The selected `DlBackend` remains a per-call projection choice. For example, `blocked` plus `tiled` means blocked C matrix projections and the online attention core. `blocked` plus `full` means the same blocked projections and materialized attention. Neither string requests a GPU implementation.

## Preserve the weight directory and its initialization rules

Replace `src/inference.c` with the blocks in this part. The following portion is unchanged from Chapter 16, but is included so the replacement is complete. It frees the single weight arena and directory, creates the named views in model order, and copies only finite values from an existing training model. An unready weight directory is not a model that inference may execute.

```c title="src/inference.c"
#include "dl/inference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RETURN_IF_ERROR(call) do { \
    DlStatus status_ = (call); if (status_ != DL_OK) return status_; \
} while (0)

void dl_weights_destroy(DlWeights *w)
{
    if (w == NULL) return;
    free(w->params);
    dl_tensor_destroy(&w->arena);
    *w = (DlWeights){0};
}

static DlStatus weight_record(DlWeights *w, size_t index, size_t *cursor,
                               const char *name, size_t rows, size_t cols,
                               int decay)
{
    DlWeight *p = &w->params[index];
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    int n = snprintf(p->name, sizeof(p->name), "%s", name);
    if (n < 0 || (size_t)n >= sizeof(p->name)) return DL_RANGE;
    RETURN_IF_ERROR(dl_matrix_view(&w->arena, *cursor, rows, cols, cols, &p->value));
    p->decay = decay;
    *cursor = end;
    return DL_OK;
}

static DlStatus weight_directory(DlWeights *w)
{
    DlModelConfig c = w->config;
    size_t cursor = 0;
    RETURN_IF_ERROR(weight_record(w, 0, &cursor, "token_embedding", c.vocab, c.width, 1));
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
            RETURN_IF_ERROR(weight_record(w, 1 + layer * DL_BLOCK_PARAMS + slot,
                &cursor, name, rows[slot], cols[slot], !scale));
        }
    }
    RETURN_IF_ERROR(weight_record(w, w->param_count - 1, &cursor,
                                  "final_norm", 1, c.width, 0));
    return cursor == w->arena.count ? DL_OK : DL_STATE;
}

DlStatus dl_weights_create(DlWeights *out, DlModelConfig config,
                           size_t max_float_bytes)
{
    if (out == NULL || max_float_bytes == 0) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    DlWeights next = {.config = config};
    size_t count = 0, bytes = 0, directory_bytes = 0;
    RETURN_IF_ERROR(dl_model_config_validate(config, &next.param_count, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(next.param_count, sizeof(*next.params), &directory_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    next.params = malloc(directory_bytes);
    if (next.params == NULL) return DL_ALLOC;
    for (size_t i = 0; i < next.param_count; ++i) next.params[i] = (DlWeight){0};
    DlStatus status = dl_tensor_create(&next.arena, 1, &count);
    if (status == DL_OK) status = weight_directory(&next);
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    for (size_t i = 0; i < count; ++i) next.arena.data[i] = 0.0f;
    *out = next;
    return DL_OK;
}

DlStatus dl_weights_from_model(DlWeights *out, const DlModel *model,
                               size_t max_float_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->ready) return DL_STATE;
    DlWeights next = {0};
    RETURN_IF_ERROR(dl_weights_create(&next, model->config, max_float_bytes));
    DlStatus status = DL_OK;
    for (size_t p = 0; p < next.param_count && status == DL_OK; ++p) {
        const DlParameter *source = &model->params[p];
        DlMatrix target = next.params[p].value;
        if (strcmp(source->name, next.params[p].name) != 0 ||
            source->value.shape[0] != target.rows || source->value.shape[1] != target.cols) {
            status = DL_SHAPE;
            break;
        }
        for (size_t i = 0; i < source->value.count; ++i) {
            float value = source->value.data[i];
            if (!isfinite(value)) { status = DL_NUMERIC; break; }
            target.base[target.offset + i] = value;
        }
    }
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    next.ready = 1;
    *out = next;
    return DL_OK;
}
```

`weight_directory` visits the same unique parameter order as the model and checkpoint format. Scale parameters still carry the metadata indicating that they are not decayed; inference itself performs no optimizer operation. The final norm is last, and the tied output projection continues to read weight record zero.

## A single layout with one explicit storage difference

The `Views` record still names the layer's hidden states, attention projections, context, and feed-forward intermediates. Its attention view type includes the old score/probability fields, but those fields remain empty in the tiled layout. An empty view record owns no floats.

```c title="src/inference.c"
typedef struct {
    DlMatrix hidden, norm, branch, middle;
    DlAttentionCache attention;
    DlFfnCache ffn;
} Views;

static DlStatus place(DlTensor *arena, size_t *cursor, size_t rows, size_t cols,
                       DlMatrix *view)
{
    size_t count = 0, end = 0;
    RETURN_IF_ERROR(dl_size_mul(rows, cols, &count));
    RETURN_IF_ERROR(dl_size_add(*cursor, count, &end));
    if (arena->data != NULL)
        RETURN_IF_ERROR(dl_matrix_view(arena, *cursor, rows, cols, cols, view));
    *cursor = end;
    return DL_OK;
}

static DlStatus layout(DlTensor *arena, DlModelConfig c, size_t time, int tiled,
                        Views *v, size_t *count)
{
    size_t cursor = 0, score_rows = 0;
    size_t kv = c.kv_heads * (c.width / c.query_heads);
    if (!tiled) RETURN_IF_ERROR(dl_size_mul(time, c.query_heads, &score_rows));
#define PLACE(rows_, cols_, member_) \
    RETURN_IF_ERROR(place(arena, &cursor, rows_, cols_, &v->member_))
    PLACE(time, c.width, hidden);
    PLACE(time, c.width, norm);
    PLACE(time, c.width, branch);
    PLACE(time, c.width, middle);
    PLACE(time, c.width, attention.q0);
    PLACE(time, kv, attention.k0);
    PLACE(time, kv, attention.v);
    PLACE(time, c.width, attention.q);
    PLACE(time, kv, attention.k);
    if (!tiled) {
        PLACE(score_rows, time, attention.scores);
        PLACE(score_rows, time, attention.probs);
    }
    PLACE(time, c.width, attention.context);
    PLACE(time, c.ffn_width, ffn.gate);
    PLACE(time, c.ffn_width, ffn.value);
    PLACE(time, c.ffn_width, ffn.hidden);
#undef PLACE
    *count = cursor;
    return DL_OK;
}
```

The important branch surrounds **both** score placement calls. Avoiding a call to materialized attention while leaving these two allocations in the arena would not achieve the chapter's memory outcome. No part of the tiled sublayer receives these empty views.

`place` has the same two roles as before. With an empty arena it performs checked counting without constructing pointers; with a live arena it creates borrowed views at the resulting offsets. At a shorter prefix length, the layout is remapped compactly inside the capacity-sized owner. Nothing reads the unused tail. This is disposable full-prefix storage, not capacity-strided persistent history.

### Derive the float count from the actual named buffers

Let $C$ be prefix capacity, $D$ decoder width, $K_w=H_{kv}d$ shared key/value width, and $F$ feed-forward width. Seven $[C,D]$ arrays hold `hidden`, `norm`, `branch`, `middle`, unrotated queries, rotated queries, and context. Three $[C,K_w]$ arrays hold unrotated keys, values, and rotated keys. Three $[C,F]$ arrays hold the feed-forward intermediates.

Therefore the tiled float count is

$$
A_{\rm tiled}=7CD+3CK_w+3CF.
$$

The original full-prefix owner adds two $[H_qC,C]$ arrays:

$$
A_{\rm full}=A_{\rm tiled}+2H_qC^2.
$$

This owner is reused across decoder blocks, so neither expression is multiplied by layer count. The final logits are caller-owned, not hidden in this arena. The tiled double owner contributes

$$
E_{\rm tiled}=3H_qC+RS+Rd+2R.
$$

On a four-byte-float, eight-byte-double build, the new constructor's combined numeric payload is $4A_{\rm tiled}+8E_{\rm tiled}$ bytes. With fixed architecture and fixed tile sizes, doubling $C$ doubles the float arena and the statistic-plane term. The tile/state term stays fixed. The complete test checks these relationships from the actual owner counts, not from an assumed allocator behavior.

## Construct both owners before transferring either

The next block destroys the optional double owner and constructs either strategy. The helper receives zero tile dimensions only from the legacy constructor; the public tiled constructor rejects zeros. It checks the float layout and metadata lengths, then passes only the **remaining** numeric budget to the double constructor.

```c title="src/inference.c"
void dl_inference_destroy(DlInference *s)
{
    if (s == NULL) return;
    dl_tiled_workspace_destroy(&s->tiled);
    free(s->inv_freq);
    free(s->positions);
    dl_tensor_destroy(&s->arena);
    *s = (DlInference){0};
}

static DlStatus create_inference(DlInference *out, const DlWeights *weights,
                                  size_t capacity, size_t qr, size_t kc,
                                  size_t max_float_bytes)
{
    if (out == NULL || weights == NULL || weights->params == NULL ||
        capacity == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (!weights->ready) return DL_STATE;
    if (out->weights != NULL || out->arena.data != NULL ||
        out->positions != NULL || out->inv_freq != NULL || out->capacity != 0 ||
        out->tiled.data != NULL || out->tiled.count != 0 || out->tiled.row_capacity != 0 ||
        out->tiled.head_dim != 0 || out->tiled.query_tile != 0 || out->tiled.key_tile != 0)
        return DL_STATE;
    if (capacity - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlInference next = {.weights = weights, .capacity = capacity};
    Views views = {0};
    size_t count = 0, bytes = 0, position_bytes = 0, frequency_bytes = 0;
    size_t frequencies = weights->config.width / weights->config.query_heads / 2;
    RETURN_IF_ERROR(layout(&next.arena, weights->config, capacity, qr != 0, &views, &count));
    RETURN_IF_ERROR(dl_size_mul(count, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(capacity, sizeof(*next.positions), &position_bytes));
    RETURN_IF_ERROR(dl_size_mul(frequencies, sizeof(*next.inv_freq), &frequency_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    if (qr != 0) {
        size_t rows = 0;
        RETURN_IF_ERROR(dl_size_mul(capacity, weights->config.query_heads, &rows));
        if (bytes == max_float_bytes) return DL_RANGE;
        RETURN_IF_ERROR(dl_tiled_workspace_create(&next.tiled, rows, frequencies * 2,
                          qr, kc, max_float_bytes - bytes));
    }
    DlStatus status = dl_tensor_create(&next.arena, 1, &count);
    if (status == DL_OK) {
        next.positions = malloc(position_bytes);
        next.inv_freq = malloc(frequency_bytes);
        if (next.positions == NULL || next.inv_freq == NULL) status = DL_ALLOC;
    }
    if (status == DL_OK) {
        for (size_t i = 0; i < capacity; ++i) next.positions[i] = i;
        status = dl_rope_frequencies(weights->config.width / weights->config.query_heads,
                                     weights->config.rope_theta, next.inv_freq, frequencies);
    }
    if (status != DL_OK) { dl_inference_destroy(&next); return status; }
    *out = next;
    return DL_OK;
}

DlStatus dl_inference_create(DlInference *out, const DlWeights *weights,
                             size_t capacity, size_t max_float_bytes)
{
    return create_inference(out, weights, capacity, 0, 0, max_float_bytes);
}

DlStatus dl_inference_create_tiled(DlInference *out, const DlWeights *weights,
                                   size_t capacity, size_t qr, size_t kc,
                                   size_t max_payload_bytes)
{
    if (qr == 0 || kc == 0) return DL_SHAPE;
    return create_inference(out, weights, capacity, qr, kc, max_payload_bytes);
}
```

The temporary `next` owns every successful partial allocation. If creating its float arena, positions, frequencies, or RoPE values fails, the common destructor releases both numeric owners and metadata. The caller receives the complete object only at `*out = next`. No workspace pointer retained by the weights points back to this temporary stack object.

A failed exact-budget test is especially useful here. The test computes the accepted combined payload from a successful workspace, then tries one byte less and expects `DL_RANGE` with an empty caller. It tries the exact amount and expects success. This catches a constructor that checks only floats, allocates doubles outside the bound, or transfers partial ownership before all checks have succeeded.

Allocation budgets bound requested numeric payload, not the resident set size of the process. `malloc` bookkeeping, libc, executable pages, metadata, stack frames, and caller outputs are separate. A capacity request can still fail with `DL_ALLOC` even when it is below the declared budget; the budget is not an allocation guarantee.

## Keep one decoder graph and dispatch only attention

The final source block retains the established preflight, embedding lookup, block order, residuals, final normalization, and tied projection. Only the attention call differs. The tiled branch builds a six-view borrowed record from the already mapped linear intermediates.

```c title="src/inference.c"
static DlConstMatrix weight(const DlWeights *w, size_t index)
{
    return dl_matrix_read(w->params[index].value);
}

DlStatus dl_inference_forward(DlInference *s, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits)
{
    if (s == NULL || s->weights == NULL || s->arena.data == NULL || ids == NULL)
        return DL_ARGUMENT;
    if (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED) return DL_ARGUMENT;
    const DlWeights *w = s->weights;
    if (!w->ready) return DL_STATE;
    DlModelConfig c = w->config;
    if (length == 0 || length > s->capacity || first_row >= length) return DL_RANGE;
    size_t end = 0;
    RETURN_IF_ERROR(dl_matrix_end(dl_matrix_read(logits), &end));
    if (logits.cols != c.vocab || logits.rows > length - first_row) return DL_SHAPE;
    if (logits.base == s->arena.data || logits.base == w->arena.data) return DL_ALIAS;
    for (size_t i = 0; i < length; ++i) if (ids[i] >= c.vocab) return DL_RANGE;
    Views v = {0};
    size_t count = 0;
    RETURN_IF_ERROR(layout(&s->arena, c, length, s->tiled.data != NULL, &v, &count));
    DlAttentionConfig a = {1, length, c.query_heads, c.kv_heads, c.width / c.query_heads};
    RETURN_IF_ERROR(dl_embedding_forward(weight(w, 0), ids, length, v.hidden));
    for (size_t layer = 0; layer < c.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        DlAttentionWeights aw = {weight(w, base + DL_P_QUERY), weight(w, base + DL_P_KEY),
            weight(w, base + DL_P_VALUE), weight(w, base + DL_P_OUTPUT)};
        DlFfnWeights fw = {weight(w, base + DL_P_GATE), weight(w, base + DL_P_UP),
                           weight(w, base + DL_P_DOWN)};
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.hidden),
            weight(w, base + DL_P_ATTN_NORM), c.epsilon, v.norm));
        if (s->tiled.data != NULL) {
            DlTiledAttentionCache ta = {v.attention.q0, v.attention.k0, v.attention.v,
                                        v.attention.q, v.attention.k, v.attention.context};
            RETURN_IF_ERROR(dl_attention_tiled_forward(backend, a, dl_matrix_read(v.norm), aw,
                s->positions, length, s->inv_freq, a.head_dim / 2, ta, &s->tiled, v.branch));
        } else {
            RETURN_IF_ERROR(dl_attention_forward(backend, a, dl_matrix_read(v.norm), aw,
                s->positions, length, s->inv_freq, a.head_dim / 2, v.attention, v.branch));
        }
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.hidden), dl_matrix_read(v.branch), v.middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.middle),
            weight(w, base + DL_P_FFN_NORM), c.epsilon, v.norm));
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(v.norm), fw, v.ffn, v.branch));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.middle), dl_matrix_read(v.branch), v.hidden));
    }
    DlMatrix selected = v.hidden;
    selected.offset += first_row * selected.ld; /* Bounded by length above. */
    selected.rows = logits.rows;
    DlMatrix norm = v.norm;
    norm.rows = logits.rows;
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(selected),
        weight(w, w->param_count - 1), c.epsilon, norm));
    return dl_matmul(backend, dl_matrix_read(norm), DL_NOTRANS,
                     weight(w, 0), DL_TRANS, logits, DL_SET);
}

#undef RETURN_IF_ERROR
```

There is no duplicated tiled decoder with a subtly different residual graph. Both branches see the same normalized input, weights, absolute position array, and frequencies. The final selection of output rows also remains identical: full-prefix execution computes every hidden row, but can normalize and project only a chosen subset into caller-owned vocabulary logits.

A successful call may be followed by a shorter or longer prefix call. The next call remaps its views and regenerates every needed value. Saved row statistics from the preceding call or preceding layer are not valid training state. They belong to this one reusable layer workspace and are intentionally overwritten.

This lifetime is sufficient for inference. A later training tape must preserve each layer's own forward state or deliberately recompute it. Merely retaining the final layer's online statistics would not support backpropagation through earlier layers.

## Count everything live, not just one allocation

For the supplied pilot configuration, $D=32$, $H_q=4$, $H_{kv}=2$, $d=8$, and $F=80$. At $C=64$ with $R=16,S=32$, the tiled float arena has 32,768 floats, or 131,072 bytes. Its double owner has 1,440 doubles, or 11,520 bytes. The combined 142,592 bytes should be compared with the materialized full-prefix arena's 262,144 bytes—not with a misleading “16 by 32 tile only” number.

The same call also has the following storage:

| Storage | Count or source of truth | Lifetime |
|---|---|---|
| Model values | `weights.arena.count * sizeof(float)` | Shared by workspace borrowers |
| Weight directory | `weights.param_count * sizeof(DlWeight)` | Same as weight owner |
| Inference float arena | $7CD+3CK_w+3CF$ floats | Workspace capacity |
| Tiled numeric owner | $3H_qC+RS+Rd+2R$ doubles | Workspace capacity |
| Absolute positions | $C$ `size_t` values | Workspace capacity |
| RoPE frequencies | $d/2$ doubles | Workspace capacity |
| Generator input/transcript | $C$ bytes and $C$ `size_t` IDs | Generation request |
| Returned logits | $V$ floats for one selected row | Caller-owned |
| Sampling probabilities | $V$ doubles | Caller-owned |
| Blocked matrix kernel arrays | $8\cdot32$ doubles and $64\cdot32$ floats | During one projection call |

Here $V$ is vocabulary size. The blocked kernel's explicit local arrays total 10,240 bytes on the target representation. They are stack storage, not part of the constructor budget. The scalar backend does not have those arrays. The tiled core has no variable-length array or heap allocation during computation; it uses the already counted double owner and scalar locals.

At the pilot's vocabulary size of 256, weights contain 29,856 floats, or 119,424 bytes; logits use 1,024 bytes and sampling probabilities 2,048. Positions and IDs each use 512 bytes on a 64-bit `size_t` target; frequencies use 32 bytes and the transcript 64. These are requested array sizes, not a measured process footprint. Struct sizes and allocator overhead should be measured with the actual build when reporting process memory.

The cache strategy has a different comparison point: its pilot scratch is 123,392 bytes plus 16,384 bytes of persistent history. That does not contradict the tiled full-prefix saving. The strategies retain different state and perform different work; `tiled` still recalculates all prefix projections.

## Expose the strategy without changing sampling

Replace `app/generate.c` with the complete file below. The optional final argument now accepts `full`, `cache`, or `tiled`. Omitting it retains `full`. The new constructor uses the starting 16-by-32 tile dimensions. The same temperature, RNG, greedy tie policy, checkpoint loading, output refusal, and byte escaping continue to apply.

```c title="app/generate.c"
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
```

Argument parsing still happens before file output. Prompt bytes are not C strings: zero and high-bit bytes remain valid token IDs for the byte model. Exclusive creation protects an existing transcript, and a failed write removes only the new incomplete file created by this invocation.

```c title="app/generate.c"
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
```

The distinction between selection and consumption remains unchanged. With a prompt of $P$ bytes and $N>0$ selections, the transcript has $P+N$ bytes. The last selected token has not been processed to predict another token, so the final consumed prefix length is $P+N-1$. For `NEW=0`, no forward call or RNG draw occurs.

The `workspace` number in the existing payload line remains the float-arena bytes for compatibility. The new `tiled_double_payload_bytes` line supplies the double allocation explicitly. Add them for the tiled numeric workspace total. A zero value under `full` or `cache` is expected; it does not mean those strategies have no scratch.

The program does not promise that small logit differences always preserve samples. A nearly tied greedy maximum or a categorical draw close to a probability boundary can change the selected token. Test logits on identical prefixes first. Sequence equality is useful additional evidence for particular fixtures, not a replacement for numerical comparison.

## Extend the build without changing the floating-point contract

Replace the root Makefile. The new source and header join the common build, `test_tiled` joins the cumulative suite, and the generation CLI target runs all three strategies. There is no fast-math switch to make the online loops appear faster at the cost of changing nonfinite behavior or reduction semantics.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/attention_tiled.c src/rng.c src/model.c src/data.c src/optim.c src/train.c src/checkpoint.c src/inference.c src/sample.c src/cache.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/attention_tiled.h include/dl/rng.h include/dl/model.h include/dl/data.h include/dl/optim.h include/dl/train.h include/dl/checkpoint.h include/dl/inference.h include/dl/sample.h include/dl/cache.h tests/tiny.h tests/training_fixture.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim test_training test_checkpoint test_inference test_sample test_cache test_tiled

.PHONY: all debug sanitize release test test-assets test-training-cli test-generation-cli clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data build/debug/train build/debug/generate
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data build/sanitize/train build/sanitize/generate
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data build/release/train build/release/generate

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

test-assets:
	python3 tests/test_prepare.py

clean:
	rm -rf build

build/debug/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-training-cli: build/release/train
	python3 tests/test_train_cli.py

build/debug/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-generation-cli: build/release/train build/release/generate
	python3 tests/test_generate_cli.py build/release/train build/release/generate full
	python3 tests/test_generate_cli.py build/release/train build/release/generate cache
	python3 tests/test_generate_cli.py build/release/train build/release/generate tiled
```

On the native Mac toolchain, `make test` uses `xcrun clang` as before. An equivalent supported C17 compiler can be selected with `make CC=clang test`. The explicit requirements remain ordinary C17, the math library, and the sanitizer support used by the diagnostic build; no new accelerator library or compiler-version gate is introduced.

The binaries now have a complete tiled inference path. [Part 3](https://25d-228.github.io/gpt-c/chapter-18-part-3.html) establishes its numerical and storage behavior before using it for text generation.
