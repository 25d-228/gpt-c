# Chapter 20 — Bounded C loading and imported position limits

This continues [the mapping and converter](https://25d-228.github.io/gpt-c/chapter-20.html). Begin from the complete Chapter 19 implementation, then apply the replacements below. No tensor operator changes in this chapter.

## A successful load publishes one immutable owner

The state transition is

$$
\text{empty output}\longrightarrow\text{private unready candidate}
\longrightarrow\text{validated and closed file}\longrightarrow\text{ready weights}.
$$

The candidate owns one float arena of $P$ elements and a small directory of borrowed views. It owns neither parameter gradients nor Adam moments. On a read, format, numeric, budget, or close failure, destroy the candidate and leave the caller's empty object unchanged. Do not load into an existing owner while workspaces borrow it; that is a state error even when its new shape would be identical.

Our allocation size comes from the caller's expected `DlModelConfig`, checked by `dl_model_config_validate` and the explicit weight budget. Header dimensions are compared with that geometry, never trusted to resize it. This protects against a file declaring a gigantic tensor, but it does not make a gigantic caller-supplied expected configuration cheap. The budget remains the caller's explicit resource bound.

## Preserve a source position limit

The existing mathematical RoPE implementation accepts positions over a wider numeric range than this source model declares. Those are different bounds. A finite angle is not evidence that a pretrained checkpoint should be used beyond its context limit.

Append `position_limit` to the weight metadata, not to each matrix or cache entry. Zero means the weight owner came from the earlier training/copy path and has no additional imported limit; existing numerical position checks still apply. A loaded imported owner has a positive limit. All workspaces borrow this immutable metadata at a stable address.

Replace `include/dl/inference.h` with the complete header below. The only new data field is `position_limit`; the remaining interfaces are the named Chapter 19 prerequisites. No owner may be copied, moved, or edited while it is borrowed.

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
    size_t position_limit;          /* 0: training origin; imported: max prefix. */
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
 * A nonzero position_limit bounds the capacity of EVERY inference/cache owner.
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

### Keep the empty-owner test complete

Replace `dl_weights_create` and `dl_weights_from_model` in `src/inference.c` with these complete functions. Their only behavioral change is to include `position_limit` in the empty-owner test. Checking the new scalar prevents a partially populated object from being treated as initialized empty storage. The remaining functions and file-scope includes stay as in Chapter 19.

```c title="src/inference.c"
DlStatus dl_weights_create(DlWeights *out, DlModelConfig config,
                           size_t max_float_bytes)
{
    if (out == NULL || max_float_bytes == 0) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->position_limit != 0 || out->ready) return DL_STATE;
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
```

The constructor initializes `DlWeights next = {.config = config};`, so the new field is zero for ordinary training-origin weights. The existing destructor assigns `(DlWeights){0}` after freeing the owners, so repeated destruction resets the field too. No additional allocation or cleanup function is needed.

```c title="src/inference.c"
DlStatus dl_weights_from_model(DlWeights *out, const DlModel *model,
                               size_t max_float_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL) return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->position_limit != 0 || out->ready) return DL_STATE;
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

Replace the static `create_inference` helper next. It is shared by full-prefix and tiled inference, so its new limit check covers both. `capacity` is already checked positive before subtracting one; a disallowed capacity is rejected before allocating any workspace.

```c title="src/inference.c"
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
    if (capacity - 1 > DL_ROPE_MAX_POSITION ||
        (weights->position_limit != 0 && capacity > weights->position_limit))
        return DL_RANGE;
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
```

The same check must cover cached decoding. Replace `include/dl/cache.h` with the following header, then replace the complete `dl_cache_create` function in `src/cache.c` with the implementation beneath it. The rest of `cache.c` is unchanged.

```c title="include/dl/cache.h"
#ifndef DL_CACHE_H
#define DL_CACHE_H

#include "dl/inference.h"

typedef struct {
    const DlWeights *weights;       /* Immutable, stable-address borrow. */
    size_t capacity, length;        /* length counts COMMITTED input tokens. */
    DlTensor history;              /* [layers,2,capacity,Hkv*d]: rotated K, V. */
    DlTensor arena;                /* Reusable prompt/one-row scratch. */
    size_t *positions;             /* capacity absolute positions. */
    double *inv_freq;              /* head_dim/2 frequencies. */
} DlKvCache;

/* Empty {0} owner only. Budget counts BOTH float arenas, not metadata/stack.
 * Positive capacity, last position <= DL_ROPE_MAX_POSITION. A nonzero
 * weights.position_limit also bounds capacity. No allocation
 * during prefill/decode/reset. All borrowing caches/workspaces must die before
 * weights; never copy/move live owners, edit metadata/weights, or use concurrently.
 * Construction failure leaves out unchanged; destroy is repeatable. */
DlStatus dl_cache_create(DlKvCache *out, const DlWeights *weights,
                          size_t capacity, size_t max_float_bytes);
void dl_cache_destroy(DlKvCache *cache);
/* Forget the transcript by setting length=0, not by erasing bytes. Not a wipe. */
DlStatus dl_cache_reset(DlKvCache *cache);

/* B=1, dense causal attention, no eviction, padding mask, dropout or sampling.
 * Prefill requires EMPTY history and 1<=count<=capacity; it processes the full
 * prompt layer by layer. Decode requires NONEMPTY history and consumes exactly
 * one token at absolute position length. Each returns [1,V] logits predicting
 * the token AFTER its last input, not the input itself. Logits are caller-owned
 * with canonical base/count; their ENTIRE allocation must differ from weights,
 * history and scratch. IDs/metadata and all cross-type storage are disjoint.
 * Invalid IDs/backend/shape/alias/capacity/state are rejected before mutation.
 * Success commits length only after every block AND final projection succeed.
 * On later numeric failure: old length and ALL committed K/V entries survive;
 * tentative suffix/scratch and logits are invalid. Retrying valid inputs is safe
 * because every reachable new entry is overwritten before it can be read.
 * Prefill never appends to an existing transcript: call reset first. Reset does
 * not change another cache, caller logits, weights, or sampling RNG state. */
DlStatus dl_cache_prefill(DlKvCache *cache, DlBackend backend,
                           const size_t *ids, size_t count, DlMatrix logits);
DlStatus dl_cache_decode(DlKvCache *cache, DlBackend backend,
                          size_t id, DlMatrix logits);

#endif
```


```c title="src/cache.c"
DlStatus dl_cache_create(DlKvCache *out, const DlWeights *weights,
                          size_t capacity, size_t max_float_bytes)
{
    if (out == NULL || weights == NULL || weights->params == NULL ||
        capacity == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (!weights->ready) return DL_STATE;
    if (out->weights != NULL || out->history.data != NULL || out->arena.data != NULL ||
        out->positions != NULL || out->inv_freq != NULL ||
        out->capacity != 0 || out->length != 0) return DL_STATE;
    if (capacity - 1 > DL_ROPE_MAX_POSITION ||
        (weights->position_limit != 0 && capacity > weights->position_limit))
        return DL_RANGE;
    DlModelConfig c = weights->config;
    size_t params = 0, values = 0;
    RETURN_IF_ERROR(dl_model_config_validate(c, &params, &values));
    size_t d = c.width / c.query_heads, kv = c.kv_heads * d;
    DlKvCache next = {.weights = weights, .capacity = capacity};
    Views views = {0};
    size_t scratch = 0, history = 0, total = 0, bytes = 0;
    size_t position_bytes = 0, frequency_bytes = 0;
    RETURN_IF_ERROR(layout(&next.arena, c, capacity, capacity, &views, &scratch));
    RETURN_IF_ERROR(dl_size_mul(c.layers, 2, &history));
    RETURN_IF_ERROR(dl_size_mul(history, capacity, &history));
    RETURN_IF_ERROR(dl_size_mul(history, kv, &history));
    RETURN_IF_ERROR(dl_size_add(history, scratch, &total));
    RETURN_IF_ERROR(dl_size_mul(total, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_mul(capacity, sizeof(*next.positions), &position_bytes));
    RETURN_IF_ERROR(dl_size_mul(d / 2, sizeof(*next.inv_freq), &frequency_bytes));
    if (bytes > max_float_bytes) return DL_RANGE;
    const size_t shape[] = {c.layers, 2, capacity, kv};
    DlStatus status = dl_tensor_create(&next.history, 4, shape);
    if (status == DL_OK) status = dl_tensor_create(&next.arena, 1, &scratch);
    if (status == DL_OK) {
        next.positions = malloc(position_bytes);
        next.inv_freq = malloc(frequency_bytes);
        if (next.positions == NULL || next.inv_freq == NULL) status = DL_ALLOC;
    }
    if (status == DL_OK)
        status = dl_rope_frequencies(d, c.rope_theta, next.inv_freq, d / 2);
    if (status != DL_OK) { dl_cache_destroy(&next); return status; }
    *out = next;
    return DL_OK;
}
```

This placement matters: reject an excessive capacity before allocating history or scratch. A cache whose capacity is legal already prevents a later decode from exceeding it; no second hidden position counter is introduced. The fixture uses a limit of eight, constructs all three workspace kinds at eight successfully, and rejects nine without publishing an owner.

## Extend the existing checkpoint boundary

Replace `include/dl/checkpoint.h` with this header. We retain the training save/resume APIs and their v1/v2 file formats. The weight-only entry point now recognizes a third **file family**, `DLWGHT01`; it is not training version 3 and cannot resume AdamW.

```c title="include/dl/checkpoint.h"
#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"
#include "dl/inference.h"

/* Versions 1/2: explicit little-endian integers and IEEE binary32/binary64.
 * Materialized training writes the unchanged v1 format. Tiled training writes
 * v2 (magic DLCKPT02, version=2, 24-byte policy extension at byte 256,
 * trailer ENDCKP02). The extension holds attention=1, query_tile, key_tile.
 * Resume must match the saved strategy/tiles; v1 means materialized only.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, matrix backend, attention strategy/tiles, schedule and fixed mode.
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

/* Inference-only read of training v1/v2 or weight-only DLWGHT01. expected config is
 * supplied by the caller, never used unchecked from the file. Empty out only.
 * Allocates P weight floats, not a DlModel, gradients, moments or datasets.
 * Validates header/directory/finite weights/moments/trailer/exact EOF; consumes
 * embedded data in bounded chunks without requiring original dataset files.
 * Unlike resume, does NOT compare dataset identity or import trainer RNG/state.
 * DLWGHT01 contains a 128-byte header, 96-byte canonical directory records,
 * FP32 payload, and ENDWGT01 trailer, never optimizer/run/data state. It sets
 * weights.position_limit; all inference/cache constructors enforce that limit.
 * Canonical names/shapes/counts/offsets are validated before reading payload.
 * max_file_bytes bounds logical file bytes (8-byte magic is read twice, plus
 * one EOF probe); input must be a seekable ordinary file. max_float_bytes
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

### Decode bounded chunks, not arbitrary typed pointers

In `src/checkpoint.c`, insert the following two new static functions immediately **before** `dl_checkpoint_load_weights`. Keep the prior file-scope includes, static assertions, `Io`, `RETURN_IF_ERROR`, transfer/expect helpers, and training functions unchanged.

The imported payload reader has a fixed 4,096-byte stack array. Each successful transfer contains at most 1,024 float words. It assembles a `uint32_t` from explicitly widened bytes and copies that word's representation into a float with `memcpy`. Casting `bytes` to `float *` would make alignment, host byte order, and aliasing assumptions that the file format does not grant.

The platform checks inherited from Chapter 15 require eight-bit bytes and compatible binary32/binary64 representations. Widening before each shift avoids shifting a signed promoted byte into its sign bit. Finiteness is checked before committing each candidate value. A late nonfinite value leaves part of the **private** arena filled, but that candidate will be destroyed rather than exposed.

```c title="src/checkpoint.c"
static DlStatus imported_float_values(Io *io, float *destination, size_t count)
{
    unsigned char bytes[4096];
    while (count != 0) {
        size_t n = count < sizeof(bytes) / 4 ? count : sizeof(bytes) / 4;
        RETURN_IF_ERROR(transfer(io, bytes, 4 * n));
        for (size_t i = 0; i < n; ++i) {
            const unsigned char *p = bytes + 4 * i;
            uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            float value = 0.0f;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) return DL_NUMERIC;
            destination[i] = value;
        }
        destination += n;
        count -= n;
    }
    return DL_OK;
}
```

### Validate every directory entry before reading values

`imported_contents` compares architecture tag, dimensions, counts, normalization settings, rotary base, and initialization metadata against the constructed directory. The imported position limit is validated as positive and representable. It is source metadata within the documented numeric range, not an authentication claim.

After 128 header bytes, the expected remaining work is directory bytes plus $4P$ weight bytes plus the trailer. This must fit the remaining logical file budget before payload processing begins. Every expected directory offset is calculated from checked local counts. No file offset controls a seek, allocation, or destination pointer.

A duplicate or omitted record cannot be hidden by preserving total bytes. The next expected name must match its complete 64-byte zero-padded representation, then its rows, columns, offset, and count must match. Valid records are in arena order, allowing one linear payload decode into the owner.

```c title="src/checkpoint.c"
static DlStatus imported_contents(Io *io, DlWeights *w)
{
    DlModelConfig c = w->config;
    RETURN_IF_ERROR(expect_bytes(io, "DLWGHT01", 8));
    const uint64_t fixed[] = {1, 1, c.vocab, c.layers, c.width, c.query_heads,
        c.kv_heads, c.ffn_width, w->param_count, w->arena.count};
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]); ++i)
        RETURN_IF_ERROR(expect_integer(io, fixed[i], 8));
    uint64_t positions = 0;
    RETURN_IF_ERROR(integer(io, &positions, 8));
    if (positions == 0 || positions > SIZE_MAX || positions - 1 > DL_ROPE_MAX_POSITION)
        return DL_RANGE;
    RETURN_IF_ERROR(expect_float(io, c.epsilon));
    RETURN_IF_ERROR(expect_float(io, c.init_std));
    RETURN_IF_ERROR(expect_double(io, c.rope_theta));
    RETURN_IF_ERROR(expect_integer(io, c.seed, 8));
    size_t directory_bytes = 0, payload = 0, bytes = 0, total = 0;
    RETURN_IF_ERROR(dl_size_mul(w->param_count, 96, &directory_bytes));
    RETURN_IF_ERROR(dl_size_add(128, directory_bytes, &payload));
    RETURN_IF_ERROR(dl_size_mul(w->arena.count, sizeof(float), &bytes));
    RETURN_IF_ERROR(dl_size_add(payload, bytes, &total));
    RETURN_IF_ERROR(dl_size_add(total, 8, &total));
    RETURN_IF_ERROR(expect_integer(io, payload, 8));
    /* Header consumed exactly 128 bytes. Bound remaining work before payload. */
    if ((uint64_t)(total - 128) > io->remaining) return DL_RANGE;
    size_t cursor = payload;
    for (size_t p = 0; p < w->param_count; ++p) {
        DlWeight *param = &w->params[p];
        DlMatrix v = param->value;
        size_t n = v.rows * v.cols; /* Construction checked every product. */
        RETURN_IF_ERROR(expect_bytes(io, param->name, sizeof(param->name)));
        RETURN_IF_ERROR(expect_integer(io, v.rows, 8));
        RETURN_IF_ERROR(expect_integer(io, v.cols, 8));
        RETURN_IF_ERROR(expect_integer(io, cursor, 8));
        RETURN_IF_ERROR(expect_integer(io, n, 8));
        cursor += n * sizeof(float); /* Sum bounded above. */
    }
    /* Directory order is the arena order: no seek using an untrusted offset. */
    RETURN_IF_ERROR(imported_float_values(io, w->arena.data, w->arena.count));
    RETURN_IF_ERROR(expect_bytes(io, "ENDWGT01", 8));
    if (fgetc(io->file) != EOF) return DL_RANGE;
    if (ferror(io->file)) return DL_STATE;
    w->position_limit = (size_t)positions;
    return DL_OK;
}
```

For the tiny fixture, $J=20$, $P=1128$, and the payload begins at $128+20\cdot96=2048$. The shared embedding contains $16\cdot8=128$ floats, so the next tensor begins at $2048+128\cdot4=2560$. The first directory record therefore contains offset 2048 and count 128. Reading its first value uses `arena.data[0]`; reading the next tensor begins at `arena.data[128]`. The directory offsets describe file bytes, whereas arena offsets count floats. Mixing those units is a fourfold indexing error.

The trailer and EOF check reject a truncated or concatenated file. A valid structure containing different finite values still passes. That is intentional: structural checks cannot establish which pretrained run produced a tensor.

### Dispatch without changing the public ownership rule

Replace the complete existing `dl_checkpoint_load_weights` with this function. Preserve the final `#undef RETURN_IF_ERROR` after it.

We inspect eight magic bytes, rewind a seekable ordinary file, and run the appropriate complete parser. For old files, `weight_contents` still validates and discards optimizer/data state exactly as Chapter 19 did. For the new family, it reads only the weight-only representation. Unknown magic reaches the old parser and is rejected there.

The budget measures **logical file size**, not total I/O traffic: the initial eight bytes are read twice and EOF needs one probe. This distinction avoids a misleading claim that at most `max_file_bytes` physical reads occur. The file must not be edited concurrently between the initial read and validation.

```c title="src/checkpoint.c"
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes)
{
    if (path == NULL || path[0] == '\0' || out == NULL || max_file_bytes == 0)
        return DL_ARGUMENT;
    if (out->params != NULL || out->arena.data != NULL ||
        out->param_count != 0 || out->position_limit != 0 || out->ready) return DL_STATE;
    RETURN_IF_ERROR(platform_check());
    DlWeights next = {0};
    RETURN_IF_ERROR(dl_weights_create(&next, expected, max_float_bytes));
    FILE *file = fopen(path, "rb");
    if (file == NULL) { dl_weights_destroy(&next); return DL_STATE; }
    Io io = {file, max_file_bytes, 1};
    unsigned char magic[8];
    DlStatus status = transfer(&io, magic, sizeof(magic));
    if (status == DL_OK && fseek(file, 0, SEEK_SET) != 0) status = DL_STATE;
    if (status == DL_OK) {
        io.remaining = max_file_bytes; /* Budget is logical file length, not I/O traffic. */
        status = memcmp(magic, "DLWGHT01", 8) == 0
            ? imported_contents(&io, &next) : weight_contents(&io, &next);
    }
    if (fclose(file) != 0) status = DL_STATE;
    if (status != DL_OK) { dl_weights_destroy(&next); return status; }
    next.ready = 1;
    *out = next;
    return DL_OK;
}
```

The final assignment happens only after `fclose` succeeds. Before it, `next` owns every resource and `out` owns nothing. After it, `out` owns the arena and directory; the local value `next` is no longer destroyed. This is the same explicit ownership transfer used in earlier constructors, not a deep copy of the weights.

`dl_weights_create` initially zeroes the candidate arena and the loader then fills it. That is two memory-write passes but still **one weight-sized allocation**, not two simultaneous copies. The inspection application will expose the arena count separately from process resident memory.

## Failure behavior to rely on

| Failure | Observable state |
|---|---|
| Nonempty destination | `DL_STATE`; old owner and borrowed addresses unchanged |
| Expected configuration exceeds float budget | `DL_RANGE`; no published owner |
| Wrong version, tag, name, shape, count, or offset | `DL_SHAPE`; candidate destroyed |
| Truncation, trailing bytes, invalid position bound, insufficient logical file budget | `DL_RANGE`; candidate destroyed |
| NaN or infinity in payload | `DL_NUMERIC`; candidate destroyed, including late failures |
| Open, seek, read-error, or close failure | `DL_STATE`; candidate destroyed |

Size arithmetic may also return `DL_OVERFLOW`, and allocation may return `DL_ALLOC`. Never reuse a failed candidate by assuming an earlier tensor was valid. The caller receives either one complete immutable model or its original empty object.

Next, [Part 3](https://25d-228.github.io/gpt-c/chapter-20-part-3.html) supplies independent small files, exact-value checks, workspace integration, and an executable load-only path. No model download is needed to exercise these guarantees.
