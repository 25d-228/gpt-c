# Chapter 12 · Part 2 — Training tape and reverse scheduling

A training tape must preserve the values needed by backward, not merely the latest hidden state. Reusing an input buffer before its derivative has been evaluated can produce an internally consistent-looking but incorrect gradient. We will first make those lifetimes visible in memory, then walk backward through exactly the graph assembled in Part 1.

**Starting state:** keep the Chapter 11 files and every new definition from [Part 1](https://25d-228.github.io/gpt-c/chapter-12.html). Append every `src/model.c` block below to that file, in order with one blank line between blocks. The helpers, headers, configuration, parameter directory, and initialization routines already exist. [Part 3](https://25d-228.github.io/gpt-c/chapter-12-part-3.html) supplies the complete tests and Makefile. This is the second part of the same **4.50-hour** chapter, not an additional estimate.

## Separate saved values from temporary derivatives

Each block needs its original input, both normalized inputs, the middle residual state, and the attention/FFN internal caches until its backward call completes. Each block's final hidden output is the next block's input. The last hidden state is also needed by final normalization backward.

We retain those forward arrays for every block. Backward visits one block at a time, so one attention-gradient workspace and one FFN-gradient workspace can be reused across all blocks. That reuse is safe because parameter gradients and the preceding hidden gradient have already been written before the next block starts.

| Storage | Lifetime |
|---|---|
| `hidden[0..L]` | Embedding output and every block output, through their backward consumers |
| `blocks[ℓ].norm1`, `middle`, `norm2` | Required normalization and projection inputs for block $\ell$ |
| `blocks[ℓ].attention`, `.ffn` | Saved operator intermediates for block $\ell$ |
| `attention_out`, `ffn_out` | Used by forward residual additions; retained here for simple inspection |
| `final_norm`, `logits` | Output projection and loss state |
| `dlogits`, `g`, `other`, `du`, `dn` | Reusable model-level gradient working space |
| `attention_scratch`, `ffn_scratch` | Reused only after the current block has finished consuming them |

The two branch outputs could be reused more aggressively because their backward routines do not need them. We deliberately retain them in this readable materialized reference. Later workspace and tiled-attention chapters will change memory policy explicitly rather than obscure it here.

All float arrays are views into **one** owner, `arena`. They retain the arena's canonical base and total count; their offsets distinguish disjoint slices. This is compatible with the existing alias checks: sharing a base does not mean overlapping spans. Rebasing every slice to a different apparent allocation would hide the true ownership relation.

## Plan the arena before allocating its floats

`place` advances a checked element cursor. During the first layout pass, the arena is empty: it only counts the required elements. After allocation, the same function constructs each view with `dl_matrix_view`. This makes the size calculation and actual layout share one source of truth.

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
```

For the default diagnostic case, each hidden matrix has $4\times16=64$ floats. `hidden[0]` occupies offsets 0–63, and `hidden[1]` occupies 64–127. The first block's `norm1` begins at 128. Its coordinate `[row=2,column=3]` is arena element $128+2\times16+3=163$. Freeing `norm1.base` would therefore free the entire arena, not one normalized row; no view is ever passed to `free`.

Both attention cache and attention scratch have corresponding query/key/value geometries. The following layout helper receives pointers to their view records so it can place either set. Its `score_rows=M*Hq` count equals $BH_qT$; the score/probability matrix then has $T$ columns. The helper is about storage, not the interpretation of gradient versus forward values.

```c title="src/model.c"
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
    RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, scores));
    RETURN_IF_ERROR(place(t, cursor, score_rows, t->attention_config.time, probs));
    return place(t, cursor, m, d, context);
}
```

The complete layout records every matrix. There are no pointers hidden in a helper that owns an undisclosed allocation. The loop's `i<=layers` is safe because the parameter-size validator already rejects layer counts that cannot support the directory; the descriptor allocation separately checks `layers+1`.

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

Let $S=BH_qT^2$ be the number of elements in one score matrix. One attention cache contains $3M(D+K)+2S$ floats. One FFN cache contains $3MF$. The gradient versions have the same total sizes and are allocated once, not once per block. Counting the layout above gives

$$
A=(L+1)\left[6MD+3M(D+K)+2S+3MF\right]+2MV
$$

arena floats. The $6(L+1)MD$ term combines hidden states, the five additional block-width arrays, final normalization, and four model-level gradient arrays. The $2MV$ term is logits plus logit gradients. This count includes some intentionally retained forward outputs; it is not a theoretical minimum.

With $L=1$, $M=4$, $D=16$, $K=8$, $F=40$, $V=16$, and $S=64$, the bracket is $384+288+128+480=1280$, so $A=2\times1280+128=2688$ floats or **10,752 bytes**. Adding parameter/gradient floats gives 34,688 bytes before descriptors, integer metadata, frequencies, allocator overhead, or process overhead. This is a layout calculation, not measured resident memory.

For the later roadmap byte-model shape, $L=4$, $B=2$, $T=128$, $D=256$, $K=64$, $F=704$, $V=256$, the same formula gives **8,650,752 floats, exactly 33 MiB** of arena storage. It is still a quadratic materialized path. Longer sequences grow the score terms as $T^2$, and model weights and optimizer arrays are additional allocations. A float-arena budget bounds this particular workspace; it does not claim to bound the whole process.

## Allocate metadata and make failure cleanup ordinary

`allocate_tape` first creates arrays of empty view records. It can then run the counting pass, compare the float bytes with the caller's budget, allocate the arena, and run the placement pass. A one-byte budget deliberately fails after the two descriptor allocations, allowing the test to check partial cleanup without exhausting memory.

The integer metadata allocation stores input IDs, targets, and positions in three adjacent ranges. Positions are initialized to `row % time`, so each batch item starts again at position zero. They are fixed full-sequence positions in this chapter, not an incremental-decoding counter or a mask based on position values.

```c title="src/model.c"
static DlStatus allocate_tape(DlTrainTape *t, size_t max_float_bytes)
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
    if (bytes > max_float_bytes) return DL_RANGE;
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

The final frequency allocation uses `double`, not float storage disguised as another type. Sizes are checked in the element type that will actually be allocated. `inv_freq` is filled by the existing RoPE routine and remains constant through matching forward/backward calls.

The destructor releases exactly five owners: frequencies, integer metadata, block records, hidden-view records, and the float tensor. It never frees a borrowed subpointer. A partially constructed tape may have any initial prefix of these owners; `free(NULL)` and the empty tensor destructor make the same cleanup routine work in every case.

```c title="src/model.c"
void dl_train_tape_destroy(DlTrainTape *t)
{
    if (t == NULL) return;
    free(t->inv_freq);
    free(t->metadata);
    free(t->blocks);
    free(t->hidden);
    dl_tensor_destroy(&t->arena);
    *t = (DlTrainTape){0};
}

DlStatus dl_train_tape_create(DlTrainTape *out, const DlModel *model,
                             size_t batch, size_t time, size_t max_float_bytes)
{
    if (out == NULL || model == NULL || model->params == NULL ||
        batch == 0 || time == 0 || max_float_bytes == 0) return DL_ARGUMENT;
    if (out->model != NULL || out->arena.data != NULL || out->hidden != NULL ||
        out->blocks != NULL || out->metadata != NULL || out->inv_freq != NULL)
        return DL_STATE;
    if (time - 1 > DL_ROPE_MAX_POSITION) return DL_RANGE;
    DlTrainTape candidate = {0};
    candidate.model = model;
    candidate.attention_config = (DlAttentionConfig){batch, time,
        model->config.query_heads, model->config.kv_heads,
        model->config.width / model->config.query_heads};
    RETURN_IF_ERROR(dl_size_mul(batch, time, &candidate.rows));
    DlStatus status = allocate_tape(&candidate, max_float_bytes);
    if (status != DL_OK) {
        dl_train_tape_destroy(&candidate);
        return status;
    }
    *out = candidate;
    return DL_OK;
}
```

After success, `out->model` borrows the already constructed model's address. Moving the tape's temporary into the output does not invalidate its slices because the allocated arrays and arena stay at their allocated addresses. Moving the **model** afterward would invalidate its identity relationship, which is why that operation is prohibited while tapes exist.

## Adapt the directory to the existing operator interfaces

A block starts at directory offset `1+9*layer`. The following four helpers merely collect borrowed views in the field order required by `attention.h` and `ffn.h`. They do not duplicate weights, transpose data, create gradients, or call hidden numerical code.

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
```

For example, `attention_weights` maps `query`, `key`, `value`, and `output` to the four previously implemented projections. `ffn_weights` maps `gate`, `up`, and `down` into the existing FFN `gate`, `value`, and `down` fields. Their gradient helpers select the parallel gradient owners, never the value owners.

## Track eligible state rather than trusting old bytes

A model starts with a valid zero gradient. `dl_model_zero_grad` restores that accumulation boundary. `dl_model_mark_changed` increments the parameter revision and invalidates gradients; call it **before** any deliberate external value edit, and do not perform the edit if it fails. The increment is checked rather than allowed to wrap, because a wrapped revision could make stale state appear current.

`finite_parameters` scans either values or gradients. The validity flag is a scheduling promise, not a substitute for numerical validation. A caller with a pointer can still write a NaN; the scans and update preflight detect it where needed.

```c title="src/model.c"
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

The state transitions are small enough to inspect directly:

| Operation | Tape readiness | Gradient eligibility |
|---|---|---|
| Construct model | No tape yet | Valid zero gradient |
| Begin any forward attempt | Old tape invalidated | Unchanged |
| Finish successful forward | Ready at current parameter revision | Unchanged |
| Begin actual backward computation | Consumed | Invalid until complete success |
| Finish successful backward | Consumed | Valid SET/ADD result |
| Mark values changed or apply an update | Older tape revisions no longer match | Invalid |
| Clear gradients | Unchanged | Valid zero accumulation boundary |

A backward preflight rejection, such as an invalid mode or invalid ADD boundary, does not consume a still-current tape because no gradient computation began. A new forward attempt does invalidate the previous tape even if its IDs are rejected. These rules are intentional; do not assume every status has identical rollback semantics.

## Forward is the graph, in order

Forward checks the model/tape relationship, backend, row count, IDs, and finite weights before copying IDs/targets. Caller input arrays must be separate from model/tape storage. After copying, the caller may reuse or change its input arrays without changing the saved example that backward will differentiate.

Every block then executes the equations from Part 1. The tape arrays ensure that `hidden[layer]`, `middle`, and both normalized inputs survive. The saved attention metadata is the same for every block because the sequence and RoPE convention are the same, but each block owns different attention activations.

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
        RETURN_IF_ERROR(dl_attention_forward(backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            count, t->inv_freq, t->attention_config.head_dim / 2,
            b->attention, b->attention_out));
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

The final projection explicitly reads `weight(m,0)` again. There is no separate output-head owner or inferred transpose layout. The operator receives `DL_TRANS` for the weight view, producing $ZE^{\mathsf T}$ while the table remains stored $[V,D]$.

The scalar loss is first written into a local `candidate_loss`. Only after every numerical call succeeds are `backend`, `revision`, `ready`, and the caller's loss committed. A later arithmetic error can leave many activation arrays partially written, but the tape is not ready and its results must be discarded. Parameter values and parameter-gradient arrays are not modified by forward.

## Derive the shared table gradient before scheduling backward

Write $\bar A=\partial\mathcal L/\partial A$. At the last projection $O=ZE^{\mathsf T}$, the ordinary matrix derivatives are

$$
\bar Z=\bar O E,\qquad
\bar E_{\mathrm{output}}=\bar O^{\mathsf T}Z.
$$

But changing $E_{v,j}$ also changes every initial hidden entry $X^0_{r,j}$ whose token ID equals $v$. Therefore

$$
\bar E_{v,j}
=\sum_{r=0}^{M-1}\bar O_{r,v}Z_{r,j}
+\sum_{\{r:i_r=v\}}\bar X^0_{r,j}.
$$

The first sum is a matrix product across all token rows. The second is the embedding scatter-add, including repeated IDs. This is the multivariable chain rule applied to two uses of the **same variable**. It is not an approximation and not an average of two gradients. Weight tying's motivation is discussed in [Press and Wolf][tying]; the equation here specifies the computation for this decoder.

For a numerical trace, take two token rows, width two, vocabulary three, and both input IDs equal to 1:

$$
Z=\begin{bmatrix}1&2\\3&-1\end{bmatrix},\qquad
\bar O=\begin{bmatrix}0.1&-0.2&0.1\\-0.3&0.4&-0.1\end{bmatrix}.
$$

Then the output-path gradient is

$$
\bar O^{\mathsf T}Z=
\begin{bmatrix}-0.8&0.5\\1.0&-0.8\\-0.2&0.3\end{bmatrix}.
$$

Suppose backward through the blocks gives $\bar X^0$ rows $(0.2,-0.1)$ and $(0.4,0.5)$. Their lookup contributions both land in table row 1, summing to $(0.6,0.4)$. The final table gradient is

$$
\bar E=
\begin{bmatrix}-0.8&0.5\\1.6&-0.4\\-0.2&0.3\end{bmatrix}.
$$

Using `DL_SET` for the final lookup scatter would erase the first matrix. Updating the table after the output-product derivative but before reversing the blocks would instead differentiate with inconsistent weights. Neither mistake necessarily produces a memory error.

## Reverse residuals without losing either path

For a block's second residual, $X^{\ell+1}=U^\ell+F_\ell(R^\ell)$, the incoming $\bar X^{\ell+1}$ has two destinations. Copy it into $\bar U^\ell$ for the identity path. Pass the same incoming array to FFN backward, then normalization backward, and **add** that normalized-branch contribution into $\bar U^\ell$.

For the first residual, $U^\ell=X^\ell+A_\ell(N^\ell)$, copy $\bar U^\ell$ into $\bar X^\ell$. Reverse attention and its input normalization, then add that branch's contribution into $\bar X^\ell$. The attention routine already adds its Q/K/V input paths, and the FFN routine already adds its gate/value input paths; the model must add the two residual paths around them.

As a scalar trace, suppose the combined derivative of the FFN-normalization branch is 2, the attention-normalization branch is 3, and the incoming derivative is 1. Then $\bar U=1+2\times1=3$ and $\bar X=3+3\times3=12$. Dropping the second-residual identity yields 8 instead; dropping the first-residual identity yields 9. Shapes and finiteness are unchanged.

The model uses four working matrices: `g` is the incoming block derivative, `du` holds the middle residual derivative, `dn` is a reused normalized-input derivative, and `dx` is a borrowed view of `other`, the outgoing block derivative. After a block finishes, swapping the **view records** `g` and `dx` makes the produced derivative the next incoming derivative without copying any floats or changing an owner.

| Reverse operation | Destination policy |
|---|---|
| Loss derivative | `dlogits` SET; mean applied here only |
| Output table derivative | Table gradient uses caller SET/ADD mode |
| Output projection input, final norm input | Scratch SET |
| Second residual identity | `du` SET from `g` |
| FFN input derivative | `dn` SET |
| Second norm input derivative | `du` ADD |
| First residual identity | `dx` SET from `du` |
| Attention input derivative | `dn` SET |
| First norm input derivative | `dx` ADD |
| Lookup derivative after all blocks | Table gradient ADD, regardless of caller mode |

All other parameter gradients use the caller's SET/ADD mode. Scratch is always overwritten. A reverse loop begins with `remaining=L` and runs while it is nonzero; `layer=remaining-1` visits $L-1,\ldots,0$ without the unsigned-underflow trap of testing `layer>=0`.

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
        RETURN_IF_ERROR(dl_attention_backward(t->backend, t->attention_config,
            dl_matrix_read(b->norm1), attention_weights(m, base), t->positions,
            t->rows, t->inv_freq, t->attention_config.head_dim / 2, b->attention,
            dl_matrix_read(t->du), t->attention_scratch, t->dn, DL_SET,
            attention_grads(m, base), mode));
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

The final embedding call always uses `DL_ADD`. For a SET backward, it adds the lookup path to the freshly written output path. For an ADD backward, it adds the lookup path to the previous gradient plus the newly added output path. Neither case needs a second table allocation or a second optimizer entry.

Backward consumes readiness and marks gradients invalid **before** writing `dlogits` or any parameter derivative. If a later FFN or attention call fails, earlier parameter gradients may already have been modified. Do not retry ADD on those bytes. Rebuild from a known accumulation boundary: clear gradients, perform a fresh successful forward, then redo every lost accumulation contribution. The simpler single-batch recovery is a fresh forward followed by SET.

The saved parameter revision prevents an update between forward and backward. It does not prove that somebody has not modified a cache array directly; the header prohibits such edits. The tests deliberately corrupt one saved gate only to exercise the failure contract. Exposed C records make the mechanism inspectable, not tamper-proof.

## Apply one complete, preflighted gradient step

For step size $\eta>0$, the smoke update is

$$
\theta_i'=\theta_i-\eta\bar\theta_i
$$

for every unique parameter coordinate. No momentum, weight decay, clipping, or scheduling is hidden here. First scan every candidate in double precision, checking both inputs, the computed result, and float range. Only then store the candidates. This two-pass design ensures a late NaN or out-of-range value cannot leave half of the model updated.

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

This completes `src/model.c`. The arithmetic is recomputed in the second pass, so the no-concurrent-edit contract matters: the preflighted values must not change between passes. The final revision increment cannot fail after a successful preflight in this single-threaded program, because its overflow condition was checked before stores and nothing else can change the revision.

A small descent step is useful but not sufficient evidence of a correct derivative. Missing one path can still leave a direction that decreases this batch's loss. [Part 3 — Checking the assembled model](https://25d-228.github.io/gpt-c/chapter-12-part-3.html) therefore combines an independent forward calculation, both tied-table contributions, selected coordinates in every parameter family, whole-model directions, two-block reversal, state failures, and exact update accounting.

[tying]: https://arxiv.org/html/1608.05859v3#S3

