# Chapter 17 · Part 2 — Computing and committing a cached step

[Part 1](https://25d-228.github.io/gpt-c/chapter-17.html) allocated separate persistent history and reusable scratch. This part completes `src/cache.c`, then replaces the generation application. It remains part of the same **4.50-hour** chapter. No training operator, sampling rule, checkpoint format, or full-prefix inference function changes.

## Borrow exactly the suffix being written

Append the following helpers to `src/cache.c`. `history_view` is private: its layer, kind, starting row, and row count come from the already-bounded execution loop. The constructor proved the full history product fits `size_t`, so every smaller in-range address also fits. The call to `dl_matrix_view` still checks the resulting geometry.

```c title="src/cache.c"
/* Only constructed, unchanged owners reach these internal helpers. */
static DlConstMatrix weight(const DlWeights *w, size_t index)
{
    return dl_matrix_read(w->params[index].value);
}

static DlStatus history_view(DlKvCache *s, size_t layer, size_t kind,
                               size_t start, size_t rows, DlMatrix *out)
{
    size_t kv = s->weights->config.kv_heads *
                (s->weights->config.width / s->weights->config.query_heads);
    /* The constructor checked 2*L*C*K, and caller bounds every index. */
    size_t offset = ((2 * layer + kind) * s->capacity + start) * kv;
    return dl_matrix_view(&s->history, offset, rows, kv, kv, out);
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}
```

For an append at committed length $S$, make read views of rows `[0,S+n)` and write views of rows `[S,S+n)` for the current block. The key and value planes are different regions of the same owner. Their canonical bases match, but their bounding spans do not overlap. Both differ from every temporary matrix in the scratch owner.

`store_float` protects each numerical storage boundary. A double dot product may be finite yet larger than `FLT_MAX`; accepting it would introduce infinity when casting to float. An error here invalidates tentative results, but does not turn those results into committed history.

## One allowed attention row at a time

The next complete helper receives only private views constructed by our layout code. `v.q.rows` is the number of **new** rows, not total context length. `allowed = start + r + 1` is the causal mask expressed as a loop bound. That distinction matters even during full-prompt prefill, where all prompt keys already occupy history.

```c title="src/cache.c"
/* Materialize one allowed score/probability ROW, not an online softmax.
 * Inputs/views are private, sized by layout/history_view; no unknown aliases. */
static DlStatus attend(DlModelConfig c, size_t start, Views v,
                        DlConstMatrix key, DlConstMatrix value)
{
    size_t d = c.width / c.query_heads, group = c.query_heads / c.kv_heads;
    double scale = 1.0 / sqrt((double)d);
    float *scores = v.scores.base + v.scores.offset;
    float *probs = v.probs.base + v.probs.offset;
    for (size_t r = 0; r < v.q.rows; ++r) {
        size_t allowed = start + r + 1;
        for (size_t h = 0; h < c.query_heads; ++h) {
            size_t kh = h / group;
            size_t qa = v.q.offset + r * v.q.ld + h * d;
            double maximum = -INFINITY;
            for (size_t t = 0; t < allowed; ++t) {
                size_t ka = key.offset + t * key.ld + kh * d;
                double dot = 0.0;
                for (size_t j = 0; j < d; ++j)
                    dot += (double)v.q.base[qa + j] * (double)key.base[ka + j];
                RETURN_IF_ERROR(store_float(&scores[t], dot * scale));
                maximum = fmax(maximum, (double)scores[t]);
            }
            double denominator = 0.0;
            for (size_t t = 0; t < allowed; ++t)
                denominator += exp((double)scores[t] - maximum);
            if (!isfinite(denominator) || denominator <= 0.0) return DL_NUMERIC;
            for (size_t t = 0; t < allowed; ++t)
                RETURN_IF_ERROR(store_float(&probs[t],
                    exp((double)scores[t] - maximum) / denominator));
            for (size_t j = 0; j < d; ++j) {
                double sum = 0.0;
                for (size_t t = 0; t < allowed; ++t)
                    sum += (double)probs[t] *
                        (double)value.base[value.offset + t * value.ld + kh * d + j];
                RETURN_IF_ERROR(store_float(&v.context.base[
                    v.context.offset + r * v.context.ld + h * d + j], sum));
            }
        }
    }
    return DL_OK;
}
```

Read the four passes separately. The first forms scaled scores and their maximum after float storage. The second computes the double denominator. The third stores float probabilities. The fourth contracts those probabilities with unrotated values. Each coordinate's key loop visits positions in ascending order, matching the reference. There is no reduction over other query heads: they share input K/V arrays but produce different probabilities and contexts.

The attention row always contains at least its diagonal token, so its allowed set is nonempty. Subtracting the maximum makes at least one exponential equal to one. The denominator check protects the invariant without silently fabricating an empty distribution. Slots beyond `allowed` are neither read nor required to contain a mask sentinel; they are outside this private row's logical extent.

## A three-token calculation with shared values

Use one block, $D=4$, two query heads of width $d=2$, one key/value head, and epsilon $10^{-5}$. Set both normalization scales to one, $W_q=W_o=I_4$, let $W_k$ and $W_v$ select the first two coordinates, and set all feed-forward matrices to zero. The four vocabulary rows are

$$
E=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
1&1&0&0\\
0&0&0&0
\end{bmatrix}.
$$

Prefill IDs `[0,1]`, then decode ID `2`. The normalized first two coordinates before attention are approximately $(1.999960,0)$, $(0,1.999960)$, and $(1.414199,1.414199)$. Since $d=2$, the single frequency is one; the rotary angles are zero, one, and two radians. The cached rows are therefore:

| Position | Rotated key | Unrotated value |
|---:|---|---|
| 0 | $(1.999960,0)$ | $(1.999960,0)$ |
| 1 | $(-1.682908,1.080583)$ | $(0,1.999960)$ |
| 2 | $(-1.874443,0.697413)$ | $(1.414199,1.414199)$ |

For the third token's first query head, the scaled scores are approximately $[-2.650809,2.763464,2.828371]$. Subtracting the last score and normalizing gives probabilities $[0.00214943,0.48273912,0.51511144]$. Its context is $(0.732769,1.693929)$.

The second query head is zero, so its three scores are zero and its probabilities are exactly the uniform mathematical distribution. It reads the **same values**, producing approximately $(1.138053,1.138053)$. Group sharing did not require its probabilities to match the first head's. After concatenation, identity output projection, and the residual, the hidden row is

$$
x=(1.732769,\;2.693929,\;1.138053,\;1.138053).
$$

The zero feed-forward output leaves it unchanged. Final RMSNorm and the tied table give logits approximately

$$
z=(0.96675674,\;1.50301288,\;2.46976961,\;0).
$$

These are logits for the token **after** ID `2`. They are not probabilities and are not a reconstruction target for ID `2`. The test in Part 3 uses these independently calculated values, as well as both K/V planes, rather than relying solely on agreement with another decoder loop. Small differences from displayed decimals are expected because actual intermediate arrays store float values.

## Execute a step without publishing half a prefix

The internal `run` function handles $n$ new rows: all prompt rows for prefill, one row for decode. Public wrappers enforce which state permits each case. Before writing, `run` checks the backend, all IDs, capacity, output geometry, and allocation separation. A `[1,V]` output may have an offset or padding, provided its view is truthful.

Follow the block loop in three stages. First normalize and project only the new rows, then rotate their Q/K with absolute positions. Second attend over the visible history and merge the attention residual. Third run the feed-forward branch and its residual. The reused `hidden` array then becomes the new rows' input to the next block. No old hidden row needs to survive because its per-block K/V has already been retained.

```c title="src/cache.c"
static DlStatus run(DlKvCache *s, DlBackend backend, const size_t *ids,
                      size_t count, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (ids == NULL || (backend != DL_BACKEND_REF && backend != DL_BACKEND_BLOCKED))
        return DL_ARGUMENT;
    const DlWeights *w = s->weights;
    DlModelConfig c = w->config;
    if (count == 0 || count > s->capacity - s->length) return DL_RANGE;
    size_t end = 0;
    RETURN_IF_ERROR(dl_matrix_end(dl_matrix_read(logits), &end));
    if (logits.rows != 1 || logits.cols != c.vocab) return DL_SHAPE;
    if (logits.base == s->history.data || logits.base == s->arena.data ||
        logits.base == w->arena.data) return DL_ALIAS;
    for (size_t i = 0; i < count; ++i) if (ids[i] >= c.vocab) return DL_RANGE;
    Views v = {0};
    size_t scratch = 0, start = s->length, next = start + count;
    RETURN_IF_ERROR(layout(&s->arena, c, count, s->capacity, &v, &scratch));
    for (size_t i = 0; i < count; ++i) s->positions[i] = start + i;
    RETURN_IF_ERROR(dl_embedding_forward(weight(w, 0), ids, count, v.hidden));
    for (size_t layer = 0; layer < c.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        size_t d = c.width / c.query_heads;
        DlMatrix key = {0}, value = {0}, new_key = {0}, new_value = {0};
        RETURN_IF_ERROR(history_view(s, layer, 0, 0, next, &key));
        RETURN_IF_ERROR(history_view(s, layer, 1, 0, next, &value));
        RETURN_IF_ERROR(history_view(s, layer, 0, start, count, &new_key));
        RETURN_IF_ERROR(history_view(s, layer, 1, start, count, &new_value));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.hidden),
            weight(w, base + DL_P_ATTN_NORM), c.epsilon, v.norm));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_QUERY), DL_TRANS, v.q0, DL_SET));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_KEY), DL_TRANS, v.k0, DL_SET));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.norm), DL_NOTRANS,
            weight(w, base + DL_P_VALUE), DL_TRANS, new_value, DL_SET));
        RETURN_IF_ERROR(dl_rope_forward(dl_matrix_read(v.q0), d, s->positions,
            count, s->inv_freq, d / 2, v.q));
        RETURN_IF_ERROR(dl_rope_forward(dl_matrix_read(v.k0), d, s->positions,
            count, s->inv_freq, d / 2, new_key));
        RETURN_IF_ERROR(attend(c, start, v, dl_matrix_read(key), dl_matrix_read(value)));
        RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(v.context), DL_NOTRANS,
            weight(w, base + DL_P_OUTPUT), DL_TRANS, v.branch, DL_SET));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.hidden), dl_matrix_read(v.branch), v.middle));
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.middle),
            weight(w, base + DL_P_FFN_NORM), c.epsilon, v.norm));
        DlFfnWeights fw = {weight(w, base + DL_P_GATE), weight(w, base + DL_P_UP),
                           weight(w, base + DL_P_DOWN)};
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(v.norm), fw, v.ffn, v.branch));
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.middle), dl_matrix_read(v.branch), v.hidden));
    }
    DlMatrix last = v.hidden, norm = v.norm;
    last.offset += (count - 1) * last.ld;
    last.rows = norm.rows = 1;
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(last),
        weight(w, w->param_count - 1), c.epsilon, norm));
    RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(norm), DL_NOTRANS,
        weight(w, 0), DL_TRANS, logits, DL_SET));
    s->length = next;             /* Sole commit point, including final logits. */
    return DL_OK;
}
```

There are two deliberate asymmetries. First, the value projection and key rotation write directly to persistent memory, but only to its uncommitted suffix. Second, only the last new hidden row is normalized and projected to vocabulary logits. Prefill still computes every prompt hidden row through every block, because later prompt tokens depend on earlier rows' K/V. It merely avoids producing unused vocabulary logits for earlier prompt positions.

The sole assignment to `s->length` comes **after** the final vocabulary projection. A block may have succeeded while a later normalization or output projection overflows. Committing after the last block but before the final logits would still incorrectly report success for an unusable prediction.

The two wrappers are small but necessary. Decode on an empty cache does not silently become prefill, and prefill on a nonempty cache does not silently replace the transcript. End the file with these wrappers and the macro cleanup.

```c title="src/cache.c"
DlStatus dl_cache_prefill(DlKvCache *s, DlBackend backend,
                           const size_t *ids, size_t count, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (s->length != 0) return DL_STATE;
    return run(s, backend, ids, count, logits);
}

DlStatus dl_cache_decode(DlKvCache *s, DlBackend backend,
                          size_t id, DlMatrix logits)
{
    RETURN_IF_ERROR(live(s));
    if (s->length == 0) return DL_STATE;
    return run(s, backend, &id, 1, logits);
}

#undef RETURN_IF_ERROR
```

## What survives each failure?

The old committed length $S$ remains visible throughout a call. Read views may temporarily cover `[0,S+n)`, but their new suffix is populated in the current block before attention reads it. No write view ever begins before $S$.

| Result | Public length | Old committed K/V | Tentative suffix and output |
|---|---:|---|---|
| Argument, state, capacity, shape, or alias rejection | unchanged | unchanged | unchanged |
| Numerical failure after computation starts | unchanged | unchanged | discard; some cells may be written |
| Complete success, including logits | $S+n$ | unchanged | new suffix committed; logits valid |
| Reset | 0 | old bytes retained but no longer visible | old logits remain caller-owned |

This is not a copy-on-write snapshot or an undo log. We preserve the prefix by restricting writes, and preserve logical validity by withholding the new length. A failed suffix is safe to reuse because a subsequent call overwrites every new key/value coordinate it will make visible before reading it. A failing input will generally fail again; retry with a valid input or reset and start a valid prompt. Never interpret logits returned with an error status.

If selecting a token fails **after** a successful cached forward, the cache is still committed through its input. Chapter 16's sampler preserves RNG state on its own failure. A caller repairing only the sampling request should reuse the already-valid logits rather than consume the same input again. The command-line application simply aborts on any error; it does not implement automatic retry or cache serialization.

## Integrate without changing the sampling distribution

Replace `app/generate.c` with the next two blocks. Its existing numeric parser, binary prompt reader, and exclusive-output writer remain fully specified. `dl/cache.h` is explicitly included; relying on a transitive include would hide the application's new dependency.

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

#define TRY(call) do { status = (call); if (status != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status)); goto done; \
} } while (0)
```

The final command-line argument is optional: `full` retains Chapter 16's baseline and is the default; `cache` selects the new stateful path. These are inference strategies, separate from the `ref|blocked` matrix backend and `greedy|sample` selection mode.

In cached mode, prefill runs once only when at least one new token is requested. At step zero the returned logits are already ready. At later steps, consume `ids[length-1]`, the most recently selected token. Then select exactly once. The sampler sees the same logits and the same sequence of RNG states as the baseline when numerical comparisons agree.

```c title="app/generate.c"
int main(int argc, char **argv)
{
    if ((argc != 11 && argc != 12) ||
        (argc == 12 && strcmp(argv[11], "full") != 0 && strcmp(argv[11], "cache") != 0) ||
        (strcmp(argv[7], "greedy") != 0 && strcmp(argv[7], "sample") != 0) ||
        (strcmp(argv[10], "ref") != 0 && strcmp(argv[10], "blocked") != 0)) {
        fprintf(stderr, "usage: %s CONFIG CHECKPOINT PROMPT OUTPUT CAPACITY NEW "
                        "greedy|sample TEMPERATURE SEED ref|blocked [full|cache]\n", argv[0]);
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
           cached ? "cache" : "full", cached ? cache.length : (requested ? length - 1 : 0),
           cache.history.count * sizeof(float));
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

The two workspaces are empty at declaration, but only the chosen strategy allocates one. The report separates temporary float payload from persistent K/V payload. `consumed` is zero for zero-new-token requests, and otherwise excludes the final selected token. Although the cache API can consume exactly $C$ inputs and return one more prediction, the application does not allocate an extra transcript slot beyond its declared capacity.

The application still reads nonempty raw byte prompts, requires a byte vocabulary, refuses transcript overflow, writes raw bytes to a newly created file only after successful generation, and escapes terminal presentation separately. No implicit beginning/end token, truncation, sliding window, or tokenizer change is introduced. Existing commands without the optional final argument continue to use full-prefix inference.

Continue with [Part 3 — Checking cache equivalence and failure recovery](https://25d-228.github.io/gpt-c/chapter-17-part-3.html).
