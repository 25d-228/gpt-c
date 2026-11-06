# Chapter 21, Part 2 — Observing C execution and selecting vocabulary IDs

[Part 1](https://25d-228.github.io/gpt-c/chapter-21.html) defined the token and trace boundaries. Now C consumes those IDs and executes the existing model. We add observation to the full-prefix inference workspace, not a second decoder. The persistent cache keeps its existing implementation; a prefix comparison will check its results without exposing private cache pointers.

## Observe before the workspace reuses a buffer

Chapter 16 deliberately reused one block's storage across every layer. A pointer to the normalized input from layer zero will therefore refer to something else after a later normalization. Storing pointers in a list and printing them after forward completes would produce a misleading trace, even though no allocation has been freed.

An observer must run **synchronously at the point of observation**. It may read the borrowed view until it returns. To keep values longer, it must copy them; this chapter's observer serializes them immediately. It must not change weights, IDs, outputs or workspace storage, and must not call inference recursively on these owners. C's `const` interface documents read-only access but cannot protect against a callback deliberately casting it away.

Add the following declarations immediately before the final `#endif` in `include/dl/inference.h`. `DlTraceFn` is a function-pointer type. Its `void *context` is a borrowed caller-defined object, allowing a file writer or a test recorder without global state. A non-OK return aborts the forward call. A NULL function pointer disables observation.

```c title="include/dl/inference.h"
/* Synchronous read-only observation. Each view is borrowed ONLY until return;
 * copy values to retain them. Callback must not mutate/reenter any live owner,
 * weights, input IDs, or output. A callback failure stops the forward call;
 * discard ALL logits and trace records from that failed attempt. NULL disables
 * observation. No trace allocations; the ordinary forward wrapper is unchanged.
 * Block index is zero-based; final_norm/logits use config.layers. */
typedef DlStatus (*DlTraceFn)(void *context, const char *name, size_t index,
                               DlConstMatrix value);
DlStatus dl_inference_forward_trace(DlInference *workspace, DlBackend backend,
                                    const size_t *ids, size_t length,
                                    size_t first_row, DlMatrix logits,
                                    DlTraceFn trace, void *context);

```

## One forward schedule, with explicit observation points

In `src/inference.c`, replace the old complete `dl_inference_forward` definition with the following two definitions. Keep all earlier includes, private `Views`, `layout`, `weight`, constructors and destruction code from Chapter 20 unchanged, and retain the existing final `#undef RETURN_IF_ERROR`. The complete replacement is shown: no new compute helper is omitted.

The local `TRACE` macro calls the observer with a `DlConstMatrix`. Its `do { ... } while (0)` form behaves like one statement even in a conditional. It is undefined after this function. The existing `RETURN_IF_ERROR` propagates the first non-OK status; a callback is part of the same failure boundary as a numerical operator.

Query, key, value and attention-context views remain live after their attention call, so we can observe them without instrumenting the kernel. Scores and probabilities are intentionally absent: they are not materialized by tiled attention. Similarly, the feed-forward cache exposes gate projection, value projection and multiplied hidden values before the next block overwrites them.

```c title="src/inference.c"
DlStatus dl_inference_forward_trace(DlInference *s, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits,
                              DlTraceFn trace, void *context)
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
#define TRACE(name_, index_, matrix_) do { \
    if (trace != NULL) RETURN_IF_ERROR(trace(context, name_, index_, \
                                              dl_matrix_read(matrix_))); \
} while (0)
    RETURN_IF_ERROR(dl_embedding_forward(weight(w, 0), ids, length, v.hidden));
    TRACE("embedding", 0, v.hidden);
    for (size_t layer = 0; layer < c.layers; ++layer) {
        size_t base = 1 + layer * DL_BLOCK_PARAMS;
        DlAttentionWeights aw = {weight(w, base + DL_P_QUERY), weight(w, base + DL_P_KEY),
            weight(w, base + DL_P_VALUE), weight(w, base + DL_P_OUTPUT)};
        DlFfnWeights fw = {weight(w, base + DL_P_GATE), weight(w, base + DL_P_UP),
                           weight(w, base + DL_P_DOWN)};
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.hidden),
            weight(w, base + DL_P_ATTN_NORM), c.epsilon, v.norm));
        TRACE("attn_norm", layer, v.norm);
        if (s->tiled.data != NULL) {
            DlTiledAttentionCache ta = {v.attention.q0, v.attention.k0, v.attention.v,
                                        v.attention.q, v.attention.k, v.attention.context};
            RETURN_IF_ERROR(dl_attention_tiled_forward(backend, a, dl_matrix_read(v.norm), aw,
                s->positions, length, s->inv_freq, a.head_dim / 2, ta, &s->tiled, v.branch));
        } else {
            RETURN_IF_ERROR(dl_attention_forward(backend, a, dl_matrix_read(v.norm), aw,
                s->positions, length, s->inv_freq, a.head_dim / 2, v.attention, v.branch));
        }
        TRACE("query", layer, v.attention.q0);
        TRACE("key", layer, v.attention.k0);
        TRACE("value", layer, v.attention.v);
        TRACE("rotated_query", layer, v.attention.q);
        TRACE("rotated_key", layer, v.attention.k);
        TRACE("context", layer, v.attention.context);
        TRACE("attention", layer, v.branch);
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.hidden), dl_matrix_read(v.branch), v.middle));
        TRACE("middle", layer, v.middle);
        RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(v.middle),
            weight(w, base + DL_P_FFN_NORM), c.epsilon, v.norm));
        TRACE("ffn_norm", layer, v.norm);
        RETURN_IF_ERROR(dl_ffn_forward(backend, dl_matrix_read(v.norm), fw, v.ffn, v.branch));
        TRACE("gate", layer, v.ffn.gate);
        TRACE("up", layer, v.ffn.value);
        TRACE("ffn_hidden", layer, v.ffn.hidden);
        TRACE("ffn", layer, v.branch);
        RETURN_IF_ERROR(dl_residual_add(dl_matrix_read(v.middle), dl_matrix_read(v.branch), v.hidden));
        TRACE("hidden", layer, v.hidden);
    }
    DlMatrix selected = v.hidden;
    selected.offset += first_row * selected.ld; /* Bounded by length above. */
    selected.rows = logits.rows;
    DlMatrix norm = v.norm;
    norm.rows = logits.rows;
    RETURN_IF_ERROR(dl_rmsnorm_forward(dl_matrix_read(selected),
        weight(w, w->param_count - 1), c.epsilon, norm));
    TRACE("final_norm", c.layers, norm);
    RETURN_IF_ERROR(dl_matmul(backend, dl_matrix_read(norm), DL_NOTRANS,
                              weight(w, 0), DL_TRANS, logits, DL_SET));
    TRACE("logits", c.layers, logits);
#undef TRACE
    return DL_OK;
}

DlStatus dl_inference_forward(DlInference *s, DlBackend backend,
                              const size_t *ids, size_t length,
                              size_t first_row, DlMatrix logits)
{
    return dl_inference_forward_trace(s, backend, ids, length, first_row,
                                      logits, NULL, NULL);
}

```

The ordinary wrapper now calls the traced function with two NULL arguments. Its arithmetic and ownership rules are unchanged, and observation adds no tensor allocation. The only extra work with observation disabled is testing whether the callback exists.

For $L$ blocks, the trace contains

$$
1+15L+2=15L+3
$$

tensor records: the embedding result, fifteen stages per block, final normalization and logits. Thus a two-block model gives 33 records. Per-block indices are zero-based; final normalization and logits use index $L$. Intermediate records contain every prompt row. Final normalization and projection contain only the requested rows, which the application below sets to the final row.

A callback can fail even after logits have been written. **Discard the entire attempted result and every record from that call.** The writer will remove its partial output. A new full-prefix call is safe because it overwrites disposable scratch; a failed observation is not permission to continue a half-complete trace.

## A vocabulary-ID application beside the existing timing application

Create `app/pretrained.h`. This small declaration separates command dispatch from the existing kernel timing implementation.

```c title="app/pretrained.h"
#ifndef DL_APP_PRETRAINED_H
#define DL_APP_PRETRAINED_H
int dl_compare_pretrained(int argc, char **argv);
#endif
```

Create `app/pretrained.c` from the next four listings. The ID loader uses explicit little-endian byte operations instead of a packed native struct. It accepts only a complete file with the expected vocabulary, token range and trailer. It writes the count only after successful close. A truncated file may overwrite local temporary IDs, but none are used unless the loader succeeds.

`number` accepts decimal digits only and checks overflow before multiplying by ten. In particular, a negative count cannot be converted into a huge unsigned capacity, and trailing junk is not silently ignored.

```c title="app/pretrained.c"
#include "pretrained.h"
#include "dl/cache.h"
#include "dl/checkpoint.h"
#include "dl/sample.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_IDS = 256, MAX_TRACE_TOKENS = 16 };

typedef struct {
    FILE *file;
    size_t records, values;
} Trace;

static int number(const char *text, size_t *out)
{
    if (*text == '\0') return 0;
    size_t n = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9') return 0;
        size_t digit = (size_t)(*text - '0');
        if (n > (SIZE_MAX - digit) / 10) return 0;
        n = n * 10 + digit;
    }
    *out = n;
    return 1;
}

static int read_uint(FILE *file, size_t bytes, uint64_t *out)
{
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) {
        int byte = fgetc(file);
        if (byte == EOF) return 0;
        value |= (uint64_t)(unsigned char)byte << (8 * i);
    }
    *out = value;
    return 1;
}

static int write_uint(FILE *file, uint64_t value, size_t bytes)
{
    for (size_t i = 0; i < bytes; ++i)
        if (fputc((int)((value >> (8 * i)) & UINT64_C(255)), file) == EOF) return 0;
    return 1;
}

static int load_ids(const char *path, size_t vocab, size_t ids[MAX_IDS], size_t *count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) return 0;
    char magic[8];
    uint64_t v = 0, n = 0, token = 0;
    int ok = fread(magic, 1, 8, file) == 8 && memcmp(magic, "DLIDS001", 8) == 0 &&
             read_uint(file, 8, &v) && v == vocab && read_uint(file, 8, &n) &&
             n > 0 && n <= MAX_IDS;
    for (size_t i = 0; ok && i < (size_t)n; ++i) {
        ok = read_uint(file, 4, &token) && token < vocab;
        if (ok) ids[i] = (size_t)token;
    }
    if (ok) ok = fread(magic, 1, 8, file) == 8 && memcmp(magic, "ENDIDS01", 8) == 0 &&
                 fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) ok = 0;
    if (ok) *count = (size_t)n;
    return ok;
}

static int save_ids(FILE *file, size_t vocab, const size_t *ids, size_t count)
{
    if (fwrite("DLIDS001", 1, 8, file) != 8 || !write_uint(file, vocab, 8) ||
        !write_uint(file, count, 8)) return 0;
    for (size_t i = 0; i < count; ++i)
        if (!write_uint(file, ids[i], 4)) return 0;
    return fwrite("ENDIDS01", 1, 8, file) == 8 && !ferror(file);
}

```

The trace writer records model geometry and the actual numerical strategy. The normalization epsilon is printed after widening the **stored float** to double; the reference will use that same configuration value rather than compare a source decimal with its binary32 approximation. Tensor values use nine significant digits, with the binary32 readback from Part 1.

All stage names come from fixed implementation strings. Passing an arbitrary user string to this formatter would require JSON escaping; that is not its interface. The observer respects offsets and leading dimensions instead of assuming a compact view. It rejects nonfinite outputs and bounds the total trace to 8,388,608 values. A short trace is a diagnostic, not a request to dump every activation from an arbitrarily long prompt.

```c title="app/pretrained.c"
static int header(FILE *file, const char *mode, DlModelConfig c,
                   const size_t *ids, size_t n, size_t start,
                   const char *strategy, const char *backend)
{
    fprintf(file, "{\"kind\":\"header\",\"version\":1,\"mode\":\"%s\",\"ids\":[", mode);
    for (size_t i = 0; i < n; ++i) fprintf(file, "%s%zu", i ? "," : "", ids[i]);
    fprintf(file, "],\"geometry\":[%zu,%zu,%zu,%zu,%zu,%zu],\"start\":%zu,\"epsilon\":%.17g,\"theta\":%.17g,"
                  "\"settings\":{\"engine\":\"c\",\"backend\":\"%s\","
                  "\"attention\":\"%s\",\"float_stores\":32,\"reductions\":64}}\n",
            c.vocab, c.layers, c.width, c.query_heads, c.kv_heads, c.ffn_width,
            start, (double)c.epsilon, c.rope_theta, backend, strategy);
    return !ferror(file);
}

static DlStatus emit(void *context, const char *name, size_t index, DlConstMatrix a)
{
    Trace *trace = context;
    size_t count = 0, total = 0;
    if (dl_size_mul(a.rows, a.cols, &count) != DL_OK ||
        dl_size_add(trace->values, count, &total) != DL_OK ||
        total > (size_t)8 * 1024 * 1024) return DL_RANGE;
    /* All names originate in our implementation, not an unescaped user string. */
    fprintf(trace->file, "{\"kind\":\"tensor\",\"name\":\"%s\",\"index\":%zu,"
                         "\"rows\":%zu,\"cols\":%zu,\"values\":[", name, index, a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            double x = (double)a.base[a.offset + i * a.ld + j];
            if (!isfinite(x)) return DL_NUMERIC;
            fprintf(trace->file, "%s%.9g", i || j ? "," : "", x);
        }
    }
    fputs("]}\n", trace->file);
    if (ferror(trace->file)) return DL_STATE;
    ++trace->records;
    trace->values = total;
    return DL_OK;
}

```

## Forced prefixes isolate cache correctness from token choice

Consider a fixed transcript `[4,9,2,7,3]`, and begin comparison at prefix length $P=3$. We first evaluate `[4,9,2]`, then `[4,9,2,7]`, then all five IDs. Full-prefix and tiled execution recompute each supplied prefix. Cached execution prefills the first three IDs once, then consumes ID 7 and ID 3 individually. All three methods therefore condition each prediction on identical inputs.

The cache itself owns the consumed length. `logits_at` does not infer it from a transcript file or overwrite it to make a mismatch disappear. On a subsequent cached call, the requested prefix must be exactly one longer than the committed cache. An error is returned before pretending that the next row is a continuation.

```c title="app/pretrained.c"
static DlStatus logits_at(DlInference *full, DlKvCache *cache, DlBackend backend,
                           const size_t *ids, size_t n, DlMatrix logits)
{
    if (cache->weights == NULL)
        return dl_inference_forward(full, backend, ids, n, n - 1, logits);
    if (cache->length == 0) return dl_cache_prefill(cache, backend, ids, n, logits);
    if (cache->length + 1 != n) return DL_STATE;
    return dl_cache_decode(cache, backend, ids[n - 1], logits);
}

```

The command has three operations:

| Operation | Result |
|---|---|
| `trace` | All named intermediate stages for one short full/tiled prefix, plus its final-row logits |
| `prefix` | Complete vocabulary logits at each fixed prefix length from `START` through the supplied transcript |
| `generate` | Greedy IDs selected entirely by C, appended to the supplied prompt |

The trace operation permits at most 16 input IDs. The file/transcript boundary permits 256, and generation permits at most 32 newly selected IDs. These are **application diagnostic limits**, not claims about the model's trained context length. Imported position limits and workspace byte budgets remain independently enforced by the constructors.

For a prompt of $P$ IDs and $G$ selected IDs, the largest consumed prefix is $P+G-1$. The final selected ID is written to the output but need not be fed back into the model. If a newly selected EOS appears, it is included in the transcript and generation ends without consuming it. An EOS-numbered ID already in the prompt is not a stopping event.

The loaded weights are immutable and borrowed by either `DlInference` or `DlKvCache`. The vocabulary-logit tensor belongs to the application. Destruction reverses those dependencies: free output, destroy either workspace, then free weights. No optimizer, parameter gradients, training tape, or model computation in Python is involved.

```c title="app/pretrained.c"
int dl_compare_pretrained(int argc, char **argv)
{
    /* argv: program pretrained operation config weights ids output strategy backend [count [eos]] */
    if (argc < 9) goto usage;
    int tracing = strcmp(argv[2], "trace") == 0;
    int prefixes = strcmp(argv[2], "prefix") == 0;
    int generate = strcmp(argv[2], "generate") == 0;
    size_t start = 0, steps = 0, eos = 0;
    int has_eos = 0;
    if ((!tracing && !prefixes && !generate) ||
        (tracing && argc != 9) || (prefixes && argc != 10) || (generate && argc != 11)) goto usage;
    if (prefixes && (!number(argv[9], &start) || start == 0)) goto usage;
    if (generate && (!number(argv[9], &steps) || steps == 0 || steps > 32)) goto usage;
    if (generate && strcmp(argv[10], "none") != 0) {
        if (!number(argv[10], &eos)) goto usage;
        has_eos = 1;
    }
    int cached = strcmp(argv[7], "cache") == 0;
    int tiled = strcmp(argv[7], "tiled") == 0;
    if ((!cached && !tiled && strcmp(argv[7], "full") != 0) || (tracing && cached)) goto usage;
    DlBackend backend = DL_BACKEND_REF;
    if (strcmp(argv[8], "blocked") == 0) backend = DL_BACKEND_BLOCKED;
    else if (strcmp(argv[8], "ref") != 0) goto usage;
    DlModelConfig c = {0};
    DlWeights weights = {0};
    DlInference full = {0};
    DlKvCache cache = {0};
    DlTensor output = {0};
    FILE *file = NULL;
    size_t ids[MAX_IDS], n = 0, batch = 0, time = 0, capacity = 0;
    int result = 1, created = 0;
    DlStatus s = dl_model_config_load(argv[3], &c, &batch, &time);
    if (s != DL_OK) goto done;
    if (c.vocab > 100000 || !load_ids(argv[5], c.vocab, ids, &n)) {
        fputs("invalid token file or mismatched vocabulary\n", stderr); goto done;
    }
    if ((tracing && n > MAX_TRACE_TOKENS) || (prefixes && start > n) ||
        (generate && (steps > MAX_IDS - n || (has_eos && eos >= c.vocab)))) {
        fputs("invalid start, trace length, generation length, or EOS\n", stderr); goto done;
    }
    if (tracing) start = n;
    capacity = generate ? n + steps - 1 : n; /* Last selected token is not consumed. */
    s = dl_checkpoint_load_weights(argv[4], &weights, c,
          UINT64_C(2) * 1024 * 1024 * 1024, (size_t)1024 * 1024 * 1024);
    if (s != DL_OK) goto done;
    const size_t budget = (size_t)256 * 1024 * 1024;
    if (cached) s = dl_cache_create(&cache, &weights, capacity, budget);
    else if (tiled) s = dl_inference_create_tiled(&full, &weights, capacity, 16, 32, budget);
    else s = dl_inference_create(&full, &weights, capacity, budget);
    if (s == DL_OK) s = dl_tensor_create(&output, 1, &c.vocab);
    if (s != DL_OK) goto done;
    DlMatrix row = {output.data, output.count, 0, 1, c.vocab, c.vocab};
    file = fopen(argv[6], generate ? "wbx" : "wx");
    if (file == NULL) { perror("new output"); goto done; }
    created = 1;
    Trace trace = {.file = file};
    if (!generate && !header(file, argv[2], c, ids, n, start, argv[7], argv[8])) goto done;
    if (tracing) {
        s = dl_inference_forward_trace(&full, backend, ids, n, n - 1, row, emit, &trace);
    } else if (prefixes) {
        for (size_t used = start; used <= n && s == DL_OK; ++used) {
            s = logits_at(&full, &cache, backend, ids, used, row);
            if (s == DL_OK) s = emit(&trace, "logits", used, dl_matrix_read(row));
        }
    } else {
        size_t original = n, selected = 0;
        for (size_t step = 0; step < steps && s == DL_OK; ++step) {
            s = logits_at(&full, &cache, backend, ids, n, row);
            if (s == DL_OK) s = dl_greedy(output.data, c.vocab, &selected);
            if (s != DL_OK) break;
            ids[n++] = selected;
            if (has_eos && selected == eos) break;
        }
        if (s == DL_OK && !save_ids(file, c.vocab, ids, n)) s = DL_STATE;
        if (s == DL_OK) fprintf(stderr, "selected=%zu total_ids=%zu consumed=%zu eos=%s\n",
            n - original, n, n - 1, has_eos && selected == eos ? "selected" : "not-selected");
    }
    if (s != DL_OK) goto done;
    if (!generate) fprintf(file, "{\"kind\":\"end\",\"records\":%zu}\n", trace.records);
    if (ferror(file)) { perror("output write"); goto done; }
    if (fclose(file) != 0) { perror("output close"); file = NULL; goto done; }
    file = NULL;
    result = 0;
done:
    if (s != DL_OK) fprintf(stderr, "pretrained: %s\n", dl_status_name(s));
    if (file != NULL && fclose(file) != 0) perror("close failed output");
    if (result != 0 && created && remove(argv[6]) != 0) perror("remove failed output");
    dl_tensor_destroy(&output);
    dl_cache_destroy(&cache);
    dl_inference_destroy(&full);
    dl_weights_destroy(&weights);
    return result;
usage:
    fputs("usage: compare pretrained trace CONFIG WEIGHTS IDS OUT full|tiled ref|blocked\n"
          "       compare pretrained prefix CONFIG WEIGHTS IDS OUT full|tiled|cache ref|blocked START\n"
          "       compare pretrained generate CONFIG WEIGHTS IDS OUT full|tiled|cache ref|blocked STEPS EOS|none\n",
          stderr);
    return 2;
}
```

`fopen` modes `wx` and `wbx` create a new text or binary file exclusively. If computation, serialization or final close fails, cleanup removes only a destination that this invocation actually created. Existing outputs survive because the exclusive open fails before `created` becomes true. Buffered writes may fail only at close, which is why checking individual `fprintf` calls would not be sufficient.

The ownership budget is separate from the output-file budget: weight loading allows at most a one-GiB numerical weight payload, inference scratch at most 256 MiB, and the small output row is a separate allocation. These limits are not measured process resident memory; library state, parsing buffers and allocator overhead also occupy memory.

## Dispatch and build integration

In `app/compare.c`, insert `#include "pretrained.h"` immediately after `#include "dl/kernels.h"`. At the beginning of `main`, before the existing timing-argument handling, insert:

```c title="app/compare.c: main dispatch"
    if (argc > 1 && strcmp(argv[1], "pretrained") == 0)
        return dl_compare_pretrained(argc, argv);
```

Keep the rest of the existing timing application unchanged. Its existing `<string.h>` include supplies `strcmp`. The header include and these two lines are the only changes to that file.

Replace the three existing `compare` rules in the root Makefile with these rules. The header is a dependency, while only `.c` files are compiler inputs. `$<` remains `app/compare.c`, the first prerequisite; adding the new source before that prerequisite would silently change the meaning of this automatic variable.

```makefile title="Makefile: debug compare"
build/debug/compare: app/compare.c app/pretrained.c app/pretrained.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) app/pretrained.c $< -o $@ $(LDLIBS)
```


```makefile title="Makefile: sanitize compare"
build/sanitize/compare: app/compare.c app/pretrained.c app/pretrained.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) app/pretrained.c $< -o $@ $(LDLIBS)
```


```makefile title="Makefile: release compare"
build/release/compare: app/compare.c app/pretrained.c app/pretrained.h $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) app/pretrained.c $< -o $@ $(LDLIBS)
```

On the native Mac, `make build/debug/compare` uses the existing `xcrun clang` default. On another compatible system, pass its compiler explicitly; the Linux checks for this chapter used `CC=clang`. The code requires the C17 and POSIX behavior already established in earlier chapters, not a newly invented operating-system version gate.

The command `./build/debug/compare pretrained` prints the exact three usage forms and returns status 2 because arguments are missing. A syntactically valid run returns 0 only after computation and the complete output close succeed. Runtime or input failures return 1. Part 4 supplies both small offline fixtures and actual-asset commands.

Continue to [Part 3 — Independent eager equations and numerical disagreement](https://25d-228.github.io/gpt-c/chapter-21-part-3.html). It compares the data emitted here rather than replacing the C engine.
