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

static DlStatus logits_at(DlInference *full, DlKvCache *cache, DlBackend backend,
                           const size_t *ids, size_t n, DlMatrix logits)
{
    if (cache->weights == NULL)
        return dl_inference_forward(full, backend, ids, n, n - 1, logits);
    if (cache->length == 0) return dl_cache_prefill(cache, backend, ids, n, logits);
    if (cache->length + 1 != n) return DL_STATE;
    return dl_cache_decode(cache, backend, ids[n - 1], logits);
}

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
