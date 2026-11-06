# Chapter 12 — Assemble and differentiate the complete decoder

An operator can have a correct derivative while the program that connects it to other operators has an incorrect one. A residual contribution can be overwritten. A shared parameter can be updated twice. A forward pass can leave a perfectly finite activation that belongs to yesterday's weights. This chapter makes those connections explicit rather than treating a decoder as a name for a collection of layers.

We will build one owned model, a reusable training tape, and complete forward/backward entry points. Given integer input and target arrays, the program will produce a mean cross-entropy loss and a gradient for every unique parameter. A small plain-gradient step will check that those derivatives can drive an update. Data preparation, sustained optimization, and checkpoints remain separate later capabilities.

**Prerequisites:** the complete [Chapter 11 source snapshot](https://25d-228.github.io/gpt-c/media/chapter-11/reference/Makefile), including its eight numerical modules and tests. Chapter 2 supplies ownership and cleanup; Chapter 3 supplies numerical derivatives; Chapters 5–11 supply the actual operator implementations. Keep those files unchanged except for the Makefile replacement in Part 3.

**Implementation estimate:** **4.50 hours** for the complete chapter: 3.00 for composition, initialization, and ownership, and 1.50 for gradient integration and diagnostics. Reading is separate. This chapter has three connected parts: this part builds the parameter owner and configuration path; [Part 2](https://25d-228.github.io/gpt-c/chapter-12-part-2.html) allocates the tape and schedules forward/backward; [Part 3](https://25d-228.github.io/gpt-c/chapter-12-part-3.html) supplies the complete executable tests, build integration, and experiments. All three parts are needed for the runnable result.

Create `include/dl/rng.h`, `src/rng.c`, `include/dl/model.h`, `src/model.c`, `configs/tiny.conf`, and `tests/test_model.c`. Filename-labeled blocks for the same path concatenate **in reading order, with one blank line between blocks**. `src/model.c` starts here and continues in Part 2; the other new files are complete where stated. No new operator or autograd implementation is hidden in the model module. The [complete companion snapshot](https://25d-228.github.io/gpt-c/media/chapter-12/reference/Makefile) contains the matching 34 files, including all prerequisites.

> **Learning objectives**
> Follow a complete decoder's shapes and parameter ownership; initialize an explicit reproducible random stream; distinguish a training tape from an inference cache; reverse blocks and both residual paths; merge both uses of a tied embedding table; and check the assembled graph rather than trusting a plausible loss.

## Write the entire forward graph before allocating it

Use $B$ for sequences, $T$ for tokens per sequence, $M=BT$ for flattened token rows, $V$ for vocabulary size, $D$ for hidden width, $F$ for feed-forward width, and $L$ for the number of blocks. Query and key/value head counts are $H_q$ and $H_{kv}$. Each head has $d_h=D/H_q$ coordinates. The stored key/value width is $K=H_{kv}d_h$. In this model, unlike the more general attention operator, query width is exactly $D$.

The integer input at row $r$ is $i_r$. One shared table $E\in\mathbb R^{V\times D}$ produces the first hidden state:

$$
X^0_{r,j}=E_{i_r,j}.
$$

For block $\ell$, define attention normalization scale $a^\ell$, feed-forward normalization scale $b^\ell$, and the two existing sublayers $A_\ell$ and $F_\ell$:

$$
N^\ell=\operatorname{RMSNorm}(X^\ell,a^\ell),
\qquad U^\ell=X^\ell+A_\ell(N^\ell),
$$

$$
R^\ell=\operatorname{RMSNorm}(U^\ell,b^\ell),
\qquad X^{\ell+1}=U^\ell+F_\ell(R^\ell).
$$

All six arrays above have shape $[M,D]$. The attention and feed-forward operators already include their learned matrix projections. Do not add another projection around them. Their internal saved arrays retain the shapes derived in earlier chapters.

After the final block, apply one final scale vector $c$ and the **same** table $E$:

$$
Z=\operatorname{RMSNorm}(X^L,c),\qquad O=ZE^{\mathsf T}.
$$

$O$ is the $[M,V]$ logit matrix. For target IDs $y_r$, the objective is the mean natural-log cross-entropy:

$$
\mathcal L=\frac1M\sum_{r=0}^{M-1}
\left(\log\sum_{v=0}^{V-1}e^{O_{r,v}}-O_{r,y_r}\right).
$$

This is a mathematical definition, not permission to compute overflowing exponentials. The implementation calls Chapter 5's stable loss routine. There is one averaging operation, in that routine's derivative. Blocks do not divide gradients by $M$, $T$, or the head-sharing group size.

Pre-normalized RMSNorm, SwiGLU, and rotary attention follow the architectural pattern described in [LLaMA Section 2.2][llama]. Sharing input and output embeddings is a separate choice: [Press and Wolf, Section 3][tying] discusses that reuse. The selected [SmolLM2 configuration][smollm] also declares tied word embeddings. Our small model's initialization and training dimensions below are laboratory choices, not a reproduction of that model's pretraining.

## One table, two uses, one directory entry

The lookup and output projection must point to one allocation. Two initially equal arrays are not tied: after an update they can diverge. Conversely, placing the same allocation twice in an optimizer's directory applies an update twice even when backward computed the right sum.

The parameter order will be fixed:

| Directory position | Name and shape |
|---|---|
| 0 | `token_embedding`, $[V,D]$ |
| $1+9\ell+0$ | `blocks.ℓ.attn_norm`, $[1,D]$ |
| $1+9\ell+1$ | `blocks.ℓ.query`, $[D,D]$ |
| $1+9\ell+2$ | `blocks.ℓ.key`, $[K,D]$ |
| $1+9\ell+3$ | `blocks.ℓ.value`, $[K,D]$ |
| $1+9\ell+4$ | `blocks.ℓ.output`, $[D,D]$ |
| $1+9\ell+5$ | `blocks.ℓ.ffn_norm`, $[1,D]$ |
| $1+9\ell+6$ | `blocks.ℓ.gate`, $[F,D]$ |
| $1+9\ell+7$ | `blocks.ℓ.up`, $[F,D]$ |
| $1+9\ell+8$ | `blocks.ℓ.down`, $[D,F]$ |
| $1+9L$ | `final_norm`, $[1,D]$ |

Here `ℓ` denotes the decimal block number, not a literal character in a filename. `up` is the FFN operator's `value` matrix; the directory name avoids confusing it with attention values. All weights retain `[output,input]` orientation. Scale vectors use a one-row matrix because that is the normalization interface already implemented.

There are $2+9L$ parameter tensors and

$$
P=VD+L\left(2D^2+2KD+3FD+2D\right)+D
$$

stored parameter values. The two $D^2$ terms are the query and output projections. The two $KD$ terms are key and value projections. The three $FD$ terms are the two feed-forward branches and down projection. The $2D$ term counts the two block scale vectors; the final $D$ counts the last normalization.

For $V=D=16$, $L=1$, $H_q=4$, $H_{kv}=2$, and $F=40$, head width is 4 and stored key/value width is 8. Thus

$$
P=256+(512+256+1920+32)+16=2992.
$$

There are eleven tensors, not twelve. On our FP32 targets, values and their gradients use $2\times2992\times4=23936$ bytes, excluding metadata and activations. A later optimizer will iterate this directory once; scale records carry `decay=0`, while matrix records, including the shared table, carry `decay=1`.

## Define owners and borrowed records

Start `include/dl/model.h` with the existing operator headers and the model's configuration and owner records. `rng.h` is supplied shortly below. The anonymous `enum` gives sequential integer slot constants; `DL_BLOCK_PARAMS` becomes 9. These constants name offsets, not separate parameter objects.

```c title="include/dl/model.h"
#ifndef DL_MODEL_H
#define DL_MODEL_H

#include "dl/attention.h"
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
```

Each `DlParameter` owns two distinct `DlTensor` allocations. The model owns the array of parameter records. The config is copied into the model; batch size and sequence length are deliberately absent from it because changing the amount of work does not change a weight's shape.

The next records describe a training tape. `hidden` is an allocated array of **views**, while `arena` owns the floats those views describe. Likewise, the block records contain views only. `metadata` owns the integer arrays; `ids`, `targets`, and `positions` are pointers into that one allocation and must not be freed separately. The model pointer is a borrow: the model must remain alive at the same address until the tape is destroyed.

```c title="include/dl/model.h"
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
    DlBackend backend;
    uint64_t revision;
    int ready;
} DlTrainTape;
```

The extra `g`, `other`, `du`, and `dn` matrices are backward working storage, not additional learned states. Their exact schedule appears in Part 2. `embedding_grad` will be a borrowed view of the final lookup-input derivative; it owns nothing. Attention's saved training state is not a persistent decoding key/value cache.

The API separates constructing owners, borrowing parameter views, invalidating saved state, computing gradients, and updating values. `revision` is a generation number, not a checksum. A successful parameter edit increments it. Backward accepts only a tape recorded at the current generation. This catches edits made through the documented interface; C cannot detect a caller secretly writing through an exposed pointer without calling `dl_model_mark_changed`.

```c title="include/dl/model.h"
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
void dl_train_tape_destroy(DlTrainTape *tape);

/* Accessors borrow the constructed model's arrays. Index must be in range.
 * Values may be read freely. Before any external value edit, call
 * dl_model_mark_changed and proceed ONLY on success. Never edit a weight
 * between matching forward/backward. Metadata must not be changed. */
DlMatrix dl_parameter_value(DlModel *model, size_t index);
DlMatrix dl_parameter_grad(DlModel *model, size_t index);
DlStatus dl_model_mark_changed(DlModel *model);
DlStatus dl_model_zero_grad(DlModel *model);
```

All externally held views become invalid when their owner is destroyed. Do not copy a live `DlModel` or `DlTrainTape` by assignment, pass one by value as an ownership shortcut, or relocate a model while a tape borrows its address. Their constructor's final assignment is an intentional single ownership transfer from a temporary, not a second retained owner.

Parameter accessors return borrowed matrices **by value**; copying these views is safe while the owner stays alive. Their index must be in range. The code does not pretend to validate arbitrary corrupted ownership metadata or dangling pointers. The following operations expect constructed, unmodified records and no concurrent access.

```c title="include/dl/model.h"
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

This completes the header. Forward writes saved activations but leaves parameter-gradient bytes alone. Backward's `DL_SET` means “replace the model's parameter gradients,” not “replace every contribution within a shared parameter.” The tied table still needs an internal addition. `DL_ADD` means another batch mean is added, not that two batch means are automatically averaged. The optimizer chapter will give accumulation its token-weighted interpretation.

## Random state is ordinary owned data

Initialization needs a repeatable stream without hidden library-global state. Define a small `DlRng` whose entire state is one `uint64_t`. Any 64-bit value, including zero, is a valid initial state. Different model construction calls do not share it.

```c title="include/dl/rng.h"
#ifndef DL_RNG_H
#define DL_RNG_H

#include <stdint.h>

typedef struct { uint64_t state; } DlRng;

/* rng is nonnull, initialized with ANY uint64_t seed, and exclusively owned.
 * Deterministic non-cryptographic stream. No allocation or global state. */
uint64_t dl_rng_next(DlRng *rng);
/* One draw; 53-bit grid in [0,1), represented exactly in binary64. */
double dl_rng_uniform(DlRng *rng);

#endif
```

The following recurrence uses the [SplitMix64 mixing construction published by Sebastiano Vigna][splitmix]. `UINT64_C` forms a constant suitable for a 64-bit unsigned type. `>>` shifts unsigned bits to the right, filling with zeroes; `^` is bitwise exclusive OR, not exponentiation. Each output mixes a state advanced by a fixed odd increment.

Unsigned arithmetic here deliberately retains the low 64 bits: addition and multiplication are performed modulo $2^{64}$. That behavior is defined for the unsigned type; it would not justify overflowing a signed integer. [WG14 N1570, Sections 6.2.5 and 6.5.7][c-draft] documents the relevant integer rules. In contrast, dimension arithmetic must **reject** wraparound because a wrapped allocation size would describe too few bytes.

```c title="src/rng.c"
#include "dl/rng.h"

/* SplitMix64 mixing constants and recurrence: Sebastiano Vigna (2015),
 * public domain. https://prng.di.unimi.it/splitmix64.c
 * State is explicit here rather than a file-scope variable. */
uint64_t dl_rng_next(DlRng *rng)
{
    rng->state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = rng->state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
```

For a small mental example, an eight-bit unsigned value 250 plus 10 retains $260\bmod256=4$. The generator uses the same kind of reduction with 64 bits. This is reproducible numerical state, not cryptographic randomness. We use it to initialize a tiny model and deterministic tests, not for secrets or claims about statistically independent streams.

To obtain a binary64 fraction, discard eleven bits and scale the remaining 53-bit integer by $2^{-53}$. The hexadecimal floating constant `0x1.0p-53` means exactly that power of two. Every resulting grid point is representable on the target's binary64 implementation, and the largest is $1-2^{-53}$, never one.

```c title="src/rng.c"
double dl_rng_uniform(DlRng *rng)
{
    return (double)(dl_rng_next(rng) >> 11) * 0x1.0p-53;
}
```

A seed of zero produces first words `e220a8397b1dcdaf` and `6e789e6aa1b965f4` in hexadecimal; the test checks them. The integer stream does not depend on `rand()` or the C library's choice of generator. Floating initialization also calls `sqrt`, so identical integer streams do not promise bitwise-identical floating results on every possible library and machine.

The constructor consumes one continuous stream in directory order. It does not reseed at each tensor, which would repeat prefixes across weight matrices. Normalization scales consume no draws. A later data sampler should own its own random state rather than borrowing the initialization stream and accidentally depending on how many parameters were created.

## Choose and calculate the initialization rule

Let $u$ be one generated fraction and $s$ the requested standard-deviation scale. Initialize an ordinary matrix entry as

$$
w=(2u-1)\sqrt3\,s.
$$

An ideal continuous uniform variable on $[-a,a]$ has mean zero and variance $a^2/3$, obtained by integrating $x^2/(2a)$ over the interval. Choosing $a=\sqrt3s$ gives variance $s^2$. Our finite grid approximates that distribution; this is **not** a Gaussian initialization.

All normalization scales start at one. For the two matrices that write directly into residual additions—the attention output projection and FFN down projection—use $s/\sqrt{2L}$ instead. This is an explicit laboratory heuristic: if $2L$ contributions had comparable independent variances, that factor would compensate their sum. The actual nonlinear residual branches are not independent, so this is not a proof of stable activation variance or a guarantee of successful training.

The diagnostic file sets $s=0.125$. An ordinary matrix therefore lies approximately within $\pm0.216506$; a one-block residual-output matrix lies within $\pm0.153093$. The relatively visible scale helps exercise every parameter family in a small gradient check. Loading pretrained weights later replaces these values; it does not depend on reproducing an upstream initialization distribution.

## Make the configuration file executable input

Create this exact `configs/tiny.conf`. It is read at runtime by the test executable, not a decorative file or a list of settings that must be copied into C by hand.

```text title="configs/tiny.conf"
vocab=16
layers=1
width=16
query_heads=4
kv_heads=2
ffn_width=40
batch=1
time=4
epsilon=0.00001
rope_theta=10000
init_std=0.125
seed=42
```

The grammar is intentionally small: twelve `name=value` lines in this order, no spaces, blank lines, comments, or extra settings. Lines end with LF or CRLF, including the final line. Dimension fields and seed contain decimal digits only. Floating fields use C-locale `strtod` syntax; epsilon and initialization scale must fit a positive normal float, and the RoPE base must exceed one. The program starts in the C locale and does not change it.

Strict order avoids a generic configuration framework while still rejecting missing, duplicate, or silently ignored settings. The line reader accepts at most 127 bytes before LF and rejects embedded zero bytes. It reads characters as `int`, preserving the distinction between an ordinary byte and `EOF`. The temporary null-terminated string is then safe for the bounded string operations used below.

Start `src/model.c`. `RETURN_IF_ERROR` evaluates its argument once and returns a failed status. The `do { ... } while (0)` wrapper makes the macro behave as one statement when followed by a semicolon. Its only loop executes once. Functions that allocate into temporary owners may return early because their **outer constructor** performs cleanup; do not copy this pattern into an owning function without that cleanup relationship.

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
```

The size validator calculates the exact directory and value counts with Chapter 4's checked multiplication/addition. It validates the positive head counts before division or remainder. An odd head width is rejected because split-half RoPE needs pairs. The checks are about representability, not whether the operating system has enough free memory; allocations can still fail afterward.

The two count outputs are committed only after every check succeeds. `add_product` is a small arithmetic helper: calculate $ab$ without wraparound, then add it to the running count without wraparound. It is not a tensor operation.

```c title="src/model.c"
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

The line parser copies the value together with its terminating zero. If `key_size` is the key length and `used` is the full line length, the value starts at `key_size+1`, and copying `used-key_size` bytes includes that final zero. The length test occurs first, so a too-short line cannot make a later index or subtraction invalid.

The integer converter does not call `atoi`, accept a minus sign, or rely on wrapped conversion. For next decimal digit $d$, it checks $v\le(2^{64}-1-d)/10$ before evaluating $10v+d$. This is the allocation-size principle applied to text input.

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
```

`sizes` is an array of pointers to individual output fields. Dereferencing `sizes[i]` writes the selected field in the temporary configuration, not a new allocation. The seed is treated separately because it is a `uint64_t`, not necessarily the same type as `size_t`.

For floating fields, resetting `errno` prevents an older error from being mistaken for this conversion's range error. `end` must reach the terminating zero: a prefix such as `0.1junk` is not accepted. Both finiteness and sign are checked before narrowing epsilon or the initialization scale to float. The file parser deliberately rejects subnormal values for these two settings; the in-memory model validator permits any positive finite float. Neither rule changes the RMSNorm operator's mathematical contract.

```c title="src/model.c"
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
```

The parser validates batch/time separately from parameter shapes. Time determines the local absolute positions $0,\ldots,T-1$ for each sequence, which must fit the RoPE position contract. A successfully parsed file can still request more activation memory than a particular tape budget; tape allocation performs that later check.

`dl_model_config_load` closes the file on both parse success and failure. It does not overwrite the caller's configuration, batch, or time until parsing, validation, and closing have succeeded. The existing status enum has no dedicated file-I/O member, so this small adapter uses `DL_STATE` for open/read/close failures, `DL_ARGUMENT` for syntax, and `DL_RANGE` for invalid numeric ranges. Some inherited status strings mention owners or views; the calling context identifies the configuration error. The standard I/O and conversion behavior is documented in [WG14 N1570, Sections 7.21 and 7.22.1][c-draft].

```c title="src/model.c"
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

The runtime connection is now concrete: Part 3's `main` takes an optional path, defaults to `configs/tiny.conf`, calls this loader, and passes the resulting configuration to `dl_model_create`. The loaded batch/time go to `dl_train_tape_create`. Running from another directory requires an explicit path; neither the executable nor the Makefile silently searches for configuration files.

## Borrow dense matrices without transferring ownership

The parameter tensors constructed here are always dense rank-two owners. Their complete matrix view starts at offset zero, uses the true allocation count, and has leading dimension equal to its column count. `dense_view` is not a general unchecked reshape operation: it relies on that constructor invariant. These helpers never allocate or free storage.

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
```

`weight` turns a borrowed mutable view into a borrowed read-only view for forward/backward operators. It does not freeze the underlying bytes against another pointer. Saved-state immutability remains an ownership/lifetime discipline, supported by the revision checks rather than enforced by `const` alone.

## Construct each parameter, then commit the owner

`create_parameter` first records the name, then allocates its value and gradient owners. It initializes every gradient to zero. A scale record receives ones; a matrix receives the stated uniform rule. A failed second allocation leaves the first owned by the partially built record so the outer cleanup can release it.

```c title="src/model.c"
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
```

Notice that `p->value.data` is not copied into `p->grad.data`. They have equal shapes but different meanings. Initializing the gradient bytes avoids accidentally reading uninitialized storage when the first backward uses `DL_ADD`; ordinary single-batch checks still use `DL_SET` to make their replacement intent explicit.

The next function creates the stable directory. It initializes **every record** to an empty aggregate before any tensor allocation. If construction stops at record 7, cleanup can iterate all records safely: records after 7 still own nothing. `malloc` supplies storage for the records; the explicit aggregate assignments establish their C values.

```c title="src/model.c"
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
```

Within a block, `index=1+9*layer+slot` matches the directory table exactly. Shape products used here were already validated; creation still checks each actual allocation. `snprintf` returns the required character count, so a negative result or a count reaching the buffer capacity is rejected instead of silently truncating a stable parameter name.

Scale records have `decay=0`; no name-based string search is needed to decide that policy later. Residual-output initialization is selected by the semantic slot, not by coincidental matrix shape. Both the query and output matrix are $[D,D]$, but only the latter gets the residual factor.

The destructor releases nested owners before the directory and resets the object. It accepts a partly built model. The outer constructor validates into a local candidate, delegates allocation, and either destroys that candidate or transfers it to the empty output. Failed construction therefore cannot leave an apparently usable half-model in the caller.

```c title="src/model.c"
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

`revision=1` marks the first constructed parameter generation. `gradients_valid=1` initially means the zero gradient is a valid accumulation boundary. The first forward does not change it. A later explicit weight edit or successful update invalidates it; gradient bytes may remain for inspection, but they must not be used as an update until recomputed.

The tests deliberately choose `init_std=FLT_MAX` once. This passes the basic positive-finite field check but makes the uniform bound exceed the float range **after** the first value/gradient owners were allocated. The constructor must return `DL_RANGE`, release both allocations, and leave the output empty. That exercises actual partial construction without requesting an enormous allocation or depending on an operating-system memory failure.

Continue directly to [Part 2 — Training tape and reverse scheduling](https://25d-228.github.io/gpt-c/chapter-12-part-2.html). The model now owns parameters, but a parameter owner alone does not implement a forward/backward call; the remaining `src/model.c` blocks complete that capability.

## Technical references

[LLaMA Section 2.2][llama] identifies the pre-normalization, RMSNorm, SwiGLU, and rotary-position architectural pattern. [Press and Wolf, Section 3][tying] discusses input/output weight tying; the derivative of that reuse is derived explicitly in Part 2. The [official SmolLM2-135M configuration][smollm] declares the intended later import's tied embedding setting.

[Vigna's public-domain SplitMix64 source][splitmix] supplies the integer recurrence. [WG14 N1570][c-draft] is a publicly available C11 committee draft used here for longstanding unsigned arithmetic, shifts, standard I/O, and conversion semantics also used by this C17 program; it is not presented as a newly released compiler standard or an LTS designation. The ownership layout, uniform initialization rule, directory order, and file grammar are fully specified above.

[llama]: https://arxiv.org/html/2302.13971v1#S2.SS2
[tying]: https://arxiv.org/html/1608.05859v3#S3
[smollm]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
[splitmix]: https://prng.di.unimi.it/splitmix64.c
[c-draft]: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf

