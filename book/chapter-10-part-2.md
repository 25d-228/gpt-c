# Chapter 10 — Checking causal attention and shared heads

This is Part 2 of **Causal grouped-query attention, forward**, not a separate implementation unit. [Part 1](https://25d-228.github.io/gpt-c/chapter-10.html) supplies every operator, shape, and state rule. Here we supply the complete driver and build integration, then make mistakes that produce plausible but incorrect attention.

**Prerequisite code state:** all Chapter 9 files, plus `include/dl/attention.h` and `src/attention_ref.c` from Part 1. Create `tests/test_attention.c` from its filename-labeled blocks below in order, then replace the root Makefile with the final listing. No other previous file changes. The cumulative [Chapter 10 companion state](https://25d-228.github.io/gpt-c/media/chapter-10/reference/Makefile) contains all 27 files, including the 23 unchanged prerequisite files.

The chapter's **4.00-hour** estimate includes 1.25 hours for these traces, tests, and variations; it is not an additional estimate on top of Part 1. We are checking a forward operator. Attention backward and its finite differences come in Chapter 11.

## Construct a complete three-token example

First extend the two-token calculation to three tokens and two query heads sharing **one** key/value head. Choose $B=1,T=3,D=3,H_q=2,H_{kv}=1,d=2$. Thus $Q_w=4$ and $K_w=2$. The unequal input/query/key widths keep a mistaken transpose from hiding behind a square shape.

Let $r=\sqrt2$ and $a=\log2$. Take the input $X=I_3$, the three-by-three identity matrix. Since $XW^{\mathsf T}=W^{\mathsf T}$, each input row simply selects one column of each projection weight. Use:

$$
W_q=\begin{pmatrix}1&0&-r\\0&-r&0\\0&r&0\\1&0&-r\end{pmatrix},
\qquad
W_k=\begin{pmatrix}0&0&-2a\\a&-a&a\end{pmatrix},
\qquad
W_v=\begin{pmatrix}1&0&3\\0&2&1\end{pmatrix}.
$$

The projections before rotation are therefore

$$
Q^0=\begin{pmatrix}1&0&0&1\\0&-r&r&0\\-r&0&0&-r\end{pmatrix},
\quad
K^0=\begin{pmatrix}0&a\\0&-a\\-2a&a\end{pmatrix},
\quad
V=\begin{pmatrix}1&0\\0&2\\3&1\end{pmatrix}.
$$

Set absolute positions to $[0,2,4]$ and the single frequency to $\pi/4$. This is a **synthetic diagnostic frequency**, not the model's geometric frequency schedule. It is allowed by the rotary interface and produces angles $0,\pi/2,\pi$. With two coordinates, split-half pairing is the ordinary two-coordinate rotation. After rotation:

$$
Q=\begin{pmatrix}1&0&0&1\\r&0&0&r\\r&0&0&r\end{pmatrix},
\qquad
K=\begin{pmatrix}0&a\\a&0\\2a&-a\end{pmatrix}.
$$

At time zero, both heads have one permitted key, so each returns $(1,0)$. Their actual scores differ, but that cannot change a one-key probability.

At time one, head zero scores $(0,a)$ and has probabilities $(1/3,2/3)$. Its output is $(1/3,4/3)$. Head one scores $(a,0)$ and has probabilities $(2/3,1/3)$. Its output is $(2/3,2/3)$. They use exactly the same two stored keys and values; only their queries differ.

At time two, the two score rows are $(0,a,2a)$ and $(a,0,-a)$. The corresponding probabilities and head outputs are:

| Head | Probabilities over times 0, 1, 2 | Weighted value result |
|---|---|---|
| 0 | $(1,2,4)/7$ | $(13/7,8/7)$ |
| 1 | $(4,2,1)/7$ | $(1,5/7)$ |

For example, head one's first coordinate is $4/7\cdot1+2/7\cdot0+1/7\cdot3=1$; its second is $4/7\cdot0+2/7\cdot2+1/7\cdot1=5/7$.

Concatenate the two heads in each row and choose

$$
W_o=\begin{pmatrix}1&0&0&0\\0&1&1&0\\0&0&0&2\end{pmatrix}.
$$

The first output coordinate selects head zero's first coordinate. The second combines head zero's second and head one's first. The third doubles head one's second. The complete result is

$$
Y=\begin{pmatrix}
1&1&0\\
1/3&2&4/3\\
13/7&15/7&10/7
\end{pmatrix}.
$$

This checks output projection as well as attention. Checking only the context matrix would miss a transposed or omitted $W_o$. The C fixture stores $r$ and $a$ as float and evaluates trigonometry in double, so compare against these real-valued fractions with a tolerance rather than demanding bitwise rational arithmetic.

## Give every borrowed view a real owner

Create `tests/test_attention.c`. We use named integer indices for the fourteen owners: five read-only inputs and nine forward destinations. The `enum` assigns consecutive integer constants beginning with zero; `NARRAY` is the count after the last real index. `Fixture` owns its tensors and the separately allocated position/frequency arrays. Its matrix-view members only borrow from those owners.

The following basic helpers report failures independently of `assert`, so optimized builds run exactly the same checks.

```c title="tests/test_attention.c"
#include "dl/attention.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { X, WQ, WK, WV, WO, Q0, K0, V, Q, K, SCORES, PROBS, CONTEXT, Y, NARRAY };

typedef struct {
    DlAttentionConfig c;
    DlTensor owner[NARRAY];
    DlMatrix m[NARRAY];
    size_t *positions;
    double *freq;
} Fixture;

static int require(int yes, const char *where)
{
    if (!yes) fprintf(stderr, "%s\n", where);
    return yes;
}

static int status_is(DlStatus got, DlStatus want, const char *where)
{
    if (got == want) return 1;
    fprintf(stderr, "%s: %s; expected %s\n", where,
            dl_status_name(got), dl_status_name(want));
    return 0;
}

static int near_value(double got, double want, double tol, const char *where)
{
    if (isfinite(got) && isfinite(want) && fabs(got - want) <= tol) return 1;
    fprintf(stderr, "%s: %.12g; expected %.12g, tolerance %.3g\n",
            where, got, want, tol);
    return 0;
}

static float *cell(DlMatrix m, size_t i, size_t j)
{
    return &m.base[m.offset + i * m.ld + j];
}

static void fill(DlMatrix m, float value)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) *cell(m, i, j) = value;
    }
}

static void fixture_destroy(Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) dl_tensor_destroy(&f->owner[a]);
    free(f->positions);
    free(f->freq);
    *f = (Fixture){0};
}
```

`cell` is intentionally a test-only address helper. The fixture constructor below establishes its small valid dimensions; it is not an alternative unchecked public tensor interface. `fill` touches logical cells only. `fixture_destroy` is safe after partial construction because every tensor and pointer began zero-initialized, and the inherited destroy routine accepts empty owners.

The constructor accepts only small test sizes and explicitly bounds them before using ordinary size products. It allocates **separate** owners for all fourteen matrices. With padding enabled, it adds a prefix guard and row gaps, fills all allocation bytes represented by float elements with a sentinel, and then fills only logical inputs or destinations. The actual attention routine has checked arithmetic and no such small test-size cap.

```c title="tests/test_attention.c"
/* Small TEST fixtures only; these caps are not attention's API limits.
 * f is initialized empty. Never copy this owning struct while live. */
static int fixture_create(Fixture *f, DlAttentionConfig c, size_t d, size_t pad)
{
    if (c.batch == 0 || c.batch > 2 || c.time == 0 || c.time > 8 ||
        c.query_heads == 0 || c.query_heads > 8 || c.kv_heads == 0 ||
        c.query_heads % c.kv_heads != 0 || c.head_dim == 0 ||
        c.head_dim > 8 || c.head_dim % 2 != 0 || d == 0 || d > 8 || pad > 2)
        return 0;
    f->c = c;
    const size_t m = c.batch * c.time, q = c.query_heads * c.head_dim;
    const size_t k = c.kv_heads * c.head_dim, sr = m * c.query_heads;
    const size_t rows[NARRAY] = {m, q, k, k, d, m, m, m, m, m, sr, sr, m, m};
    const size_t cols[NARRAY] = {d, d, d, d, q, q, k, k, q, k, c.time, c.time, q, d};
    for (size_t a = 0; a < NARRAY; ++a) {
        const size_t ld = cols[a] + pad, offset = pad != 0 ? 1 : 0;
        const size_t count = rows[a] * ld + (pad != 0 ? 2 : 0);
        DlStatus s = dl_tensor_create(&f->owner[a], 1, &count);
        if (s != DL_OK) goto fail;
        s = dl_matrix_view(&f->owner[a], offset, rows[a], cols[a], ld, &f->m[a]);
        if (s != DL_OK) goto fail;
        for (size_t j = 0; j < count; ++j) f->owner[a].data[j] = -777.0f;
        for (size_t i = 0; i < rows[a]; ++i) {
            for (size_t j = 0; j < cols[a]; ++j) {
                *cell(f->m[a], i, j) = a < Q0
                    ? (float)((int)((17 * i + 13 * j + 7 * a) % 23) - 11) / 32.0f
                    : NAN;
            }
        }
    }
    f->positions = malloc(m * sizeof(*f->positions));
    f->freq = malloc((c.head_dim / 2) * sizeof(*f->freq));
    if (f->positions == NULL || f->freq == NULL) goto fail;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) f->positions[b * c.time + t] = 5 + 2 * b + t;
    }
    if (dl_rope_frequencies(c.head_dim, 100000.0, f->freq, c.head_dim / 2) != DL_OK)
        goto fail;
    return 1;
fail:
    fixture_destroy(f);
    return 0;
}
```

The input pattern is deterministic and uses no global random state. Outputs begin as NaN. A successful overwrite must replace every logical destination, including masked scores/probabilities, without consulting the old NaN. Padding remains `-777.0f`.

If any tensor or metadata allocation fails, `goto fail` releases every owner acquired so far. The constructor is called only on an initialized empty fixture. A successful fixture must not be copied by value: that would copy owning pointers and invite double-free. Returning the small borrowed weight/cache structs by value, on the other hand, transfers no ownership and is safe while the fixture remains alive.

The next helpers assemble those borrowed structs, invoke the layer, and check sentinel storage. The position count is $BT$ and the frequency count is $d/2$, not the number of query heads.

```c title="tests/test_attention.c"
static DlAttentionWeights weights(const Fixture *f)
{
    return (DlAttentionWeights){dl_matrix_read(f->m[WQ]), dl_matrix_read(f->m[WK]),
        dl_matrix_read(f->m[WV]), dl_matrix_read(f->m[WO])};
}

static DlAttentionCache cache(const Fixture *f)
{
    return (DlAttentionCache){f->m[Q0], f->m[K0], f->m[V], f->m[Q], f->m[K],
                              f->m[SCORES], f->m[PROBS], f->m[CONTEXT]};
}

static DlStatus run(Fixture *f, DlBackend backend)
{
    return dl_attention_forward(backend, f->c, dl_matrix_read(f->m[X]), weights(f),
        f->positions, f->c.batch * f->c.time, f->freq, f->c.head_dim / 2,
        cache(f), f->m[Y]);
}

static int guards(const Fixture *f)
{
    for (size_t a = 0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t j = 0; j < m.count; ++j) {
            const int logical = j >= m.offset && (j - m.offset) / m.ld < m.rows
                             && (j - m.offset) % m.ld < m.cols;
            if (!logical && m.base[j] != -777.0f) return 0;
        }
    }
    return 1;
}

static void outputs_fill(Fixture *f, float value)
{
    for (size_t a = Q0; a < NARRAY; ++a) fill(f->m[a], value);
}

static int outputs_are(const Fixture *f, float value)
{
    for (size_t a = Q0; a < NARRAY; ++a) {
        const DlMatrix m = f->m[a];
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                if (*cell(m, i, j) != value) return 0;
            }
        }
    }
    return guards(f);
}
```

`guards` determines logical membership using the same declared offset and physical leading dimension as the view, but does not ask the attention routine which cells it wrote. Prefix, row-gap, and trailing elements must retain the sentinel. The error tests set every logical output to `-31.0f` and require it to remain unchanged after a preflight rejection.

Probability invariants deserve their own check. Every allowed entry must be finite and in $[0,1]$, every forbidden score/probability pair must be negative infinity and zero, and each permitted row must sum approximately to one. For time zero, the selected value is the only possible weighted result. The equality is exact here because the probability is one and all values are finite float numbers.

```c title="tests/test_attention.c"
static int row_invariants(const Fixture *f)
{
    const DlAttentionConfig c = f->c;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t h = 0; h < c.query_heads; ++h) {
            for (size_t t = 0; t < c.time; ++t) {
                const size_t sr = (b * c.query_heads + h) * c.time + t;
                double sum = 0.0;
                for (size_t s = 0; s < c.time; ++s) {
                    const float score = *cell(f->m[SCORES], sr, s);
                    const float p = *cell(f->m[PROBS], sr, s);
                    if (s > t) {
                        if (!require(isinf(score) && score < 0.0f && p == 0.0f,
                                     "masked cell is not (-infinity, zero)")) return 0;
                    } else {
                        if (!require(isfinite(score) && isfinite(p) && p >= 0.0f && p <= 1.0f,
                                     "invalid allowed score/probability")) return 0;
                        sum += (double)p;
                    }
                }
                if (!near_value(sum, 1.0, 2e-7, "probability row sum")) return 0;
                if (t == 0) {
                    const size_t kh = h / (c.query_heads / c.kv_heads);
                    for (size_t j = 0; j < c.head_dim; ++j) {
                        if (!near_value(*cell(f->m[CONTEXT], b * c.time, h * c.head_dim + j),
                                        *cell(f->m[V], b * c.time, kh * c.head_dim + j),
                                        0.0, "one-key row equals its value")) return 0;
                    }
                }
            }
        }
    }
    return 1;
}
```

These invariants are necessary, not sufficient. Selecting the wrong but in-range shared head can still give a normalized row and a valid weighted average. We therefore also need hand calculations and a separately written computation of the intended output.

## Run the hand-calculated sublayer with both projection backends

Append the complete fixture driver. The source arrays store weights in output-by-input order; copying them into padded views does not transpose their meaning. The test checks rotated queries and keys before final outputs and probabilities, which helps locate a disagreement before blaming softmax.

```c title="tests/test_attention.c"
static int hand_trace(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1, 3, 2, 1, 2}, 3, 2)) return 0;
    const float r = (float)sqrt(2.0), a = (float)log(2.0);
    const float x[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float wq[12] = {1, 0, -r, 0, -r, 0, 0, r, 0, 1, 0, -r};
    const float wk[6] = {0, 0, -2 * a, a, -a, a};
    const float wv[6] = {1, 0, 3, 0, 2, 1};
    const float wo[12] = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2};
    const float *src[5] = {x, wq, wk, wv, wo};
    for (size_t v = 0; v < 5; ++v) {
        for (size_t i = 0; i < f.m[v].rows; ++i) {
            for (size_t j = 0; j < f.m[v].cols; ++j)
                *cell(f.m[v], i, j) = src[v][i * f.m[v].cols + j];
        }
    }
    f.positions[0] = 0; f.positions[1] = 2; f.positions[2] = 4;
    f.freq[0] = acos(-1.0) / 4.0; /* Synthetic quarter-turn fixture. */
    int ok = status_is(run(&f, backend), DL_OK, "hand forward");
    const double want_q[12] = {1, 0, 0, 1, r, 0, 0, r, r, 0, 0, r};
    const double want_k[6] = {0, a, a, 0, 2 * a, -a};
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 4; ++j)
            ok = near_value(*cell(f.m[Q], i, j), want_q[i * 4 + j], 3e-7, "hand rotated Q");
        for (size_t j = 0; ok && j < 2; ++j)
            ok = near_value(*cell(f.m[K], i, j), want_k[i * 2 + j], 3e-7, "hand rotated K");
    }
    const double want_y[9] = {1, 1, 0, 1.0 / 3.0, 2, 4.0 / 3.0,
                             13.0 / 7.0, 15.0 / 7.0, 10.0 / 7.0};
    const double want_p[6] = {1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0,
                             4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0};
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 3; ++j)
            ok = near_value(*cell(f.m[Y], i, j), want_y[i * 3 + j], 1e-6, "hand Y");
    }
    for (size_t h = 0; ok && h < 2; ++h) {
        for (size_t s = 0; ok && s < 3; ++s)
            ok = near_value(*cell(f.m[PROBS], h * 3 + 2, s), want_p[h * 3 + s],
                            2e-7, "hand head probability");
    }
    ok = ok && row_invariants(&f) && require(guards(&f), "hand padding changed");
    if (ok) printf("hand Y[2] (%s): [%.6f, %.6f, %.6f]\n",
        backend == DL_BACKEND_REF ? "reference" : "blocked",
        (double)*cell(f.m[Y], 2, 0), (double)*cell(f.m[Y], 2, 1),
        (double)*cell(f.m[Y], 2, 2));
    fixture_destroy(&f);
    return ok;
}
```

The expected last row prints as `[1.857143, 2.142857, 1.428571]`. The tolerance on probabilities is $2\times10^{-7}$, and the tolerance on the final fractions is $10^{-6}$. They cover the float projections, rotations, saved scores/probabilities, and final store in this small fixture; they are not a universal tolerance for a deep model.

Both backends traverse the same attention code but use different matrix-product implementations. Agreement between them alone cannot establish that the common head mapping or mask is correct. That is why the expected fractions do not come from a first production run.

## Build a small independent computation from the inputs

The next checker deliberately redoes projections, rotations, scoring, normalization, and output mixing with simple double-precision arithmetic. It does **not** read the production cache and does not call production matrix multiplication or RoPE. Recomputing projections inside inner loops would be wasteful in the model, but makes a tiny reference easy to inspect.

For a selected rotary coordinate, `plane = j % half` identifies its pair, and `j < half` chooses the first or second rotated component. This is an independent expression of split-half rotation, not a call to the function being checked. The checker keeps intermediates in double rather than reproducing every float storage boundary, so its final values should be close rather than necessarily identical.

```c title="tests/test_attention.c"
/* Independent, slow double arithmetic from INPUTS, not the saved cache.
 * This checker deliberately does not call production matmul/RoPE/attention.
 * It omits intermediate float stores; compare with a tolerance, not bitwise. */
static double projection(const Fixture *f, size_t row, size_t weight, size_t col)
{
    double sum = 0.0;
    for (size_t j = 0; j < f->m[X].cols; ++j)
        sum += (double)*cell(f->m[X], row, j) * (double)*cell(f->m[weight], col, j);
    return sum;
}

static double rotated(const Fixture *f, size_t row, size_t weight,
                       size_t head, size_t j)
{
    const size_t half = f->c.head_dim / 2;
    const size_t plane = j % half, base = head * f->c.head_dim;
    const double angle = (double)f->positions[row] * f->freq[plane];
    const double u = projection(f, row, weight, base + plane);
    const double v = projection(f, row, weight, base + plane + half);
    return j < half ? cos(angle) * u - sin(angle) * v
                    : sin(angle) * u + cos(angle) * v;
}

static int oracle(const Fixture *f, double *expected)
{
    const DlAttentionConfig c = f->c;
    if (c.time > 8 || c.query_heads * c.head_dim > 32) return 0;
    for (size_t b = 0; b < c.batch; ++b) {
        for (size_t t = 0; t < c.time; ++t) {
            double context[32] = {0};
            const size_t row = b * c.time + t;
            for (size_t h = 0; h < c.query_heads; ++h) {
                double score[8] = {0}, maximum = -INFINITY, den = 0.0;
                const size_t kh = h / (c.query_heads / c.kv_heads);
                for (size_t s = 0; s <= t; ++s) {
                    for (size_t j = 0; j < c.head_dim; ++j)
                        score[s] += rotated(f, row, WQ, h, j)
                                  * rotated(f, b * c.time + s, WK, kh, j);
                    score[s] /= sqrt((double)c.head_dim);
                    maximum = fmax(maximum, score[s]);
                }
                for (size_t s = 0; s <= t; ++s) den += exp(score[s] - maximum);
                for (size_t j = 0; j < c.head_dim; ++j) {
                    for (size_t s = 0; s <= t; ++s)
                        context[h * c.head_dim + j] += exp(score[s] - maximum) / den
                            * projection(f, b * c.time + s, WV, kh * c.head_dim + j);
                }
            }
            for (size_t j = 0; j < f->m[Y].cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < c.query_heads * c.head_dim; ++k)
                    sum += context[k] * (double)*cell(f->m[WO], j, k);
                expected[row * f->m[Y].cols + j] = sum;
            }
        }
    }
    return 1;
}
```

`score[8]` and `context[32]` have explicit test-only bounds checked before use. The caller supplies `expected` large enough for the known fixture output; the following driver uses at most six rows of five coordinates. These arrays do not constrain the production operator and are not a hidden variable-length allocation.

The cases include one key/value head, an intermediate grouping, one key/value head per query, three queries sharing each key/value head, and a length-one sequence. They use two batches, nonzero positions, unequal model/query/key widths, and both padded and dense views. Both projection backends run every case. We also retain copies of all input allocations to detect unintended input writes, including changes to their padding.

```c title="tests/test_attention.c"
static int oracle_cases(void)
{
    const DlAttentionConfig cases[5] = {
        {2, 3, 4, 1, 2}, {2, 3, 4, 2, 2}, {2, 3, 4, 4, 2},
        {2, 3, 6, 2, 4}, {2, 1, 4, 2, 2}
    };
    size_t count = 0;
    double worst = 0.0;
    for (size_t ci = 0; ci < 5; ++ci) {
        for (int be = 0; be < 2; ++be) {
            Fixture f = {0};
            if (!fixture_create(&f, cases[ci], 5, ci == 2 ? 0 : 2)) return 0;
            double expected[30] = {0};
            float saved[5][512];
            int ok = 1;
            for (size_t a = 0; a < 5; ++a) {
                if (f.owner[a].count > 512) { ok = 0; break; }
                memcpy(saved[a], f.owner[a].data, f.owner[a].count * sizeof(float));
            }
            ok = ok && oracle(&f, expected) &&
                status_is(run(&f, (DlBackend)be), DL_OK, "oracle forward");
            for (size_t i = 0; ok && i < f.m[Y].rows; ++i) {
                for (size_t j = 0; ok && j < f.m[Y].cols; ++j) {
                    const double want = expected[i * f.m[Y].cols + j];
                    const double got = (double)*cell(f.m[Y], i, j);
                    worst = fmax(worst, fabs(got - want));
                    ok = near_value(got, want, 2e-6 + 1e-5 * fabs(want), "double oracle Y");
                }
            }
            for (size_t a = 0; ok && a < 5; ++a)
                ok = require(memcmp(saved[a], f.owner[a].data,
                             f.owner[a].count * sizeof(float)) == 0, "input was changed");
            ok = ok && row_invariants(&f) && require(guards(&f), "oracle guards changed");
            fixture_destroy(&f);
            if (!ok) return 0;
            ++count;
        }
    }
    printf("independent oracle: %zu cases, max absolute error %.6g\n", count, worst);
    return 1;
}
```

The diagnostic prints a maximum absolute discrepancy; it is not a speed measurement. The test accepts an absolute component of $2\times10^{-6}$ plus a relative component of $10^{-5}|y_{\mathrm{expected}}|$. The absolute part matters near zero. A malformed NaN cannot pass because `near_value` requires both numbers to be finite.

This checker is independent in implementation, not an independent architecture specification. It shares the intended mask and head convention with the chapter. The hand fixture, axis trace, primary references, and later pretrained comparison provide different kinds of evidence. Ten passing cases do not establish equivalence for every possible shape or extreme input.

## Test causality and batch isolation as dependencies

Set all rotary positions to seven on purpose. If mask membership is incorrectly inferred from numeric position labels, every future key would appear to be at the same permitted position. The correct implementation still follows time order.

Run a baseline, then change the final token of batch zero. Its own output must change measurably; otherwise an unchanged earlier output could be a vacuous test. Earlier rows of that sequence and every row of the other sequence must stay unchanged. Finally change all inputs in batch one and confirm that no row of batch zero changes.

```c title="tests/test_attention.c"
static int isolation(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2, 3, 4, 2, 2}, 5, 2)) return 0;
    /* Equal positions are legal and do NOT make future tokens visible. */
    for (size_t i = 0; i < 6; ++i) f.positions[i] = 7;
    float old_y[30] = {0};
    int ok = status_is(run(&f, backend), DL_OK, "isolation baseline");
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 5; ++j) old_y[i * 5 + j] = *cell(f.m[Y], i, j);
    }
    for (size_t j = 0; j < 5; ++j) *cell(f.m[X], 2, j) += (float)(j + 1);
    ok = ok && status_is(run(&f, backend), DL_OK, "future perturbation");
    double changed = 0.0;
    for (size_t i = 0; ok && i < 6; ++i) {
        for (size_t j = 0; ok && j < 5; ++j) {
            if (i == 2) changed += fabs((double)*cell(f.m[Y], i, j) - old_y[i * 5 + j]);
            else ok = near_value(*cell(f.m[Y], i, j), old_y[i * 5 + j], 0.0,
                                 "future/batch isolation");
        }
    }
    ok = ok && require(changed > 1e-5, "future perturbation had no measurable effect");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 5; ++j) old_y[i * 5 + j] = *cell(f.m[Y], i, j);
    }
    for (size_t i = 3; i < 6; ++i) {
        for (size_t j = 0; j < 5; ++j) *cell(f.m[X], i, j) -= (float)(i + j + 1);
    }
    ok = ok && status_is(run(&f, backend), DL_OK, "other batch perturbation");
    for (size_t i = 0; ok && i < 3; ++i) {
        for (size_t j = 0; ok && j < 5; ++j)
            ok = near_value(*cell(f.m[Y], i, j), old_y[i * 5 + j], 0.0, "batch isolation");
    }
    ok = ok && row_invariants(&f) && require(guards(&f), "isolation padding");
    fixture_destroy(&f);
    return ok;
}
```

The unchanged comparisons use zero tolerance on this controlled test. They rerun the **same backend** with exactly the same relevant input cells and operation order; they are not comparing different platforms or independently rounded algorithms. The causal claim is about successful calls with finite inputs. The rejection tests below separately demonstrate that the global preflight still checks a non-finite future row.

The tests also keep batch boundaries distinct from the score matrix's row layout. Omitting the batch offset from key or value row arithmetic reads valid memory from batch zero. Sanitizers have no reason to report that logical mistake.

## Compare equal effective heads, not unrelated models

To check the grouped and multi-head endpoints, create two separate fixtures. The first stores fewer key/value heads. The second is ordinary multi-head attention, but its key/value **weight rows are deliberately copied** from the appropriate shared head. Query weights, output weights, inputs, positions, and frequencies start identically. Thus both models have the same effective vectors for every query head.

Repeating weights in this diagnostic fixture is not how production GQA is implemented. It constructs a separately allocated baseline whose mathematical equivalence we can state precisely. Merely generating unrelated random weights for the two head configurations would not predict equal results.

```c title="tests/test_attention.c"
/* Expand weights ONLY in a separate MHA fixture to prove equal effective
 * tensors, not in the production GQA cache or projection. */
static int matched_heads(DlBackend backend, size_t kv_heads)
{
    Fixture a = {0}, b = {0};
    if (!fixture_create(&a, (DlAttentionConfig){1, 3, 4, kv_heads, 2}, 5, 2) ||
        !fixture_create(&b, (DlAttentionConfig){1, 3, 4, 4, 2}, 5, 2)) {
        fixture_destroy(&a); fixture_destroy(&b); return 0;
    }
    for (size_t w = WK; w <= WV; ++w) {
        for (size_t h = 0; h < 4; ++h) {
            for (size_t j = 0; j < 2; ++j) {
                for (size_t d = 0; d < 5; ++d)
                    *cell(b.m[w], h * 2 + j, d) =
                        *cell(a.m[w], (h / (4 / kv_heads)) * 2 + j, d);
            }
        }
    }
    int ok = status_is(run(&a, backend), DL_OK, "grouped matched forward") &&
             status_is(run(&b, backend), DL_OK, "expanded matched forward");
    const size_t compare[3] = {PROBS, CONTEXT, Y};
    for (size_t n = 0; ok && n < 3; ++n) {
        const size_t v = compare[n];
        for (size_t i = 0; ok && i < a.m[v].rows; ++i) {
            for (size_t j = 0; ok && j < a.m[v].cols; ++j)
                ok = near_value(*cell(a.m[v], i, j), *cell(b.m[v], i, j),
                                1e-7, "matched MHA/GQA result");
        }
    }
    fixture_destroy(&a); fixture_destroy(&b);
    return ok;
}
```

The comparisons include every probability, every concatenated head result, and the final output. Run with `kv_heads` equal to one and two, and with both projection backends. This makes four matched-model comparisons in addition to the ten double-oracle cases.

No averaging over the sharing queries occurs in forward. In backward, those queries will contribute to the same key/value gradient destination. These tests fix the intended forward equivalence before that accumulation is implemented.

## Reject malformed calls without altering any destination

The next function uses two short local preprocessor macros to keep repeated full calls readable. A trailing backslash continues a macro definition onto the next physical source line. `REJECT` wraps its statements in `do { ... } while (0)` so it behaves as one statement at its call site. Its arguments are parenthesized and the expression under test is evaluated once. `goto done` exits to the common cleanup path on a failure. Both names are removed with `#undef` immediately afterward; they are not public interfaces.

The tests change copied configuration/view metadata, not live owner metadata. They reject empty sequences, invalid head counts, unsupported odd rotary widths, overflowed counts, wrong weight shapes, invalid strides, forbidden aliases, missing/wrong metadata, and non-finite inputs. There is no fabricated allocation of `SIZE_MAX` elements: size rejection happens before any such pointer access.

```c title="tests/test_attention.c"
static int rejected_calls(void)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){2, 3, 4, 2, 2}, 5, 2)) return 0;
    outputs_fill(&f, -31.0f);
    int ok = 1;
#define REJECT(expr, code, label) do { \
    if (!status_is((expr), (code), (label)) || \
        !require(outputs_are(&f, -31.0f), "preflight changed destinations")) { \
        ok = 0; goto done; \
    } \
} while (0)
#define CALL(cfg, wt, pos, np, fr, nf, ca, out) \
    dl_attention_forward(DL_BACKEND_REF, (cfg), dl_matrix_read(f.m[X]), (wt), \
        (pos), (np), (fr), (nf), (ca), (out))
    REJECT(run(&f, (DlBackend)77), DL_ARGUMENT, "unknown backend");
    DlAttentionConfig bad = f.c;
    bad.time = 0;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "empty sequence");
    bad = f.c; bad.kv_heads = 0;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "zero KV heads");
    bad = f.c; bad.query_heads = 3;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "nondivisible heads");
    bad = f.c; bad.head_dim = 3;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "odd rotary width");
    bad = f.c; bad.batch = SIZE_MAX;
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_OVERFLOW, "row product overflow");
    bad = (DlAttentionConfig){1, SIZE_MAX / 16, 4, 2, 2};
    REJECT(CALL(bad, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_OVERFLOW, "storage product overflow");
    DlAttentionWeights w = weights(&f);
    --w.key.cols;
    REJECT(CALL(f.c, w, f.positions, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "wrong key projection width");
    DlMatrix y = f.m[Y]; y.ld = y.cols - 1;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, cache(&f), y),
           DL_ARGUMENT, "invalid leading dimension");
    DlAttentionCache a = cache(&f); a.probs = a.scores;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, a, f.m[Y]),
           DL_ALIAS, "probabilities overlap scores");
    a = cache(&f); a.q = a.q0;
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, a, f.m[Y]),
           DL_ALIAS, "in-place rotation");
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 1, cache(&f), f.m[X]),
           DL_ALIAS, "output overlaps input");
    REJECT(CALL(f.c, weights(&f), NULL, 6, f.freq, 1, cache(&f), f.m[Y]),
           DL_ARGUMENT, "null positions");
    REJECT(CALL(f.c, weights(&f), f.positions, 5, f.freq, 1, cache(&f), f.m[Y]),
           DL_SHAPE, "position count");
    REJECT(CALL(f.c, weights(&f), f.positions, 6, f.freq, 0, cache(&f), f.m[Y]),
           DL_SHAPE, "frequency count");
    f.freq[0] = NAN;
    REJECT(run(&f, DL_BACKEND_REF), DL_ARGUMENT, "nonfinite frequency");
    f.freq[0] = 1.0;
    if (SIZE_MAX > UINT32_MAX) {
        f.positions[0] = DL_ROPE_MAX_POSITION + (size_t)1;
        REJECT(run(&f, DL_BACKEND_REF), DL_RANGE, "position cap");
        f.positions[0] = 5;
    }
    const float saved = *cell(f.m[WO], 4, 7);
    *cell(f.m[WO], 4, 7) = NAN;
    REJECT(run(&f, DL_BACKEND_REF), DL_NUMERIC, "late weight preflight");
    *cell(f.m[WO], 4, 7) = saved;
    *cell(f.m[X], 5, 4) = INFINITY;
    REJECT(run(&f, DL_BACKEND_REF), DL_NUMERIC, "future input still must be finite");
done:
#undef CALL
#undef REJECT
    fixture_destroy(&f);
    return ok;
}
```

A non-finite entry in the **last output-projection weight** is caught before the first query projection writes. This is the reason for whole-call finite-input preflight rather than relying only on nested operations' checks.

The final input test places infinity in batch one's last row. Even outputs from earlier valid rows remain at the sentinel: the call is rejected, not partially evaluated. This error behavior does not contradict causal dataflow.

The tests exercise aliases between score/probability storage, pre-/post-rotation query storage, and input/final output. Distinct allocations in the normal fixture are not an invitation to skip alias validation in the public function: the caller could pass different views of the same allocation instead.

## Separate stable normalization from arithmetic failure

The first extreme-score case uses two finite scores near opposite ends of the float range. Their difference cannot fit in float, but does fit in double. The smaller shifted exponential underflows to zero; the larger contributes one. No infinity is created by normalization.

The next case produces scores near 1000 and 1001 and checks the logistic probability of the latter. Then two different failures demonstrate that stable softmax does not repair unrelated overflow: a later allowed score cannot be stored as float, and a final output projection overflows after the complete attention context has been written.

```c title="tests/test_attention.c"
static int extremes_and_partial_failure(DlBackend backend)
{
    Fixture f = {0};
    if (!fixture_create(&f, (DlAttentionConfig){1, 2, 1, 1, 2}, 2, 2)) return 0;
    for (size_t a = 0; a < 5; ++a) fill(f.m[a], 0.0f);
    *cell(f.m[X], 0, 0) = *cell(f.m[X], 1, 1) = 1.0f;
    *cell(f.m[WO], 0, 0) = *cell(f.m[WO], 1, 1) = 1.0f;
    *cell(f.m[WQ], 0, 0) = *cell(f.m[WQ], 0, 1) = 1.0f;
    *cell(f.m[WK], 0, 0) = FLT_MAX; *cell(f.m[WK], 0, 1) = -FLT_MAX;
    *cell(f.m[WV], 0, 0) = 1.0f; *cell(f.m[WV], 0, 1) = 3.0f;
    *cell(f.m[WV], 1, 0) = 2.0f; *cell(f.m[WV], 1, 1) = 4.0f;
    f.positions[0] = f.positions[1] = 0;
    int ok = status_is(run(&f, backend), DL_OK, "extreme finite scores") &&
        near_value(*cell(f.m[PROBS], 1, 0), 1.0, 0.0, "extreme winner") &&
        near_value(*cell(f.m[PROBS], 1, 1), 0.0, 0.0, "extreme underflow") &&
        near_value(*cell(f.m[Y], 1, 0), 1.0, 0.0, "extreme output") &&
        row_invariants(&f);
    *cell(f.m[WQ], 0, 0) = *cell(f.m[WQ], 0, 1) = (float)sqrt(2.0);
    *cell(f.m[WK], 0, 0) = 1000.0f; *cell(f.m[WK], 0, 1) = 1001.0f;
    const double p = 1.0 / (1.0 + exp(-1.0));
    ok = ok && status_is(run(&f, backend), DL_OK, "large common score") &&
        near_value(*cell(f.m[PROBS], 1, 1), p, 1e-7, "max-shifted probability");

    outputs_fill(&f, NAN);
    *cell(f.m[WQ], 0, 0) = 1.0f; *cell(f.m[WQ], 0, 1) = FLT_MAX;
    fill(f.m[WK], 0.0f);
    *cell(f.m[WK], 1, 0) = 1.0f; *cell(f.m[WK], 0, 1) = 2.0f;
    ok = ok && status_is(run(&f, backend), DL_NUMERIC, "later score overflow") &&
        near_value(*cell(f.m[CONTEXT], 0, 0), 1.0, 0.0, "earlier row was written") &&
        require(isnan(*cell(f.m[CONTEXT], 1, 0)) && isnan(*cell(f.m[Y], 0, 0)),
                "failed score unexpectedly completed output");

    outputs_fill(&f, NAN);
    *cell(f.m[WQ], 0, 1) = 1.0f;
    *cell(f.m[WO], 1, 0) = *cell(f.m[WO], 1, 1) = FLT_MAX;
    ok = ok && status_is(run(&f, backend), DL_NUMERIC, "output projection overflow") &&
        require(isfinite(*cell(f.m[CONTEXT], 1, 0)), "cache did not finish before output failure") &&
        require(isnan(*cell(f.m[Y], 0, 1)), "overflowed Y cell was stored") &&
        require(guards(&f), "numeric failure changed padding");
    fixture_destroy(&f);
    return ok;
}
```

In the score-failure case, query row zero has already written a valid-looking context. Query row one and final output remain incomplete. In the output-projection failure, all context rows exist, but $Y$ is unusable. The test inspects these states only to establish the failure contract, then destroys them; it never treats them as successful results.

The exact subset of a failed matrix output may differ between scalar and blocked traversal, as Chapter 6 explains. The checks therefore do not demand identical partial $Y$ arrays from the two backends. They demand an explicit numerical failure, no out-of-range store into the failing cell, and unchanged padding.

Finish the test file with its complete entry point. Failure returns a nonzero exit status; success is not inferred from the absence of a crash.

```c title="tests/test_attention.c"
int main(void)
{
    if (!hand_trace(DL_BACKEND_REF) || !hand_trace(DL_BACKEND_BLOCKED) ||
        !oracle_cases() || !rejected_calls()) return EXIT_FAILURE;
    for (int be = 0; be < 2; ++be) {
        const DlBackend backend = (DlBackend)be;
        if (!isolation(backend) || !matched_heads(backend, 1) ||
            !matched_heads(backend, 2) || !extremes_and_partial_failure(backend))
            return EXIT_FAILURE;
    }
    puts("attention forward: hand, oracle, causality, sharing, boundaries, failures passed");
    return EXIT_SUCCESS;
}
```

## Build the new operator without losing inherited checks

Replace the root Makefile with this complete version. It adds one source module, one header dependency, and one executable to the existing targets. It retains all earlier tests and the small matrix-comparison smoke run.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h tests/tiny.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention

.PHONY: all debug sanitize release test clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare

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

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

clean:
	rm -rf build
```

From the learner workspace root on the target Mac:

```sh title="Terminal — cumulative builds and checks"
make test
```

The default compiler remains `xcrun clang`. An equivalent compiler can be selected explicitly; for example, the Linux authoring checks use `make CC=clang test`. Keep the supplied `-fno-fast-math` behavior: the implementation distinguishes non-finite input, masked negative infinity, and finite arithmetic results. Compiler modes that assume infinities and NaNs never occur are unsuitable for these contracts; see the [Clang floating-point documentation][clang-fp].

For only the new executable, without skipping any build mode:

```sh title="Terminal — focused attention checks"
make build/debug/test_attention build/sanitize/test_attention build/release/test_attention
./build/debug/test_attention
./build/sanitize/test_attention
./build/release/test_attention
```

A representative successful run prints:

```text
hand Y[2] (reference): [1.857143, 2.142857, 1.428571]
hand Y[2] (blocked): [1.857143, 2.142857, 1.428571]
independent oracle: 10 cases, max absolute error 7.92276e-09
attention forward: hand, oracle, causality, sharing, boundaries, failures passed
```

The oracle error is an observation from Clang 17 on x86_64 Linux, not a portable expected last digit. The acceptance tolerances live in the source. The debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized runs all passed there, including the inherited test suite. A separate leak-enabled run of the new sanitizer executable reported no leaks. These are not claims of execution on the M4 Mac or of measured attention performance.

The suite runs both hand traces, ten independent-double cases, four matched-head comparisons, causal/batch perturbations with both backends, rejection checks, and both late numerical-failure paths. There are no new attention finite-difference counts in this chapter: a backward implementation is needed before that comparison would mean anything.

## Diagnose deliberate mistakes by their first wrong intermediate

Work on a temporary copy of `src/attention_ref.c`, change one expression at a time, and restore it afterward. Rebuild `test_attention` in all three modes. Do not keep an incorrect implementation merely because it passes a memory sanitizer.

**Change the sharing map to `h % c.kv_heads`.** With one key/value head this is indistinguishable from the correct map, so the hand fixture still passes. The two-key/value-head oracle case fails: a checked output becomes approximately 0.0297546 where the independent result is -0.197510 in the observed run. Inspect the head-to-key table before adjusting a tolerance. All accesses remain in bounds.

**Scale by the full query width.** Replace `sqrt((double)c.head_dim)` with `sqrt((double)(c.query_heads * c.head_dim))`. In the hand fixture, the second row's first output becomes approximately 0.379863 instead of $1/3$. The probabilities remain normalized but are less selective. The number of coordinates in one dot product, not the total number of projected coordinates, determines this scale.

**Remove causality consistently.** To create a fully noncausal diagnostic, make the `s > t` branch unreachable and extend all three allowed-key loops from `s <= t` to `s < c.time`. The first hand output changes from one to approximately 1.697998 in the observed run. This is not rounding noise. The future-token perturbation also has a direct mathematical path to earlier rows. Changing only the mask branch while leaving the later loops causal would instead create inconsistent saved state, a different bug.

**Read keys and values from batch zero.** Replace `b * c.time + s` by `s` in the `kr` and `vr` definitions only. The one-batch hand calculation still passes. A two-batch oracle comparison fails; the observed first discrepancy is -0.003662109375 where zero is expected. A valid address can still belong to the wrong sequence.

All four isolated variations returned numerical test failures without an AddressSanitizer or UndefinedBehaviorSanitizer report in the authoring environment. That is the expected distinction: those tools diagnose memory/undefined behavior, not the intended attention function.

For a valid variation, change the six-query-head oracle case from two to **three** key/value heads. The shared-head group size becomes two rather than three; the fixture constructor allocates the new projection widths and the independent checker follows the declared grouping. This variation passes. The outputs are not expected to equal the old case's outputs because the deterministic weight inventory now differs.

## What to inspect when your own result disagrees

Start at the first divergent stage rather than the final scalar. A wrong `q0` or `k0` points toward weight orientation, input row selection, or matrix stride. Correct projections but wrong `q`/`k` point toward head-local coordinate pairing or the position array. Correct rotated vectors but wrong allowed scores point toward head sharing, batch offsets, or the $1/\sqrt d$ factor.

If scores agree but probabilities do not, check that the maximum and denominator cover **only the allowed keys in that one row**, and that score subtraction widens before subtracting. A correct allowed probability row with nonzero future entries means masked destinations were not explicitly written. A sum below one after masking suggests normalization happened over too many keys.

Correct probabilities with wrong context point toward value-head selection or `V` stride. Correct context with wrong $Y$ points toward the output matrix's orientation or column order. Sanitizers remaining silent in any of these cases is not surprising; all can be in-bounds numerical mistakes.

The completed sublayer now has explicit shapes, projected inputs, rotary metadata, scores, probabilities, and head outputs. Preserve this interface and its successful tape. Chapter 11 will reverse the weighted sum, softmax, scaled dot products, rotations, and four projections, with **addition** of every shared-head gradient contribution rather than assignment by the last query head.

## Technical references

The definitions and source references for scaled attention, grouped heads, and rotary coordinates are in [Part 1's technical references](https://25d-228.github.io/gpt-c/chapter-10.html#technical-references). The test fixtures and scalar checker above are fully specified here; no external framework is needed to generate their expected values.

Clang, [floating-point behavior controls][clang-fp] and [AddressSanitizer usage][asan], document the relevant compiler and diagnostic features. Neither tool substitutes for the hand, shape, dependency, and independent-numerical checks.

[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage

