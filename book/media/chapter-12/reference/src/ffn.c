#include "dl/ffn.h"

#include <float.h>
#include <math.h>

/* Private callers have at most eight inputs and seven written views. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                             const DlMatrix *out, size_t no)
{
    size_t ie[8] = {0}, oe[7] = {0};
    for (size_t i = 0; i < ni; ++i) {
        const DlStatus s = dl_matrix_end(in[i], &ie[i]);
        if (s != DL_OK) return s;
    }
    for (size_t o = 0; o < no; ++o) {
        const DlConstMatrix v = dl_matrix_read(out[o]);
        const DlStatus s = dl_matrix_end(v, &oe[o]);
        if (s != DL_OK) return s;
        for (size_t i = 0; i < ni; ++i) {
            if (v.base == in[i].base && v.offset < ie[i] &&
                in[i].offset < oe[o]) return DL_ALIAS;
        }
        for (size_t p = 0; p < o; ++p) {
            if (v.base == out[p].base && v.offset < oe[p] &&
                out[p].offset < oe[o]) return DL_ALIAS;
        }
    }
    return DL_OK;
}

static int shape(DlConstMatrix v, size_t rows, size_t cols)
{
    return v.rows == rows && v.cols == cols;
}

static int ffn_shapes(DlConstMatrix x, DlFfnWeights w, DlFfnCache c)
{
    const size_t m = x.rows, d = x.cols, f = w.gate.rows;
    return shape(w.gate, f, d) && shape(w.value, f, d) && shape(w.down, d, f)
        && shape(dl_matrix_read(c.gate), m, f)
        && shape(dl_matrix_read(c.value), m, f)
        && shape(dl_matrix_read(c.hidden), m, f);
}

static int finite_cells(DlConstMatrix v)
{
    for (size_t i = 0; i < v.rows; ++i) {
        for (size_t j = 0; j < v.cols; ++j) {
            if (!isfinite(v.base[v.offset + i * v.ld + j])) return 0;
        }
    }
    return 1;
}

static int valid_backend(DlBackend backend)
{
    return backend == DL_BACKEND_REF || backend == DL_BACKEND_BLOCKED;
}

static int valid_mode(DlWriteMode mode)
{
    return mode == DL_SET || mode == DL_ADD;
}

static DlStatus store_float(float *out, double value)
{
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

/* Both outputs use t=exp(-|a|), never an exponentially large argument.
 * Compute sigmoid's derivative without subtracting a rounded s from one. */
typedef struct {
    double value, slope;
} Silu;

static Silu silu(double a)
{
    const double t = exp(-fabs(a));
    const double den = 1.0 + t;
    const double s = a >= 0.0 ? 1.0 / den : t / den;
    const double ds = t / (den * den);
    return (Silu){a * s, s + a * ds};
}

DlStatus dl_ffn_forward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                        DlFfnCache cache, DlMatrix y)
{
    if (!valid_backend(backend)) return DL_ARGUMENT;
    const DlConstMatrix in[4] = {x, w.gate, w.value, w.down};
    const DlMatrix out[4] = {cache.gate, cache.value, cache.hidden, y};
    DlStatus s = check_views(in, 4, out, 4);
    if (s != DL_OK) return s;
    if (!ffn_shapes(x, w, cache) || !shape(dl_matrix_read(y), x.rows, x.cols))
        return DL_SHAPE;
    for (size_t i = 0; i < 4; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    s = dl_matmul(backend, x, DL_NOTRANS, w.gate, DL_TRANS, cache.gate, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, x, DL_NOTRANS, w.value, DL_TRANS, cache.value, DL_SET);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < x.rows; ++i) {
        for (size_t j = 0; j < w.gate.rows; ++j) {
            const double a = (double)cache.gate.base[cache.gate.offset + i * cache.gate.ld + j];
            const double b = (double)cache.value.base[cache.value.offset + i * cache.value.ld + j];
            const Silu z = silu(a);
            s = store_float(&cache.hidden.base[cache.hidden.offset + i * cache.hidden.ld + j],
                            z.value * b);
            if (s != DL_OK) return s;
        }
    }
    return dl_matmul(backend, dl_matrix_read(cache.hidden), DL_NOTRANS,
                      w.down, DL_TRANS, y, DL_SET);
}

DlStatus dl_ffn_backward(DlBackend backend, DlConstMatrix x, DlFfnWeights w,
                         DlFfnCache cache, DlConstMatrix dy,
                         DlFfnScratch scratch, DlMatrix dx,
                         DlWriteMode dx_mode, DlFfnGrads dw,
                         DlWriteMode weight_mode)
{
    if (!valid_backend(backend) || !valid_mode(dx_mode) || !valid_mode(weight_mode))
        return DL_ARGUMENT;
    const DlConstMatrix in[8] = {x, w.gate, w.value, w.down,
        dl_matrix_read(cache.gate), dl_matrix_read(cache.value),
        dl_matrix_read(cache.hidden), dy};
    const DlMatrix out[7] = {scratch.hidden, scratch.gate, scratch.value,
                             dx, dw.gate, dw.value, dw.down};
    DlStatus s = check_views(in, 8, out, 7);
    if (s != DL_OK) return s;
    const size_t m = x.rows, d = x.cols, f = w.gate.rows;
    if (!ffn_shapes(x, w, cache) || !shape(dy, m, d)
        || !shape(dl_matrix_read(dx), m, d)
        || !shape(dl_matrix_read(scratch.hidden), m, f)
        || !shape(dl_matrix_read(scratch.gate), m, f)
        || !shape(dl_matrix_read(scratch.value), m, f)
        || !shape(dl_matrix_read(dw.gate), f, d)
        || !shape(dl_matrix_read(dw.value), f, d)
        || !shape(dl_matrix_read(dw.down), d, f)) return DL_SHAPE;
    for (size_t i = 0; i < 8; ++i) {
        if (!finite_cells(in[i])) return DL_NUMERIC;
    }
    if (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) return DL_NUMERIC;
    if (weight_mode == DL_ADD && (!finite_cells(dl_matrix_read(dw.gate))
        || !finite_cells(dl_matrix_read(dw.value))
        || !finite_cells(dl_matrix_read(dw.down)))) return DL_NUMERIC;

    s = dl_matmul(backend, dy, DL_NOTRANS, w.down, DL_NOTRANS,
                   scratch.hidden, DL_SET);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dy, DL_TRANS, dl_matrix_read(cache.hidden), DL_NOTRANS,
                   dw.down, weight_mode);
    if (s != DL_OK) return s;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < f; ++j) {
            const double a = (double)cache.gate.base[cache.gate.offset + i * cache.gate.ld + j];
            const double b = (double)cache.value.base[cache.value.offset + i * cache.value.ld + j];
            const double g = (double)scratch.hidden.base[scratch.hidden.offset + i * scratch.hidden.ld + j];
            const Silu z = silu(a);
            s = store_float(&scratch.gate.base[scratch.gate.offset + i * scratch.gate.ld + j],
                            (g * b) * z.slope);
            if (s != DL_OK) return s;
            s = store_float(&scratch.value.base[scratch.value.offset + i * scratch.value.ld + j],
                            g * z.value);
            if (s != DL_OK) return s;
        }
    }
    s = dl_matmul(backend, dl_matrix_read(scratch.gate), DL_TRANS,
                   x, DL_NOTRANS, dw.gate, weight_mode);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dl_matrix_read(scratch.value), DL_TRANS,
                   x, DL_NOTRANS, dw.value, weight_mode);
    if (s != DL_OK) return s;
    s = dl_matmul(backend, dl_matrix_read(scratch.gate), DL_NOTRANS,
                   w.gate, DL_NOTRANS, dx, dx_mode);
    if (s != DL_OK) return s;
    return dl_matmul(backend, dl_matrix_read(scratch.value), DL_NOTRANS,
                      w.value, DL_NOTRANS, dx, DL_ADD);
}
