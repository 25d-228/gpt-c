#include "dl/rope.h"

#include <float.h>
#include <math.h>

_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG >= 53,
               "RoPE requires binary double with at least 53 significand bits");

DlStatus dl_rope_frequencies(size_t head_dim, double theta,
                             double *inv_freq, size_t count)
{
    if (inv_freq == NULL || !isfinite(theta) || theta <= 1.0) return DL_ARGUMENT;
    if (head_dim == 0 || head_dim % 2 != 0 || count != head_dim / 2) {
        return DL_SHAPE;
    }
    if (count > SIZE_MAX / sizeof(*inv_freq)) return DL_OVERFLOW;
    /* The first pass keeps the caller's output unchanged on a math failure.
     * The same immutable arguments are recomputed only after all pass. */
    for (size_t j = 0; j < count; ++j) {
        const double exponent = -(double)j / (double)count;
        const double value = pow(theta, exponent);
        if (!isfinite(value) || value <= 0.0 || value > 1.0) return DL_NUMERIC;
    }
    for (size_t j = 0; j < count; ++j) {
        inv_freq[j] = pow(theta, -(double)j / (double)count);
    }
    return DL_OK;
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

static DlStatus rope_prepare(DlConstMatrix x, size_t head_dim,
                              const size_t *positions, size_t position_count,
                              const double *inv_freq, size_t frequency_count,
                              DlMatrix y, DlWriteMode mode)
{
    if (positions == NULL || inv_freq == NULL ||
        (mode != DL_SET && mode != DL_ADD)) return DL_ARGUMENT;
    size_t xe = 0, ye = 0;
    DlStatus s = dl_matrix_end(x, &xe);
    if (s != DL_OK) return s;
    s = dl_matrix_end(dl_matrix_read(y), &ye);
    if (s != DL_OK) return s;
    if (head_dim == 0 || head_dim % 2 != 0 || x.cols % head_dim != 0 ||
        x.rows != y.rows || x.cols != y.cols || position_count != x.rows ||
        frequency_count != head_dim / 2) return DL_SHAPE;
    if (position_count > SIZE_MAX / sizeof(*positions) ||
        frequency_count > SIZE_MAX / sizeof(*inv_freq)) return DL_OVERFLOW;
    if (x.base == y.base && x.offset < ye && y.offset < xe) return DL_ALIAS;
    for (size_t i = 0; i < position_count; ++i) {
        if (positions[i] > DL_ROPE_MAX_POSITION) return DL_RANGE;
    }
    for (size_t j = 0; j < frequency_count; ++j) {
        if (!isfinite(inv_freq[j]) || inv_freq[j] <= 0.0 || inv_freq[j] > 1.0) {
            return DL_NUMERIC;
        }
    }
    if (!finite_cells(x) ||
        (mode == DL_ADD && !finite_cells(dl_matrix_read(y)))) return DL_NUMERIC;
    return DL_OK;
}

static DlStatus rope_apply(DlConstMatrix x, size_t head_dim,
                            const size_t *positions, size_t position_count,
                            const double *inv_freq, size_t frequency_count,
                            DlMatrix y, DlWriteMode mode, int transpose)
{
    DlStatus status = rope_prepare(x, head_dim, positions, position_count,
                                   inv_freq, frequency_count, y, mode);
    if (status != DL_OK) return status;
    const size_t half = head_dim / 2;
    const size_t heads = x.cols / head_dim;
    for (size_t i = 0; i < x.rows; ++i) {
        const size_t xr = x.offset + i * x.ld;
        const size_t yr = y.offset + i * y.ld;
        for (size_t j = 0; j < half; ++j) {
            const double angle = (double)positions[i] * inv_freq[j];
            const double c = cos(angle);
            double s = sin(angle);
            if (!isfinite(c) || !isfinite(s)) return DL_NUMERIC;
            if (transpose) s = -s;
            for (size_t h = 0; h < heads; ++h) {
                const size_t a = h * head_dim + j;
                const size_t b = a + half;
                const double u = (double)x.base[xr + a];
                const double v = (double)x.base[xr + b];
                double first = c * u - s * v;
                double second = s * u + c * v;
                if (mode == DL_ADD) {
                    first += (double)y.base[yr + a];
                    second += (double)y.base[yr + b];
                }
                if (!isfinite(first) || !isfinite(second) ||
                    fabs(first) > (double)FLT_MAX ||
                    fabs(second) > (double)FLT_MAX) return DL_NUMERIC;
                y.base[yr + a] = (float)first;
                y.base[yr + b] = (float)second;
            }
        }
    }
    return DL_OK;
}

DlStatus dl_rope_forward(DlConstMatrix x, size_t head_dim,
                         const size_t *positions, size_t position_count,
                         const double *inv_freq, size_t frequency_count,
                         DlMatrix y)
{
    return rope_apply(x, head_dim, positions, position_count,
                       inv_freq, frequency_count, y, DL_SET, 0);
}

DlStatus dl_rope_backward(DlConstMatrix dy, size_t head_dim,
                          const size_t *positions, size_t position_count,
                          const double *inv_freq, size_t frequency_count,
                          DlMatrix dx, DlWriteMode mode)
{
    return rope_apply(dy, head_dim, positions, position_count,
                       inv_freq, frequency_count, dx, mode, 1);
}
