#include "dl/norm.h"

#include <float.h>
#include <math.h>

/* Internal callers supply at most three inputs and two outputs. */
static DlStatus check_views(const DlConstMatrix *in, size_t ni,
                             const DlMatrix *out, size_t no)
{
    size_t ie[3] = {0}, oe[2] = {0};
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

static int same_shape(DlConstMatrix a, DlConstMatrix b)
{
    return a.rows == b.rows && a.cols == b.cols;
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

static int valid_mode(DlWriteMode mode)
{
    return mode == DL_SET || mode == DL_ADD;
}

static DlStatus put_value(float *out, double value, DlWriteMode mode)
{
    if (mode == DL_ADD) value += (double)*out;
    if (!isfinite(value) || fabs(value) > (double)FLT_MAX) return DL_NUMERIC;
    *out = (float)value;
    return DL_OK;
}

static double inverse_rms(DlConstMatrix x, size_t i, float epsilon)
{
    const size_t row = x.offset + i * x.ld;
    double sum = 0.0;
    for (size_t j = 0; j < x.cols; ++j) {
        const double v = (double)x.base[row + j];
        sum += v * v;
    }
    return 1.0 / sqrt(sum / (double)x.cols + (double)epsilon);
}

DlStatus dl_rmsnorm_forward(DlConstMatrix x, DlConstMatrix scale,
                            float epsilon, DlMatrix y)
{
    if (!isfinite(epsilon) || epsilon <= 0.0f) return DL_ARGUMENT;
    const DlConstMatrix in[2] = {x, scale};
    DlStatus s = check_views(in, 2, &y, 1);
    if (s != DL_OK) return s;
    if (!same_shape(x, dl_matrix_read(y)) || scale.rows != 1 ||
        scale.cols != x.cols) return DL_SHAPE;
    if (!finite_cells(x) || !finite_cells(scale)) return DL_NUMERIC;
    for (size_t i = 0; i < x.rows; ++i) {
        const double r = inverse_rms(x, i, epsilon);
        if (!isfinite(r) || r <= 0.0) return DL_NUMERIC;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[x.offset + i * x.ld + j] * r;
            const double value = z * (double)scale.base[scale.offset + j];
            s = put_value(&y.base[y.offset + i * y.ld + j], value, DL_SET);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

DlStatus dl_rmsnorm_backward(DlConstMatrix x, DlConstMatrix scale,
                             DlConstMatrix dy, float epsilon,
                             DlMatrix dx, DlWriteMode dx_mode,
                             DlMatrix dscale, DlWriteMode scale_mode)
{
    if (!isfinite(epsilon) || epsilon <= 0.0f ||
        !valid_mode(dx_mode) || !valid_mode(scale_mode)) return DL_ARGUMENT;
    const DlConstMatrix in[3] = {x, scale, dy};
    const DlMatrix out[2] = {dx, dscale};
    DlStatus s = check_views(in, 3, out, 2);
    if (s != DL_OK) return s;
    if (scale.rows != 1 || scale.cols != x.cols || !same_shape(x, dy) ||
        !same_shape(x, dl_matrix_read(dx)) ||
        !same_shape(scale, dl_matrix_read(dscale))) return DL_SHAPE;
    if (!finite_cells(x) || !finite_cells(scale) || !finite_cells(dy) ||
        (dx_mode == DL_ADD && !finite_cells(dl_matrix_read(dx))) ||
        (scale_mode == DL_ADD && !finite_cells(dl_matrix_read(dscale)))) {
        return DL_NUMERIC;
    }
    if (scale_mode == DL_SET) {
        for (size_t j = 0; j < x.cols; ++j) dscale.base[dscale.offset + j] = 0.0f;
    }
    for (size_t i = 0; i < x.rows; ++i) {
        const double r = inverse_rms(x, i, epsilon);
        if (!isfinite(r) || r <= 0.0) return DL_NUMERIC;
        const size_t xr = x.offset + i * x.ld;
        const size_t gr = dy.offset + i * dy.ld;
        double dot = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[xr + j] * r;
            const double h = (double)dy.base[gr + j]
                           * (double)scale.base[scale.offset + j];
            dot += h * z;
        }
        const double mean_dot = dot / (double)x.cols;
        for (size_t j = 0; j < x.cols; ++j) {
            const double z = (double)x.base[xr + j] * r;
            const double g = (double)dy.base[gr + j];
            const double h = g * (double)scale.base[scale.offset + j];
            const double contribution = r * (h - z * mean_dot);
            s = put_value(&dx.base[dx.offset + i * dx.ld + j],
                          contribution, dx_mode);
            if (s != DL_OK) return s;
            s = put_value(&dscale.base[dscale.offset + j], g * z, DL_ADD);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

DlStatus dl_residual_add(DlConstMatrix a, DlConstMatrix b, DlMatrix out)
{
    const DlConstMatrix in[2] = {a, b};
    DlStatus s = check_views(in, 2, &out, 1);
    if (s != DL_OK) return s;
    if (!same_shape(a, b) || !same_shape(a, dl_matrix_read(out))) return DL_SHAPE;
    if (!finite_cells(a) || !finite_cells(b)) return DL_NUMERIC;
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            const double value = (double)a.base[a.offset + i * a.ld + j]
                               + (double)b.base[b.offset + i * b.ld + j];
            s = put_value(&out.base[out.offset + i * out.ld + j], value, DL_SET);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}

DlStatus dl_residual_backward(DlConstMatrix dy, DlMatrix dst, DlWriteMode mode)
{
    if (!valid_mode(mode)) return DL_ARGUMENT;
    DlStatus s = check_views(&dy, 1, &dst, 1);
    if (s != DL_OK) return s;
    if (!same_shape(dy, dl_matrix_read(dst))) return DL_SHAPE;
    if (!finite_cells(dy) ||
        (mode == DL_ADD && !finite_cells(dl_matrix_read(dst)))) return DL_NUMERIC;
    for (size_t i = 0; i < dy.rows; ++i) {
        for (size_t j = 0; j < dy.cols; ++j) {
            s = put_value(&dst.base[dst.offset + i * dst.ld + j],
                (double)dy.base[dy.offset + i * dy.ld + j], mode);
            if (s != DL_OK) return s;
        }
    }
    return DL_OK;
}
