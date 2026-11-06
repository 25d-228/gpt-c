#include "dl/kernels.h"

#include <float.h>
#include <math.h>

/* Validated spans only. Compare offsets only after matching root bases. */
static int spans_overlap(DlConstMatrix a, size_t a_end,
                         DlConstMatrix b, size_t b_end)
{
    return a.base == b.base && a.offset < b_end && b.offset < a_end;
}

typedef struct {
    size_t m, n, k;
    size_t ai, ak, bk, bj;
} MatmulPlan;

/* Private to this file; writes plan only after all geometry checks pass. */
static DlStatus matmul_prepare(DlConstMatrix a, DlTranspose ta,
                               DlConstMatrix b, DlTranspose tb,
                               DlMatrix c, DlWriteMode mode,
                               MatmulPlan *plan)
{
    if ((ta != DL_NOTRANS && ta != DL_TRANS) ||
        (tb != DL_NOTRANS && tb != DL_TRANS) ||
        (mode != DL_SET && mode != DL_ADD)) {
        return DL_ARGUMENT;
    }
    const DlConstMatrix cr = dl_matrix_read(c);
    size_t a_end = 0, b_end = 0, c_end = 0;
    DlStatus status = dl_matrix_end(a, &a_end);
    if (status == DL_OK) {
        status = dl_matrix_end(b, &b_end);
    }
    if (status == DL_OK) {
        status = dl_matrix_end(cr, &c_end);
    }
    if (status != DL_OK) {
        return status;
    }
    const size_t m = ta == DL_NOTRANS ? a.rows : a.cols;
    const size_t k = ta == DL_NOTRANS ? a.cols : a.rows;
    const size_t bk = tb == DL_NOTRANS ? b.rows : b.cols;
    const size_t n = tb == DL_NOTRANS ? b.cols : b.rows;
    if (k != bk || c.rows != m || c.cols != n) {
        return DL_SHAPE;
    }
    if (spans_overlap(a, a_end, cr, c_end) ||
        spans_overlap(b, b_end, cr, c_end)) {
        return DL_ALIAS;
    }
    *plan = (MatmulPlan){.m = m, .n = n, .k = k,
        .ai = ta == DL_NOTRANS ? a.ld : 1,
        .ak = ta == DL_NOTRANS ? 1 : a.ld,
        .bk = tb == DL_NOTRANS ? b.ld : 1,
        .bj = tb == DL_NOTRANS ? 1 : b.ld};
    return DL_OK;
}

DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode)
{
    MatmulPlan q = {0};
    const DlStatus status = matmul_prepare(a, ta, b, tb, c, mode, &q);
    if (status != DL_OK) {
        return status;
    }
    for (size_t i = 0; i < q.m; ++i) {
        for (size_t j = 0; j < q.n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < q.k; ++p) {
                sum += (double)a.base[a.offset + i * q.ai + p * q.ak]
                     * (double)b.base[b.offset + p * q.bk + j * q.bj];
            }
            const size_t ci = c.offset + i * c.ld + j;
            if (mode == DL_ADD) {
                sum += (double)c.base[ci];
            }
            if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            c.base[ci] = (float)sum;
        }
    }
    return DL_OK;
}

DlStatus dl_matmul_blocked(DlConstMatrix a, DlTranspose ta,
                           DlConstMatrix b, DlTranspose tb,
                           DlMatrix c, DlWriteMode mode)
{
    MatmulPlan q = {0};
    const DlStatus status = matmul_prepare(a, ta, b, tb, c, mode, &q);
    if (status != DL_OK) {
        return status;
    }
    double acc[DL_BLOCK_M][DL_BLOCK_N];
    float panel[DL_BLOCK_K][DL_BLOCK_N];
    for (size_t ii = 0; ii < q.m; ) {
        const size_t im = q.m - ii < DL_BLOCK_M ? q.m - ii : DL_BLOCK_M;
        for (size_t jj = 0; jj < q.n; ) {
            const size_t jn = q.n - jj < DL_BLOCK_N ? q.n - jj : DL_BLOCK_N;
            for (size_t i = 0; i < im; ++i) {
                for (size_t j = 0; j < jn; ++j) {
                    acc[i][j] = 0.0;
                }
            }
            for (size_t pp = 0; pp < q.k; ) {
                const size_t pk = q.k - pp < DL_BLOCK_K
                    ? q.k - pp : DL_BLOCK_K;
                for (size_t p = 0; p < pk; ++p) {
                    for (size_t j = 0; j < jn; ++j) {
                        panel[p][j] = b.base[b.offset
                            + (pp + p) * q.bk + (jj + j) * q.bj];
                    }
                }
                for (size_t i = 0; i < im; ++i) {
                    for (size_t p = 0; p < pk; ++p) {
                        const double av = (double)a.base[a.offset
                            + (ii + i) * q.ai + (pp + p) * q.ak];
                        for (size_t j = 0; j < jn; ++j) {
                            acc[i][j] += av * (double)panel[p][j];
                        }
                    }
                }
                pp += pk;
            }
            for (size_t i = 0; i < im; ++i) {
                for (size_t j = 0; j < jn; ++j) {
                    const size_t ci = c.offset + (ii + i) * c.ld + jj + j;
                    double sum = acc[i][j];
                    if (mode == DL_ADD) {
                        sum += (double)c.base[ci];
                    }
                    if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                        return DL_NUMERIC;
                    }
                    c.base[ci] = (float)sum;
                }
            }
            jj += jn;
        }
        ii += im;
    }
    return DL_OK;
}

DlStatus dl_matmul(DlBackend backend, DlConstMatrix a, DlTranspose ta,
                   DlConstMatrix b, DlTranspose tb,
                   DlMatrix c, DlWriteMode mode)
{
    switch (backend) {
    case DL_BACKEND_REF:
        return dl_matmul_ref(a, ta, b, tb, c, mode);
    case DL_BACKEND_BLOCKED:
        return dl_matmul_blocked(a, ta, b, tb, c, mode);
    default:
        return DL_ARGUMENT;
    }
}

/* Validate all IDs before any caller can copy or clear its output. */
static DlStatus check_ids(const size_t *ids, size_t count,
                           size_t expected, size_t vocabulary)
{
    if (ids == NULL) {
        return DL_ARGUMENT;
    }
    if (count != expected) {
        return DL_SHAPE;
    }
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= vocabulary) {
            return DL_RANGE;
        }
    }
    return DL_OK;
}

DlStatus dl_embedding_forward(DlConstMatrix table, const size_t *ids,
                              size_t ids_count, DlMatrix out)
{
    const DlConstMatrix out_read = dl_matrix_read(out);
    size_t table_end = 0, out_end = 0;
    DlStatus status = dl_matrix_end(table, &table_end);
    if (status == DL_OK) {
        status = dl_matrix_end(out_read, &out_end);
    }
    if (status != DL_OK) {
        return status;
    }
    if (table.cols != out.cols) {
        return DL_SHAPE;
    }
    status = check_ids(ids, ids_count, out.rows, table.rows);
    if (status != DL_OK) {
        return status;
    }
    if (spans_overlap(table, table_end, out_read, out_end)) {
        return DL_ALIAS;
    }
    for (size_t i = 0; i < out.rows; ++i) {
        for (size_t j = 0; j < out.cols; ++j) {
            const float value = table.base[table.offset + ids[i] * table.ld + j];
            if (!isfinite(value)) {
                return DL_NUMERIC;
            }
            out.base[out.offset + i * out.ld + j] = value;
        }
    }
    return DL_OK;
}

DlStatus dl_embedding_backward(DlConstMatrix dy, const size_t *ids,
                               size_t ids_count, DlMatrix dtable,
                               DlWriteMode mode)
{
    if (mode != DL_SET && mode != DL_ADD) {
        return DL_ARGUMENT;
    }
    const DlConstMatrix dr = dl_matrix_read(dtable);
    size_t dy_end = 0, dt_end = 0;
    DlStatus status = dl_matrix_end(dy, &dy_end);
    if (status == DL_OK) {
        status = dl_matrix_end(dr, &dt_end);
    }
    if (status != DL_OK) {
        return status;
    }
    if (dy.cols != dtable.cols) {
        return DL_SHAPE;
    }
    status = check_ids(ids, ids_count, dy.rows, dtable.rows);
    if (status != DL_OK) {
        return status;
    }
    if (spans_overlap(dy, dy_end, dr, dt_end)) {
        return DL_ALIAS;
    }
    if (mode == DL_SET) {
        for (size_t v = 0; v < dtable.rows; ++v) {
            for (size_t j = 0; j < dtable.cols; ++j) {
                dtable.base[dtable.offset + v * dtable.ld + j] = 0.0f;
            }
        }
    }
    for (size_t i = 0; i < dy.rows; ++i) {
        for (size_t j = 0; j < dy.cols; ++j) {
            const size_t dst = dtable.offset + ids[i] * dtable.ld + j;
            const double sum = (double)dtable.base[dst]
                + (double)dy.base[dy.offset + i * dy.ld + j];
            if (!isfinite(sum) || fabs(sum) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            dtable.base[dst] = (float)sum;
        }
    }
    return DL_OK;
}

DlStatus dl_linear_backward_input(DlConstMatrix dy, DlConstMatrix w,
                                  DlMatrix dx, DlWriteMode mode)
{
    return dl_matmul_ref(dy, DL_NOTRANS, w, DL_NOTRANS, dx, mode);
}

DlStatus dl_linear_backward_weight(DlConstMatrix dy, DlConstMatrix x,
                                   DlMatrix dw, DlWriteMode mode)
{
    return dl_matmul_ref(dy, DL_TRANS, x, DL_NOTRANS, dw, mode);
}
