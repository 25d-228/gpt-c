#include "dl/kernels.h"

#include <float.h>
#include <math.h>

/* Validated spans only. Compare offsets only after matching root bases. */
static int spans_overlap(DlConstMatrix a, size_t a_end,
                         DlConstMatrix b, size_t b_end)
{
    return a.base == b.base && a.offset < b_end && b.offset < a_end;
}

DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode)
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
    const size_t ai = ta == DL_NOTRANS ? a.ld : 1;
    const size_t ak = ta == DL_NOTRANS ? 1 : a.ld;
    const size_t bk_stride = tb == DL_NOTRANS ? b.ld : 1;
    const size_t bj = tb == DL_NOTRANS ? 1 : b.ld;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += (double)a.base[a.offset + i * ai + p * ak]
                     * (double)b.base[b.offset + p * bk_stride + j * bj];
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
