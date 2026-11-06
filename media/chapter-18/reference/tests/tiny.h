#ifndef DL_TEST_TINY_H
#define DL_TEST_TINY_H

#include "dl/loss.h"

#include <math.h>

/* Owns inline arrays, not heap pointers. Views never outlive this object. */
typedef struct {
    float e[8], w[8];
    float h[6], z[12], dz[12], dh[6], de[8], dw[8];
    size_t ids[3], targets[3];
} Tiny;

/* Test-only helper: caller provides exactly rows*cols live float elements.
 * Tiny fixtures use small fixed dimensions; this is not an allocator/API. */
static DlMatrix tiny_matrix(float *base, size_t rows, size_t cols)
{
    return (DlMatrix){.base = base, .count = rows * cols, .offset = 0,
        .rows = rows, .cols = cols, .ld = cols};
}

static void tiny_init(Tiny *t)
{
    const float a = (float)log(2.0);
    *t = (Tiny){.e = {1.0f, 0.0f, 0.0f, 1.0f,
                     -1.0f, 0.5f, 0.25f, -0.5f},
        .w = {0.0f, 0.0f, a, 0.0f, 0.0f, a, a, a},
        .ids = {0, 1, 0}, .targets = {1, 0, 2}};
}

static DlStatus tiny_eval(Tiny *t, double *loss, int backward)
{
    const DlMatrix e = tiny_matrix(t->e, 4, 2);
    const DlMatrix w = tiny_matrix(t->w, 4, 2);
    const DlMatrix h = tiny_matrix(t->h, 3, 2);
    const DlMatrix z = tiny_matrix(t->z, 3, 4);
    DlMatrix dz = tiny_matrix(t->dz, 3, 4);
    DlStatus status = dl_embedding_forward(dl_matrix_read(e), t->ids, 3, h);
    if (status == DL_OK) {
        status = dl_matmul_ref(dl_matrix_read(h), DL_NOTRANS,
            dl_matrix_read(w), DL_TRANS, z, DL_SET);
    }
    if (status == DL_OK) {
        status = dl_cross_entropy_mean(dl_matrix_read(z), t->targets, 3,
            loss, backward ? &dz : NULL, DL_SET);
    }
    if (status != DL_OK || !backward) {
        return status;
    }
    const DlMatrix dh = tiny_matrix(t->dh, 3, 2);
    status = dl_linear_backward_input(dl_matrix_read(dz), dl_matrix_read(w),
        dh, DL_SET);
    if (status == DL_OK) {
        status = dl_linear_backward_weight(dl_matrix_read(dz),
            dl_matrix_read(h), tiny_matrix(t->dw, 4, 2), DL_SET);
    }
    if (status == DL_OK) {
        status = dl_embedding_backward(dl_matrix_read(dh), t->ids, 3,
            tiny_matrix(t->de, 4, 2), DL_SET);
    }
    return status;
}

#endif
