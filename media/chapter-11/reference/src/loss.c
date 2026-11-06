#include "dl/loss.h"

#include <float.h>
#include <math.h>

DlStatus dl_cross_entropy_mean(DlConstMatrix logits, const size_t *targets,
                               size_t target_count, double *loss,
                               DlMatrix *dlogits, DlWriteMode mode)
{
    if (targets == NULL || loss == NULL ||
        (mode != DL_SET && mode != DL_ADD)) {
        return DL_ARGUMENT;
    }
    size_t logits_end = 0;
    DlStatus status = dl_matrix_end(logits, &logits_end);
    if (status != DL_OK) {
        return status;
    }
    if (target_count != logits.rows) {
        return DL_SHAPE;
    }
    if (dlogits != NULL) {
        const DlConstMatrix gr = dl_matrix_read(*dlogits);
        size_t grad_end = 0;
        status = dl_matrix_end(gr, &grad_end);
        if (status != DL_OK) {
            return status;
        }
        if (gr.rows != logits.rows || gr.cols != logits.cols) {
            return DL_SHAPE;
        }
        if (gr.base == logits.base && gr.offset < logits_end &&
            logits.offset < grad_end) {
            return DL_ALIAS;
        }
    }
    for (size_t i = 0; i < logits.rows; ++i) {
        if (targets[i] >= logits.cols) {
            return DL_RANGE;
        }
    }
    /* Preflight every row, not only the first row that will be written. */
    for (size_t i = 0; i < logits.rows; ++i) {
        for (size_t j = 0; j < logits.cols; ++j) {
            if (!isfinite(logits.base[logits.offset + i * logits.ld + j])) {
                return DL_NUMERIC;
            }
            if (dlogits != NULL && mode == DL_ADD &&
                !isfinite(dlogits->base[dlogits->offset + i * dlogits->ld + j])) {
                return DL_NUMERIC;
            }
        }
    }
    const double rows = (double)logits.rows;
    double total = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        const size_t row = logits.offset + i * logits.ld;
        size_t maximum_index = 0;
        double maximum = (double)logits.base[row];
        for (size_t j = 1; j < logits.cols; ++j) {
            const double value = (double)logits.base[row + j];
            if (value > maximum) {
                maximum = value;
                maximum_index = j;
            }
        }
        double other = 0.0;
        double not_target = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) {
            const double term = exp((double)logits.base[row + j] - maximum);
            if (j != maximum_index) {
                other += term;
            }
            if (j != targets[i]) {
                not_target += term;
            }
        }
        const double denominator = 1.0 + other;
        const double row_loss =
            (maximum - (double)logits.base[row + targets[i]]) + log1p(other);
        total += row_loss / rows;
        if (!isfinite(denominator) || !isfinite(total)) {
            return DL_NUMERIC;
        }
        if (dlogits == NULL) {
            continue;
        }
        for (size_t j = 0; j < logits.cols; ++j) {
            const double numerator = j == targets[i] ? -not_target
                : exp((double)logits.base[row + j] - maximum);
            double value = (numerator / denominator) / rows;
            const size_t dst = dlogits->offset + i * dlogits->ld + j;
            if (mode == DL_ADD) {
                value += (double)dlogits->base[dst];
            }
            if (!isfinite(value) || fabs(value) > (double)FLT_MAX) {
                return DL_NUMERIC;
            }
            dlogits->base[dst] = (float)value;
        }
    }
    *loss = total;
    return DL_OK;
}
