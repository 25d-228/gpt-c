#include "tiny.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Float endpoints are the values actually written into model storage. */
static int endpoints(float original, double relative_step,
                     float *minus, float *plus)
{
    if (!isfinite(original) || !isfinite(relative_step) || relative_step <= 0.0) {
        return 0;
    }
    const double step = relative_step * fmax(1.0, fabs((double)original));
    const double low = (double)original - step;
    const double high = (double)original + step;
    if (!isfinite(low) || !isfinite(high) ||
        fabs(low) > (double)FLT_MAX || fabs(high) > (double)FLT_MAX) {
        return 0;
    }
    *minus = (float)low;
    *plus = (float)high;
    return *minus < original && original < *plus;
}

static int compare(const char *name, size_t index, double step,
                    double analytic, double numerical, double *worst)
{
    const double tolerance = 3e-5 + 2e-3 * fmax(fabs(analytic), fabs(numerical));
    const double ratio = fabs(analytic - numerical) / tolerance;
    if (!isfinite(analytic) || !isfinite(numerical) || ratio > 1.0) {
        fprintf(stderr, "FAIL %s[%zu] step=%g: analytic=%.9g numerical=%.9g "
            "tolerance=%.3g\n", name, index, step, analytic, numerical, tolerance);
        return 0;
    }
    if (ratio > *worst) {
        *worst = ratio;
    }
    return 1;
}

static int check_parameters(Tiny *t, const char *name, float *parameter,
                             const float *gradient, double step)
{
    double worst = 0.0;
    for (size_t k = 0; k < 8; ++k) {
        const float original = parameter[k];
        float minus = 0.0f, plus = 0.0f;
        if (!endpoints(original, step, &minus, &plus)) {
            fprintf(stderr, "FAIL %s[%zu]: step has no two finite endpoints\n", name, k);
            return 0;
        }
        double lp = 0.0, lm = 0.0;
        parameter[k] = plus;
        const DlStatus sp = tiny_eval(t, &lp, 0);
        parameter[k] = minus;
        const DlStatus sm = tiny_eval(t, &lm, 0);
        parameter[k] = original; /* Restore before any status/accuracy return. */
        if (sp != DL_OK || sm != DL_OK) {
            fprintf(stderr, "FAIL %s[%zu]: evaluations %s / %s\n", name, k,
                dl_status_name(sp), dl_status_name(sm));
            return 0;
        }
        const double numerical = (lp - lm) / ((double)plus - (double)minus);
        if (!compare(name, k, step, (double)gradient[k], numerical, &worst)) {
            return 0;
        }
    }
    printf("%s: 8 coordinates, step=%g, worst tolerance fraction=%.4g\n",
        name, step, worst);
    return 1;
}

static int check_direct_loss(void)
{
    float z[8] = {0.2f, -0.7f, 1.1f, 0.3f, -0.6f, 0.5f, 0.8f, -0.1f};
    float gradient[8], original_array[8];
    memcpy(original_array, z, sizeof z);
    DlMatrix zm = tiny_matrix(z, 2, 4);
    DlMatrix gm = tiny_matrix(gradient, 2, 4);
    const size_t targets[2] = {2, 0};
    double loss = 0.0, worst = 0.0;
    DlStatus status = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_SET);
    if (status != DL_OK) {
        fprintf(stderr, "FAIL direct loss: %s\n", dl_status_name(status));
        return 0;
    }
    for (size_t k = 0; k < 8; ++k) {
        const float original = z[k];
        float minus = 0.0f, plus = 0.0f;
        if (!endpoints(original, 0.003, &minus, &plus)) {
            return 0;
        }
        double lp = 0.0, lm = 0.0;
        z[k] = plus;
        const DlStatus sp = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &lp, NULL, DL_SET);
        z[k] = minus;
        const DlStatus sm = dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &lm, NULL, DL_SET);
        z[k] = original;
        if (sp != DL_OK || sm != DL_OK) {
            fprintf(stderr, "FAIL direct loss coordinate %zu\n", k);
            return 0;
        }
        const double numerical = (lp - lm) / ((double)plus - (double)minus);
        if (!compare("logits", k, 0.003, (double)gradient[k], numerical, &worst)) {
            return 0;
        }
    }
    if (memcmp(original_array, z, sizeof z) != 0) {
        fputs("FAIL direct loss did not restore storage\n", stderr);
        return 0;
    }
    printf("logits: 8 coordinates, worst tolerance fraction=%.4g\n", worst);
    return 1;
}

static int check_graph(int varied)
{
    Tiny t;
    tiny_init(&t);
    if (varied) {
        const float e[8] = {0.7f, -0.2f, 0.3f, 0.9f, -0.4f, 0.6f, 0.1f, -0.8f};
        const float w[8] = {0.2f, -0.1f, -0.3f, 0.4f, 0.5f, 0.2f, -0.2f, -0.4f};
        memcpy(t.e, e, sizeof e);
        memcpy(t.w, w, sizeof w);
    }
    float e_before[8], w_before[8];
    memcpy(e_before, t.e, sizeof t.e);
    memcpy(w_before, t.w, sizeof t.w);
    double baseline = 0.0;
    DlStatus status = tiny_eval(&t, &baseline, 1);
    if (status != DL_OK) {
        fprintf(stderr, "FAIL baseline graph: %s\n", dl_status_name(status));
        return 0;
    }
    printf("graph fixture %d: baseline loss=%.9f\n", varied, baseline);
    const double steps[3] = {0.01, 0.003, 0.001};
    for (size_t k = 0; k < 3; ++k) {
        if (!check_parameters(&t, "E", t.e, t.de, steps[k]) ||
            !check_parameters(&t, "W", t.w, t.dw, steps[k])) {
            return 0;
        }
    }
    double restored_loss = 0.0;
    status = tiny_eval(&t, &restored_loss, 0);
    if (status != DL_OK || memcmp(e_before, t.e, sizeof t.e) != 0 ||
        memcmp(w_before, t.w, sizeof t.w) != 0 || restored_loss != baseline) {
        fputs("FAIL restored graph differs from baseline\n", stderr);
        return 0;
    }
    return 1;
}

int main(void)
{
    if (!check_direct_loss() || !check_graph(0) || !check_graph(1)) {
        return EXIT_FAILURE;
    }
    puts("104 finite-difference comparisons passed; parameter storage restored");
    return EXIT_SUCCESS;
}
