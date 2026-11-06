#include "tiny.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static void status_is(DlStatus got, DlStatus want, const char *message)
{
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %s, expected %s\n", message,
            dl_status_name(got), dl_status_name(want));
        exit(EXIT_FAILURE);
    }
}

static void near(double got, double want, double atol, double rtol,
                 const char *message)
{
    if (!isfinite(got) || !isfinite(want) ||
        fabs(got - want) > atol + rtol * fmax(fabs(got), fabs(want))) {
        fprintf(stderr, "FAIL: %s: got %.17g, expected %.17g\n",
            message, got, want);
        exit(EXIT_FAILURE);
    }
}

static void fill(float *data, size_t count, float value)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = value;
    }
}

static void guards(DlMatrix m, float sentinel)
{
    for (size_t k = 0; k < m.count; ++k) {
        int logical = 0;
        for (size_t i = 0; i < m.rows; ++i) {
            const size_t start = m.offset + i * m.ld;
            if (k >= start && k - start < m.cols) {
                logical = 1;
            }
        }
        if (!logical) {
            require(m.base[k] == sentinel, "padding/guard changed");
        }
    }
}

static void test_uniform_and_contracts(void)
{
    float z[14], g[17];
    fill(z, 14, NAN);
    fill(g, 17, 91.0f);
    DlMatrix zm = {.base = z, .count = 14, .offset = 1,
        .rows = 2, .cols = 4, .ld = 6};
    DlMatrix gm = {.base = g, .count = 17, .offset = 2,
        .rows = 2, .cols = 4, .ld = 7};
    const size_t targets[2] = {0, 3};
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            z[zm.offset + i * zm.ld + j] = 0.0f;
            g[gm.offset + i * gm.ld + j] = NAN;
        }
    }
    double loss = -7.0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_SET), DL_OK, "uniform SET");
    near(loss, log(4.0), 1e-14, 1e-14, "uniform mean loss");
    for (size_t i = 0; i < 2; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < 4; ++j) {
            const double want = j == targets[i] ? -0.375 : 0.125;
            const double got = (double)g[gm.offset + i * gm.ld + j];
            near(got, want, 0.0, 0.0, "uniform gradient");
            sum += got;
        }
        near(sum, 0.0, 1e-7, 0.0, "row gradient sum");
    }
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_ADD), DL_OK, "uniform ADD");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            near((double)g[gm.offset + i * gm.ld + j],
                j == targets[i] ? -0.75 : 0.25, 0.0, 0.0, "doubled gradient");
        }
    }
    guards(gm, 91.0f);
    float saved[17];
    memcpy(saved, g, sizeof saved);
    loss = -7.0;
    const size_t bad[2] = {0, 4};
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), bad, 2,
        &loss, &gm, DL_SET), DL_RANGE, "late invalid target");
    require(loss == -7.0 && memcmp(saved, g, sizeof saved) == 0,
        "invalid target changed output");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 1,
        &loss, &gm, DL_SET), DL_SHAPE, "target count");
    DlMatrix wrong = gm;
    wrong.rows = 1;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &wrong, DL_SET), DL_SHAPE, "gradient shape");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &zm, DL_SET), DL_ALIAS, "loss output alias");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), NULL, 2,
        &loss, NULL, DL_SET), DL_ARGUMENT, "null targets");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        NULL, NULL, DL_SET), DL_ARGUMENT, "null loss");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, NULL, (DlWriteMode)99), DL_ARGUMENT, "invalid mode");
    DlMatrix empty = zm;
    empty.rows = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(empty), targets, 0,
        &loss, NULL, DL_SET), DL_ARGUMENT, "zero rows");
    const float nonfinite[3] = {NAN, INFINITY, -INFINITY};
    for (size_t k = 0; k < 3; ++k) {
        z[10] = nonfinite[k];
        status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
            &loss, &gm, DL_SET), DL_NUMERIC, "non-finite late logit");
        require(loss == -7.0 && memcmp(saved, g, sizeof saved) == 0,
            "preflight failure changed output");
    }
    z[10] = 0.0f;
    g[12] = NAN;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, &gm, DL_ADD), DL_NUMERIC, "ADD requires finite old gradient");
    require(loss == -7.0 && g[2] == saved[2], "ADD preflight wrote early row");
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), targets, 2,
        &loss, NULL, DL_SET), DL_OK, "value-only loss");
    near(loss, log(4.0), 1e-14, 1e-14, "value-only equals training loss");
    printf("uniform: loss = %.9f; target gradient = -0.375\n", loss);
}

static void test_extreme_logits(void)
{
    size_t target[1] = {2};
    float z[3] = {1000.0f, 999.0f, 998.0f}, g[3];
    DlMatrix zm = tiny_matrix(z, 1, 3);
    DlMatrix gm = tiny_matrix(g, 1, 3);
    double loss = -1.0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), target, 1,
        &loss, &gm, DL_SET), DL_OK, "shifted large logits");
    const double expected = 2.0 + log1p(exp(-1.0) + exp(-2.0));
    near(loss, expected, 1e-14, 1e-14, "large-logit loss");
    const double original = loss;
    for (size_t i = 0; i < 3; ++i) {
        z[i] -= 1000.0f;
    }
    status_is(dl_cross_entropy_mean(dl_matrix_read(zm), target, 1,
        &loss, NULL, DL_SET), DL_OK, "shift invariance");
    near(loss, original, 1e-14, 1e-14, "common shift");
    float huge[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    target[0] = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(tiny_matrix(huge, 1, 4)),
        target, 1, &loss, NULL, DL_SET), DL_OK, "huge tied maximum");
    near(loss, log(4.0), 1e-14, 1e-14, "do not add maximum then subtract");
    float extreme[2] = {FLT_MAX, -FLT_MAX}, eg[2];
    DlMatrix em = tiny_matrix(extreme, 1, 2);
    DlMatrix egm = tiny_matrix(eg, 1, 2);
    target[0] = 1;
    status_is(dl_cross_entropy_mean(dl_matrix_read(em), target, 1,
        &loss, &egm, DL_SET), DL_OK, "opposite finite extremes");
    near(loss, 2.0 * (double)FLT_MAX, 0.0, 1e-15, "wide target subtraction");
    require(loss > (double)FLT_MAX && eg[0] == 1.0f && eg[1] == -1.0f,
        "extreme loss needs double, gradient does not overflow");
    extreme[0] = 0.0f;
    extreme[1] = -50.0f;
    target[0] = 0;
    status_is(dl_cross_entropy_mean(dl_matrix_read(em), target, 1,
        &loss, &egm, DL_SET), DL_OK, "confident finite row");
    const double q = exp(-50.0);
    near(loss, log1p(q), 0.0, 1e-12, "tiny positive loss");
    near((double)eg[0], -q / (1.0 + q), 0.0, 2e-7, "tiny target derivative");
    near((double)eg[1], q / (1.0 + q), 0.0, 2e-7, "tiny other derivative");
    require(loss > 0.0 && eg[0] < 0.0f && eg[1] > 0.0f,
        "confidence must not silently erase representable derivatives");
    printf("confident: loss = %.9e; dtarget = %.9e\n", loss, (double)eg[0]);
    float single[1] = {3000.0f}, sg[1] = {NAN};
    DlMatrix sm = tiny_matrix(sg, 1, 1);
    status_is(dl_cross_entropy_mean(dl_matrix_read(tiny_matrix(single, 1, 1)),
        target, 1, &loss, &sm, DL_SET), DL_OK, "one class");
    require(loss == 0.0 && sg[0] == 0.0f, "one-class value and derivative");
    puts("extreme logits: shift, tied maxima, wide loss, one class passed");
}

static void test_embedding(void)
{
    float table[14], out[14], dt[14], dy[6] = {1, 2, 3, 4, 5, 6};
    fill(table, 14, 91.0f);
    fill(out, 14, 91.0f);
    fill(dt, 14, 91.0f);
    DlMatrix tm = {.base = table, .count = 14, .offset = 1,
        .rows = 4, .cols = 2, .ld = 3};
    DlMatrix om = {.base = out, .count = 14, .offset = 2,
        .rows = 3, .cols = 2, .ld = 4};
    DlMatrix dm = tm;
    dm.base = dt;
    const DlConstMatrix dyr = dl_matrix_read(tiny_matrix(dy, 3, 2));
    const size_t ids[3] = {2, 1, 2};
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            table[1 + v * 3 + j] = (float)(1 + 2 * v + j);
            dt[1 + v * 3 + j] = NAN;
        }
    }
    status_is(dl_embedding_forward(dl_matrix_read(tm), ids, 3, om),
        DL_OK, "padded gather");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)out[2 + i * 4 + j], (double)(1 + 2 * ids[i] + j),
                0.0, 0.0, "gathered value");
        }
    }
    guards(om, 91.0f);
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_SET), DL_OK, "scatter SET");
    const double expected[8] = {0, 0, 3, 4, 6, 8, 0, 0};
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)dt[1 + v * 3 + j], expected[v * 2 + j],
                0.0, 0.0, "repeated-ID sum");
            dt[1 + v * 3 + j] = 10.0f;
        }
    }
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_ADD), DL_OK, "scatter ADD");
    for (size_t v = 0; v < 4; ++v) {
        for (size_t j = 0; j < 2; ++j) {
            near((double)dt[1 + v * 3 + j], 10.0 + expected[v * 2 + j],
                0.0, 0.0, "preserved prior contribution");
        }
    }
    guards(dm, 91.0f);
    float saved[14], saved_out[14];
    memcpy(saved, dt, sizeof saved);
    memcpy(saved_out, out, sizeof saved_out);
    const size_t bad[3] = {2, 1, 4};
    status_is(dl_embedding_backward(dyr, bad, 3, dm, DL_SET), DL_RANGE, "late bad ID");
    require(memcmp(saved, dt, sizeof saved) == 0, "bad ID cleared table");
    status_is(dl_embedding_forward(dl_matrix_read(tm), bad, 3, om),
        DL_RANGE, "gather late bad ID");
    require(memcmp(saved_out, out, sizeof saved_out) == 0, "bad ID copied rows");
    status_is(dl_embedding_backward(dyr, ids, 2, dm, DL_SET), DL_SHAPE, "ID count");
    status_is(dl_embedding_backward(dyr, NULL, 3, dm, DL_SET), DL_ARGUMENT, "null IDs");
    status_is(dl_embedding_backward(dyr, ids, 3, dm, (DlWriteMode)99),
        DL_ARGUMENT, "embedding mode");
    DlMatrix alias = dm;
    alias.rows = 3;
    status_is(dl_embedding_backward(dl_matrix_read(alias), ids, 3, dm, DL_SET),
        DL_ALIAS, "scatter cannot clear input");
    DlMatrix gather_alias = tm;
    gather_alias.rows = 3;
    status_is(dl_embedding_forward(dl_matrix_read(tm), ids, 3, gather_alias),
        DL_ALIAS, "gather cannot overwrite table");
    DlMatrix narrow = dm;
    narrow.cols = 1;
    status_is(dl_embedding_backward(dyr, ids, 3, narrow, DL_SET), DL_SHAPE,
        "embedding feature width");
    require(memcmp(saved, dt, sizeof saved) == 0, "structural error wrote table");
    dy[0] = FLT_MAX;
    dy[4] = FLT_MAX;
    status_is(dl_embedding_backward(dyr, ids, 3, dm, DL_SET),
        DL_NUMERIC, "repeated-ID overflow");
    require(dt[7] == FLT_MAX && dt[1] == 0.0f,
        "numeric error follows documented partial writes");
    puts("embedding: padded gather, repeated scatter, ADD, rejection, overflow passed");
}

static void test_tiny_graph(void)
{
    Tiny t;
    tiny_init(&t);
    fill(t.dz, 12, NAN);
    fill(t.dh, 6, NAN);
    fill(t.de, 8, NAN);
    fill(t.dw, 8, NAN);
    double loss = -1.0;
    status_is(tiny_eval(&t, &loss, 1), DL_OK, "tiny graph");
    const double a = log(2.0);
    near(loss, (log(3.0) + 2.0 * log(6.0)) / 3.0, 2e-8, 2e-8, "tiny loss");
    const double dh[6] = {-a/9.0, a/6.0, a/6.0, 2.0*a/9.0,
        2.0*a/9.0, -a/6.0};
    const double de[8] = {a/9.0, 0.0, a/6.0, 2.0*a/9.0, 0, 0, 0, 0};
    const double dw[8] = {1.0/9.0, -5.0/18.0, -1.0/9.0, 1.0/18.0,
        -2.0/9.0, 1.0/9.0, 2.0/9.0, 1.0/9.0};
    for (size_t k = 0; k < 8; ++k) {
        near((double)t.de[k], de[k], 2e-8, 2e-7, "tiny dE");
        near((double)t.dw[k], dw[k], 2e-8, 2e-7, "tiny dW");
    }
    for (size_t k = 0; k < 6; ++k) {
        near((double)t.dh[k], dh[k], 2e-8, 2e-7, "tiny dH");
    }
    const DlConstMatrix dz = dl_matrix_read(tiny_matrix(t.dz, 3, 4));
    status_is(dl_linear_backward_input(dz, dl_matrix_read(tiny_matrix(t.w, 4, 2)),
        tiny_matrix(t.dh, 3, 2), DL_ADD), DL_OK, "input gradient ADD");
    status_is(dl_linear_backward_weight(dz, dl_matrix_read(tiny_matrix(t.h, 3, 2)),
        tiny_matrix(t.dw, 4, 2), DL_ADD), DL_OK, "weight gradient ADD");
    for (size_t k = 0; k < 6; ++k) {
        near((double)t.dh[k], 2.0 * dh[k], 4e-8, 2e-7, "input gradient doubled");
    }
    for (size_t k = 0; k < 8; ++k) {
        near((double)t.dw[k], 2.0 * dw[k], 4e-8, 2e-7, "weight gradient doubled");
    }
    printf("tiny graph: loss = %.9f; dE[0] = [%.9f, %.9f]\n",
        loss, (double)t.de[0], (double)t.de[1]);
}

int main(void)
{
    test_uniform_and_contracts();
    test_extreme_logits();
    test_embedding();
    test_tiny_graph();
    puts("loss and elementary backward checks passed");
    return EXIT_SUCCESS;
}
