#include "dot.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    const size_t n = sizeof a / sizeof a[0];

    const float result = dot_f32(a, b, n);
    const float empty = dot_f32(NULL, NULL, 0);

    printf("dot = %.0f\n", (double)result);
    printf("empty dot = %.0f\n", (double)empty);
    printf("elements = %zu, bytes per element = %zu\n", n, sizeof a[0]);
    if (result != 32.0f || empty != 0.0f) {
        fprintf(stderr, "FAIL: normal or empty dot product\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a, b, 1) != 4.0f) {
        fprintf(stderr, "FAIL: one-element prefix\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a + 1, b + 1, n - 1) != 28.0f) {
        fprintf(stderr, "FAIL: two-element suffix\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a, a, n) != 14.0f) {
        fprintf(stderr, "FAIL: overlapping read-only inputs\n");
        return EXIT_FAILURE;
    }

    puts("all checks passed");
    return EXIT_SUCCESS;
}
