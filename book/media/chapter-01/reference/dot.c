#include "dot.h"

float dot_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float product = a[i] * b[i];
        sum = sum + product;
    }
    return sum;
}
