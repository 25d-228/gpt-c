#ifndef DL_EXAMPLE_DOT_H
#define DL_EXAMPLE_DOT_H

#include <stddef.h>

/*
 * Read n initialized floats from each input and return their dot product.
 * For n > 0, a and b must each address at least n live float elements.
 * For n == 0, either pointer may be NULL; no input is accessed.
 * Inputs may overlap. This function writes neither input, retains no pointer,
 * and performs no allocation. Invalid extents are a caller error, not a status.
 */
float dot_f32(const float *a, const float *b, size_t n);

#endif
