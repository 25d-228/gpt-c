#ifndef DL_KERNELS_H
#define DL_KERNELS_H

#include "dl/tensor.h"

typedef enum { DL_NOTRANS = 0, DL_TRANS = 1 } DlTranspose;
typedef enum { DL_SET = 0, DL_ADD = 1 } DlWriteMode;
typedef enum { DL_BACKEND_REF = 0, DL_BACKEND_BLOCKED = 1 } DlBackend;

/* Fixed workspace bounds, not matrix-shape restrictions. */
enum { DL_BLOCK_M = 8, DL_BLOCK_N = 32, DL_BLOCK_K = 64 };

/* C = op(A) op(B), or C += op(A) op(B). No allocation or retained pointers.
 * Views must keep truthful live base/count metadata from their owners.
 * Inputs may overlap each other. Output's bounding span must not overlap
 * either input's span, including gaps between rows. All dimensions positive.
 * A/B elements must be initialized. SET never reads old C elements;
 * ADD requires initialized C elements. Only logical C elements are written.
 * Geometry/shape/alias errors leave C unchanged. DL_NUMERIC may follow
 * earlier cell writes: discard the whole result on that status.
 * Products and reduction use double; each accepted result is stored as float. */
DlStatus dl_matmul_ref(DlConstMatrix a, DlTranspose ta,
                       DlConstMatrix b, DlTranspose tb,
                       DlMatrix c, DlWriteMode mode);

/* Same view, precision, and SET/ADD contract as the reference.
 * Scratch: M*N doubles and K*N floats, using the block constants above.
 * Numeric failure may write a different subset than the scalar traversal. */
DlStatus dl_matmul_blocked(DlConstMatrix a, DlTranspose ta,
                           DlConstMatrix b, DlTranspose tb,
                           DlMatrix c, DlWriteMode mode);

/* Explicit selection per call; no global backend or automatic fallback.
 * An unknown backend returns DL_ARGUMENT without writing C. */
DlStatus dl_matmul(DlBackend backend, DlConstMatrix a, DlTranspose ta,
                   DlConstMatrix b, DlTranspose tb,
                   DlMatrix c, DlWriteMode mode);

/* IDs are a separate live size_t array, disjoint from float storage.
 * ids_count == out.rows; table [V,D], out [M,D]; every id is < V.
 * Only selected table rows are read. Output must not overlap table's span.
 * No writes on structural/range/alias errors; numeric failure may be partial. */
DlStatus dl_embedding_forward(DlConstMatrix table, const size_t *ids,
                              size_t ids_count, DlMatrix out);

/* dy [M,D], dtable [V,D]. SET zeros every logical dtable cell once,
 * then scatters with addition, including repeated IDs. ADD preserves prior
 * contributions. dy and dtable spans must be disjoint. ADD requires all
 * logical dtable cells initialized; accessed values must be finite.
 * Structural/range/alias errors leave dtable unchanged; DL_NUMERIC may not. */
DlStatus dl_embedding_backward(DlConstMatrix dy, const size_t *ids,
                               size_t ids_count, DlMatrix dtable,
                               DlWriteMode mode);

/* For forward Y = X W^T, X [M,K], W [N,K], dY [M,N].
 * These convenience wrappers remain reference-only. To choose a backend,
 * call dl_matmul with NN for dX or TN for dW.
 * Independent calls inherit matmul's SET/ADD, alias, and failure rules.
 * Do not overwrite X or W before every backward consumer has finished. */
DlStatus dl_linear_backward_input(DlConstMatrix dy, DlConstMatrix w,
                                  DlMatrix dx, DlWriteMode mode);
DlStatus dl_linear_backward_weight(DlConstMatrix dy, DlConstMatrix x,
                                   DlMatrix dw, DlWriteMode mode);

#endif
