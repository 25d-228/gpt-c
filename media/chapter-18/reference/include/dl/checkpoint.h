#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"
#include "dl/inference.h"

/* Version 1: explicit little-endian integers and IEEE binary32/binary64.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, backend, schedule and fixed mode.
 * Dataset bytes are included and compared exactly (not merely file lengths).
 * No untrusted file field controls an allocation. max_file_bytes also bounds
 * the temporary 3*P*sizeof(float) load payload; metadata/stdio are excluded.
 * Ordinary local files in a caller-controlled directory; no concurrent edits,
 * writers, adversarial paths, live-owner moves, or aliasing of output state.
 * DL_SHAPE: incompatibility; DL_RANGE: truncation/budget/trailing bytes;
 * DL_NUMERIC: bad weights/moments; DL_STATE: lifecycle or I/O failure.
 * SAVE: write sibling temporary file, flush/sync/close, then rename. Until the
 * successful rename, an existing destination remains untouched. No automatic
 * retries or power-loss durability guarantee (no parent-directory/full sync).
 * LOAD: read/check/close candidate first, then coordinated in-place restore.
 * A failed load preserves all live state. Success increments the LIVE model
 * revision (never imports a saved revision), invalidating every old tape. */
DlStatus dl_checkpoint_save(const char *path, const DlAdamW *optim,
                            const DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);
DlStatus dl_checkpoint_load(const char *path, DlAdamW *optim,
                            DlTrainState *state,
                            const DlByteData *training, const DlByteData *valid,
                            uint64_t max_file_bytes);

/* Inference-only read of the SAME version-1 training file. expected config is
 * supplied by the caller, never used unchecked from the file. Empty out only.
 * Allocates P weight floats, not a DlModel, gradients, moments or datasets.
 * Validates header/directory/finite weights/moments/trailer/exact EOF; consumes
 * embedded data in bounded chunks without requiring original dataset files.
 * Unlike resume, does NOT compare dataset identity or import trainer RNG/state.
 * max_file_bytes bounds parsed bytes (plus one EOF probe); max_float_bytes
 * bounds weights.
 * Only after checked close does out become immutable ready weights. Failure
 * frees the entire candidate and leaves out unchanged. Not an integrity or
 * authenticity check: structurally valid changed finite values can pass. */
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes);

#endif
