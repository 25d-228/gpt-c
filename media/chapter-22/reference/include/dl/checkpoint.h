#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"
#include "dl/inference.h"

/* Versions 1/2: explicit little-endian integers and IEEE binary32/binary64.
 * Materialized training writes the unchanged v1 format. Tiled training writes
 * v2 (magic DLCKPT02, version=2, 24-byte policy extension at byte 256,
 * trailer ENDCKP02). The extension holds attention=1, query_tile, key_tile.
 * Resume must match the saved strategy/tiles; v1 means materialized only.
 * Existing constructed model/optimizer/state must describe the SAME run:
 * model config, optimizer config, batching, matrix backend, attention strategy/tiles, schedule and fixed mode.
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

/* Inference-only read of training v1/v2 or weight-only DLWGHT01. expected config is
 * supplied by the caller, never used unchecked from the file. Empty out only.
 * Allocates P weight floats, not a DlModel, gradients, moments or datasets.
 * Validates header/directory/finite weights/moments/trailer/exact EOF; consumes
 * embedded data in bounded chunks without requiring original dataset files.
 * Unlike resume, does NOT compare dataset identity or import trainer RNG/state.
 * DLWGHT01 contains a 128-byte header, 96-byte canonical directory records,
 * FP32 payload, and ENDWGT01 trailer, never optimizer/run/data state. It sets
 * weights.position_limit; all inference/cache constructors enforce that limit.
 * Canonical names/shapes/counts/offsets are validated before reading payload.
 * max_file_bytes bounds logical file bytes (8-byte magic is read twice, plus
 * one EOF probe); input must be a seekable ordinary file. max_float_bytes
 * bounds weights.
 * Only after checked close does out become immutable ready weights. Failure
 * frees the entire candidate and leaves out unchanged. Not an integrity or
 * authenticity check: structurally valid changed finite values can pass. */
DlStatus dl_checkpoint_load_weights(const char *path, DlWeights *out,
                                     DlModelConfig expected,
                                     uint64_t max_file_bytes,
                                     size_t max_float_bytes);

#endif
