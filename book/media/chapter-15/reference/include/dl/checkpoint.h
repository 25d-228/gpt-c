#ifndef DL_CHECKPOINT_H
#define DL_CHECKPOINT_H

#include "dl/train.h"

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

#endif
