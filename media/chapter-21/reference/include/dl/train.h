#ifndef DL_TRAIN_H
#define DL_TRAIN_H

#include "dl/data.h"
#include "dl/optim.h"

typedef struct {
    uint64_t total, warmup;
    double peak, floor;
} DlSchedule;

/* Value state, no owned pointers. Only the trainer/checkpoint API mutates it.
 * Fixed mode repeats the FIRST full batch; random mode samples with replacement.
 * A boundary has optim.active==0, tokens==0, no fresh parameter gradients.
 * All dataset bytes/configuration stay immutable for a run. */
typedef struct {
    DlSchedule schedule;
    DlRng rng;
    size_t batch, time, microbatches;
    DlBackend backend;
    int fixed;
    uint64_t tokens_seen;
    DlAttentionMode attention;      /* Materialized=0 for legacy initialization. */
    size_t query_tile, key_tile;    /* Both zero for materialized; positive for tiled. */
} DlTrainState;

typedef struct {
    double loss;
    DlOptimStats optim;
} DlTrainStats;

typedef struct {
    double loss;
    uint64_t tokens;
    size_t batches, unscored;
} DlEvalStats;

/* LR for NEXT update; completed==total is end (DL_RANGE). Output untouched
 * on failure. Positive finite floor<=peak, 0<=warmup<total<=2^53. */
DlStatus dl_schedule_rate(DlSchedule s, uint64_t completed, double *out);
/* Check boundary, schedule, counters and exact bytes-per-update arithmetic.
 * No mutation. Data owners need at least one full B*T target batch each. */
DlStatus dl_train_boundary(const DlAdamW *optim, const DlTrainState *state,
                           const DlByteData *training, const DlByteData *valid);
/* Train tape/batch match state B,T. All outputs/storage disjoint, constructed
 * owners unmodified, no concurrent calls. One complete group or no update:
 * after an operational failure, RNG/counters/weights/moments unchanged and
 * partial gradients discarded. Tape/batch scratch may have changed. */
DlStatus dl_train_update(DlAdamW *optim, DlTrainState *state,
                         const DlByteData *data, DlByteBatch *batch,
                         DlTrainTape *tape, DlTrainStats *stats);
/* Separate evaluation tape AND batch (never pending training work). Limit 0
 * means all full batches; otherwise stop after limit. Reset cursor locally.
 * No optimizer/RNG/parameter-gradient writes. out untouched on failure.
 * unscored includes trailing targets AND any remainder excluded by limit. */
DlStatus dl_evaluate(DlModel *model, DlTrainTape *evaluation_tape,
                      DlBackend backend, const DlByteData *data,
                      DlByteBatch *evaluation_batch, size_t limit,
                      DlEvalStats *out);

#endif
