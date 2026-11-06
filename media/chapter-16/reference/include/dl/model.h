#ifndef DL_MODEL_H
#define DL_MODEL_H

#include "dl/attention.h"
#include "dl/ffn.h"
#include "dl/loss.h"
#include "dl/norm.h"
#include "dl/rng.h"

typedef struct {
    size_t vocab, layers, width, query_heads, kv_heads, ffn_width;
    float epsilon, init_std;
    double rope_theta;
    uint64_t seed;
} DlModelConfig;

/* Nine independent parameters per block; table first, final scale last. */
enum {
    DL_P_ATTN_NORM, DL_P_QUERY, DL_P_KEY, DL_P_VALUE, DL_P_OUTPUT,
    DL_P_FFN_NORM, DL_P_GATE, DL_P_UP, DL_P_DOWN, DL_BLOCK_PARAMS
};

typedef struct {
    char name[64];
    DlTensor value, grad;           /* Unique owners, both dense rank two. */
    int decay;                      /* Matrices: 1. Norm scales: 0. */
} DlParameter;

typedef struct {
    DlModelConfig config;
    DlParameter *params;            /* Owns param_count initialized records. */
    size_t param_count, value_count;
    uint64_t revision;
    int gradients_valid;
} DlModel;

typedef struct {
    DlMatrix norm1, attention_out, middle, norm2, ffn_out;
    DlAttentionCache attention;
    DlFfnCache ffn;
} DlBlockTape;

typedef struct {
    const DlModel *model;            /* Borrow: model must outlive this tape. */
    DlAttentionConfig attention_config;
    size_t rows;
    DlTensor arena;                 /* Sole float owner for all views below. */
    DlMatrix *hidden;               /* Owns layers+1 view records, NOT floats. */
    DlBlockTape *blocks;            /* Owns layers view records. */
    size_t *metadata;               /* Owns 3*rows size_t: IDs/targets/positions. */
    size_t *ids, *targets, *positions;
    double *inv_freq;               /* Owns head_dim/2 frequencies. */
    DlMatrix final_norm, logits, dlogits;
    DlMatrix g, other, du, dn, embedding_grad;
    DlAttentionScratch attention_scratch;
    DlFfnScratch ffn_scratch;
    DlBackend backend;
    uint64_t revision;
    int ready;
} DlTrainTape;

/* Config: positive dimensions, V>=2, D%Hq==0, Hq%Hkv==0, even D/Hq;
 * finite epsilon/init_std > 0, finite theta > 1; checked parameter sizes.
 * Parameter count is 2+9*layers. count outputs unchanged on failure. */
DlStatus dl_model_config_validate(DlModelConfig config,
                                  size_t *parameter_count, size_t *value_count);
/* Strict 12-line name=value file in the documented order, C numeric locale.
 * No spaces/comments/extra lines; LF or CRLF, final newline required.
 * DL_ARGUMENT: syntax; DL_RANGE: numeric range; DL_STATE: file I/O.
 * All three outputs unchanged on failure and disjoint from one another. */
DlStatus dl_model_config_load(const char *path, DlModelConfig *config,
                              size_t *batch, size_t *time);

/* Constructors take initialized empty owners ({0}); never copy live owners
 * or mutate metadata. Failure leaves destination unchanged. Destroy tape
 * BEFORE its model; destroy is repeatable on an empty/partly built owner.
 * No concurrent calls, edits, or destruction. No hidden global state. */
DlStatus dl_model_create(DlModel *out, DlModelConfig config);
void dl_model_destroy(DlModel *model);
/* Float-arena budget excludes parameter arrays and small metadata records.
 * Positive B,T, positions 0..T-1 per sequence. Budget 0 is invalid. */
DlStatus dl_train_tape_create(DlTrainTape *out, const DlModel *model,
                             size_t batch, size_t time, size_t max_float_bytes);
void dl_train_tape_destroy(DlTrainTape *tape);

/* Accessors borrow the constructed model's arrays. Index must be in range.
 * Values may be read freely. Before any external value edit, call
 * dl_model_mark_changed and proceed ONLY on success. Never edit a weight
 * between matching forward/backward. Metadata must not be changed. */
DlMatrix dl_parameter_value(DlModel *model, size_t index);
DlMatrix dl_parameter_grad(DlModel *model, size_t index);
DlStatus dl_model_mark_changed(DlModel *model);
DlStatus dl_model_zero_grad(DlModel *model);

/* Dense full-sequence training; ids/targets each contain rows valid IDs.
 * Separate caller-owned input storage and separate double loss output.
 * Copies IDs/targets into tape: caller may reuse inputs after return.
 * Any new forward attempt invalidates the old ready state. Success commits
 * loss/logits and a ready tape; failure leaves loss unchanged, invalidates
 * all tape results, and leaves parameters/parameter gradients unchanged.
 * No gradient clearing, weight update, dropout, or dataset shifting here. */
DlStatus dl_model_forward(DlModel *model, DlTrainTape *tape, DlBackend backend,
                          const size_t *ids, const size_t *targets,
                          size_t count, double *loss);
/* Consume ONE matching ready tape. SET replaces every unique gradient;
 * ADD needs valid finite old gradients and adds one batch's mean derivative.
 * No extra scaling. On arithmetic failure all gradients are INVALID, even
 * earlier successful ADDs; zero/recompute from a known accumulation boundary.
 * Success leaves tape.embedding_grad as the final lookup-input derivative.
 * No second backward without another forward. */
DlStatus dl_model_backward(DlModel *model, DlTrainTape *tape, DlWriteMode mode);
/* Plain-gradient smoke update, NOT AdamW. Requires valid gradients, eta>0.
 * First checks EVERY candidate in double; then updates each unique value once.
 * Failure changes nothing. Success increments revision and invalidates grads
 * and hence any older forward tapes. Call zero_grad before new accumulation. */
DlStatus dl_model_sgd(DlModel *model, double eta);

#endif
