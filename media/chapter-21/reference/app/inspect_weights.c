#include "dl/checkpoint.h"
#include "dl/sample.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s CONFIG WEIGHTS [TOKEN_ID]\n", argv[0]);
        return 2;
    }
    int result = 1;
    DlModelConfig config = {0};
    size_t batch = 0, time = 0;
    DlWeights weights = {0};
    DlInference workspace = {0};
    DlTensor logits = {0};
    DlStatus status = dl_model_config_load(argv[1], &config, &batch, &time);
    if (status != DL_OK) goto done;
    status = dl_checkpoint_load_weights(argv[2], &weights, config,
        UINT64_C(2) * 1024 * 1024 * 1024, (size_t)1024 * 1024 * 1024);
    if (status != DL_OK) goto done;
    printf("tensors=%zu values=%zu weight_payload_bytes=%zu position_limit=%zu\n",
           weights.param_count, weights.arena.count,
           weights.arena.count * sizeof(float), weights.position_limit);
    printf("directory_bytes=%zu gradients=0 moments=0 loader_chunk_bytes=4096\n",
           weights.param_count * sizeof(DlWeight));
    if (argc == 4) {
        if (argv[3][0] < '0' || argv[3][0] > '9') { status = DL_ARGUMENT; goto done; }
        char *end = NULL;
        errno = 0;
        uintmax_t parsed = strtoumax(argv[3], &end, 10);
        if (errno != 0 || *end != '\0' || parsed >= config.vocab) {
            status = DL_RANGE; goto done;
        }
        size_t id = (size_t)parsed;
        status = dl_inference_create_tiled(&workspace, &weights, 1, 1, 1,
                                            16 * 1024 * 1024);
        if (status != DL_OK) goto done;
        status = dl_tensor_create(&logits, 1, &config.vocab);
        if (status != DL_OK) goto done;
        DlMatrix row = {logits.data, logits.count, 0, 1, config.vocab, config.vocab};
        status = dl_inference_forward(&workspace, DL_BACKEND_BLOCKED, &id, 1, 0, row);
        if (status != DL_OK) goto done;
        size_t selected = 0;
        status = dl_greedy(logits.data, config.vocab, &selected);
        if (status != DL_OK) goto done;
        printf("input_id=%zu next_argmax_id=%zu selected_logit=%.9g\n",
               id, selected, (double)logits.data[selected]);
        puts("Finite C execution only; not a tokenizer or independent numerical comparison.");
    }
    result = 0;
done:
    if (status != DL_OK) fprintf(stderr, "inspect_weights: %s\n", dl_status_name(status));
    dl_tensor_destroy(&logits);
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    return result;
}
