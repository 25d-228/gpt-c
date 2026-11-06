#include "dl/data.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int show(const char *label, const DlByteData *data, const DlByteBatch *batch)
{
    for (size_t row = 0; row < batch->batch; ++row) {
        size_t start = batch->starts[row];
        if (printf("%s row %zu start=%zu x=", label, row, start) < 0 ||
            dl_bytes_fprint(stdout, data->bytes + start, batch->time) != DL_OK ||
            printf(" y=") < 0 ||
            dl_bytes_fprint(stdout, data->bytes + start + 1, batch->time) != DL_OK ||
            putchar('\n') == EOF) return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s TRAIN.bin VALIDATION.bin\n", argv[0]);
        return EXIT_FAILURE;
    }
    DlByteData train = {0}, validation = {0};
    DlByteBatch batch = {0};
    DlRng rng = {42};
    DlStatus status = DL_OK;
    int result = EXIT_FAILURE, has_batch = 0;
    size_t cursor = 0;
#define TRY(call) do { status = (call); if (status != DL_OK) goto done; } while (0)
    TRY(dl_bytes_load(&train, argv[1], 16 * 1024 * 1024));
    TRY(dl_bytes_load(&validation, argv[2], 1024 * 1024));
    TRY(dl_byte_batch_create(&batch, 2, 4));
    TRY(dl_byte_batch_sample(&train, &batch, &rng));
    if (!show("train", &train, &batch)) goto done;
    TRY(dl_byte_batch_next(&validation, &batch, &cursor, &has_batch));
    if (!has_batch) {
        fprintf(stderr, "validation has no full batch of eight targets\n");
        goto done;
    }
    if (!show("validation", &validation, &batch)) goto done;
    if (printf("train RNG=%" PRIu64 "; validation cursor=%zu\n", rng.state, cursor) < 0 ||
        fflush(stdout) == EOF) goto done;
    result = EXIT_SUCCESS;
done:
    if (status != DL_OK) fprintf(stderr, "inspect_data: %s\n", dl_status_name(status));
    dl_byte_batch_destroy(&batch);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&train);
    return result;
#undef TRY
}
