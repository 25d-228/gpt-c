# Chapter 15 · Part 3 — Running the training application

This part connects the trainer and checkpoint routines to a complete command-line application. It also provides original text fixtures and four configurations, so the first run does not depend on a network download. Every forward pass, backward pass, gradient reduction, moment update, and loss calculation remains C code.

Continue from [Part 2](https://25d-228.github.io/gpt-c/chapter-15-part-2.html). This is the third part of the same **6.00-hour** chapter. [Part 4](https://25d-228.github.io/gpt-c/chapter-15-part-4.html) supplies the executable test files named by the Makefile below. Complete all four parts before running `make test`; no future chapter is required.

## Make stopping different from changing the schedule

The application takes ten arguments after its executable name:

```text
train new|resume CONFIG TRAIN VALID CHECKPOINT TOTAL STOP MICRO fixed|random ref|blocked
```

`TOTAL` is the intended number of updates in the schedule; `STOP` is the completed-update boundary at which this invocation exits. A run with `TOTAL=160, STOP=40`, followed by a resume with `TOTAL=160, STOP=160`, uses the same schedule as one uninterrupted 160-update run. Replacing the first `TOTAL` with 40 would change that schedule and is rejected as incompatible when resumed as a 160-update run.

`MICRO` is the number of microbatches per update, not their batch size. `B` and `T` come from the configuration file. Fixed mode repeats the first full training batch; random mode samples windows with replacement. Backend selection is explicit and stored in the checkpoint. Do not change it mid-continuation and expect an identical arithmetic path.

The application uses AdamW with beta values 0.9 and 0.999, epsilon $10^{-8}$, decoupled decay 0.01, and clipping threshold 1.0. The learning-rate peak is 0.003 and the decay endpoint is 0.0003. Warmup is integer `TOTAL / 10`; consequently an eight-update pilot has no warmup, whereas a 160-update pilot warms up for sixteen updates. These are specified experiment settings, not universal recommendations.

Create `app/train.c`. The decimal parser rejects signs, empty strings, zero, and overflow rather than accepting a partial conversion. `PRIu64` below comes from `<inttypes.h>`: adjacent C string literals concatenate, allowing `"update=%" PRIu64` to use the correct format for `uint64_t` without assuming that it is `unsigned long` on every platform.

```c title="app/train.c"
#define _POSIX_C_SOURCE 200809L
#include "dl/checkpoint.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

static int positive(const char *text, uint64_t *out)
{
    uint64_t value = 0;
    if (*text == '\0') return 0;
    for (const char *p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    if (value == 0) return 0;
    *out = value;
    return 1;
}

static double seconds_between(struct timespec begin, struct timespec end)
{
    return (double)(end.tv_sec - begin.tv_sec) +
           1e-9 * (double)(end.tv_nsec - begin.tv_nsec);
}
```

## Construct owners before borrowing them

The application keeps its model at one stack address for the whole run. The optimizer and both tapes borrow that address. Every owner starts empty, and every error goes through one cleanup label, so a failure halfway through construction uses the same cleanup path as a normal exit.

The `TRY` macro prints the failed operation and the shared status name before jumping to cleanup. The existing `DL_STATE` status is used for both lifecycle and file I/O failures; the operation name is therefore important context. This small application is not introducing a new error-reporting framework.

The next three blocks continue the **same `main` function** in the same file. Keep their scopes in order; no code is omitted between them.

```c title="app/train.c"
#define TRY(call) do { status = (call); if (status != DL_OK) { \
    fprintf(stderr, "%s: %s\n", #call, dl_status_name(status)); goto done; \
} } while (0)

int main(int argc, char **argv)
{
    if (argc != 11 || (strcmp(argv[1], "new") != 0 && strcmp(argv[1], "resume") != 0) ||
        (strcmp(argv[9], "fixed") != 0 && strcmp(argv[9], "random") != 0) ||
        (strcmp(argv[10], "ref") != 0 && strcmp(argv[10], "blocked") != 0)) {
        fprintf(stderr, "usage: %s new|resume CONFIG TRAIN VALID CHECKPOINT "
                        "TOTAL STOP MICRO fixed|random ref|blocked\n", argv[0]);
        return EXIT_FAILURE;
    }
    uint64_t total = 0, stop = 0, micro = 0;
    if (!positive(argv[6], &total) || !positive(argv[7], &stop) ||
        !positive(argv[8], &micro) || total > DL_OPT_MAX_COUNT ||
        stop > total || micro > SIZE_MAX) {
        fprintf(stderr, "invalid update or microbatch count\n");
        return EXIT_FAILURE;
    }
    DlStatus status = DL_OK;
    int result = EXIT_FAILURE;
    DlModelConfig config = {0};
    DlModel model = {0};
    DlAdamW optim = {0};
    DlTrainTape tape = {0}, evaluation = {0};
    DlByteData training = {0}, validation = {0};
    DlByteBatch batch = {0}, eval_batch = {0};
    DlTrainState state = {.schedule = {total, total / 10, 0.003, 0.0003},
        .rng = {2026}, .microbatches = (size_t)micro,
        .backend = strcmp(argv[10], "ref") == 0 ? DL_BACKEND_REF : DL_BACKEND_BLOCKED,
        .fixed = strcmp(argv[9], "fixed") == 0};
    const size_t mib = (size_t)1024 * 1024;
    const uint64_t file_budget = UINT64_C(512) * 1024 * 1024;
```

The configured caps bound payloads rather than the whole process. The loader reserves its requested capacity even when the file is shorter. We check the model's weight-plus-gradient payload before allocating it, then let each optimizer/tape constructor enforce its own checked allocation budget.

`new` refuses an existing checkpoint path. That is a guard against accidentally restarting over a useful run, not a multi-writer locking protocol. The checkpoint directory and input files must remain stable and under the caller's control. `resume` creates the expected model/optimizer geometry and then loads into those existing owners; it does not construct a second model and move it over the first.

Immediately after constructing the optimizer, call `dl_adamw_abort`. This consumes the model constructor's zero-gradient state as an inactive training boundary without changing any weights. Otherwise the trainer would correctly reject the model's still-valid, unconsumed gradient state.

```c title="app/train.c"
    if (strcmp(argv[1], "new") == 0) {
        struct stat metadata;
        if (lstat(argv[5], &metadata) == 0) {
            fprintf(stderr, "new refuses an existing checkpoint; use a new path or resume\n");
            goto done;
        }
        if (errno != ENOENT) { perror("checkpoint path"); goto done; }
    }
    TRY(dl_model_config_load(argv[2], &config, &state.batch, &state.time));
    if (config.vocab != 256) { fprintf(stderr, "byte training needs vocab=256\n"); goto done; }
    TRY(dl_bytes_load(&training, argv[3], 32 * mib));
    TRY(dl_bytes_load(&validation, argv[4], 8 * mib));
    size_t parameters = 0, values = 0, model_bytes = 0;
    TRY(dl_model_config_validate(config, &parameters, &values));
    TRY(dl_size_mul(values, 2 * sizeof(float), &model_bytes));
    if (model_bytes > 128 * mib) { fprintf(stderr, "model exceeds parameter budget\n"); goto done; }
    TRY(dl_model_create(&model, config));
    TRY(dl_adamw_create(&optim, &model, (DlAdamWConfig){0.9, 0.999, 1e-8, 0.01, 1.0},
                        256 * mib));
    TRY(dl_adamw_abort(&optim)); /* Consume constructor-zero gradients as a boundary. */
    TRY(dl_train_tape_create(&tape, &model, state.batch, state.time, 512 * mib));
    TRY(dl_train_tape_create(&evaluation, &model, state.batch, state.time, 512 * mib));
    TRY(dl_byte_batch_create(&batch, state.batch, state.time));
    TRY(dl_byte_batch_create(&eval_batch, state.batch, state.time));
    TRY(dl_train_boundary(&optim, &state, &training, &validation));
    if (strcmp(argv[1], "resume") == 0)
        TRY(dl_checkpoint_load(argv[5], &optim, &state, &training, &validation, file_budget));
    if (optim.updates > stop) { fprintf(stderr, "STOP precedes restored update\n"); goto done; }
    printf("parameters=%zu B=%zu T=%zu micro=%zu backend=%s mode=%s\n",
           model.value_count, state.batch, state.time, state.microbatches, argv[10], argv[9]);
    printf("payload_bytes model=%zu optimizer=%zu train_tape=%zu eval_tape=%zu data_capacity=%zu\n",
           model_bytes, optim.count * (2 * sizeof(float) + sizeof(double)),
           tape.arena.count * sizeof(float), evaluation.arena.count * sizeof(float),
           training.capacity + validation.capacity);
    DlEvalStats evaluation_stats = {0};
    TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch, 4,
                     &evaluation_stats));
    printf("validation before update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
           optim.updates, evaluation_stats.loss, evaluation_stats.tokens, evaluation_stats.unscored);
```

`payload_bytes` reports allocation payload counts from the constructed objects. It is not a process resident-set measurement. The `model` figure includes weights and gradients; the optimizer figure includes both float moments and the double accumulator. Tape figures include their float arenas, but not all metadata or allocator overhead.

Validation uses at most four full batches. The printed `scored` and `unscored` fields expose the scope of that prefix measurement. The initial validation call lets us compare the same prefix before and after training without using its gradient.

## Time an entire update, then save at a boundary

The timer surrounds only `dl_train_update`: data sampling/copying, all microbatch forward/backward work, gradient accumulation, clipping, and the optimizer update are included. Construction, validation, printing, and checkpoint saving are outside that interval. Keep that scope when comparing measurements; this is not a timing of only the matrix kernel.

After every tenth update and at the requested stopping boundary, the application evaluates and saves. A failed save ends this invocation; it does not pretend a disk checkpoint exists for the newest in-memory update. A subsequent invocation can resume the last successful checkpoint and replay later work. There are no automatic retries or signal-handler checkpoint writes.

```c title="app/train.c"
    while (optim.updates < stop) {
        struct timespec begin = {0}, end = {0};
        if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) goto done;
        DlTrainStats stats = {0};
        TRY(dl_train_update(&optim, &state, &training, &batch, &tape, &stats));
        if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) goto done;
        printf("update=%" PRIu64 " train_loss=%.9f targets=%" PRIu64
               " lr=%.6g norm=%.6g clip=%.6g seconds=%.6f\n",
               optim.updates, stats.loss, stats.optim.tokens, stats.optim.learning_rate,
               stats.optim.grad_norm, stats.optim.clip_scale, seconds_between(begin, end));
        if (fflush(stdout) != 0) goto done;
        if (optim.updates % 10 == 0 || optim.updates == stop) {
            TRY(dl_evaluate(&model, &evaluation, state.backend, &validation, &eval_batch,
                             4, &evaluation_stats));
            printf("validation update=%" PRIu64 " loss=%.9f scored=%" PRIu64 " unscored=%zu\n",
                   optim.updates, evaluation_stats.loss, evaluation_stats.tokens,
                   evaluation_stats.unscored);
            TRY(dl_checkpoint_save(argv[5], &optim, &state, &training, &validation, file_budget));
        }
    }
    if (fflush(stdout) != 0) goto done;
    result = EXIT_SUCCESS;
done:
    if (result != EXIT_SUCCESS)
        fprintf(stderr, "training stopped; use the last completed checkpoint, if any\n");
    dl_byte_batch_destroy(&eval_batch);
    dl_byte_batch_destroy(&batch);
    dl_train_tape_destroy(&evaluation);
    dl_train_tape_destroy(&tape);
    dl_adamw_destroy(&optim);
    dl_model_destroy(&model);
    dl_bytes_destroy(&validation);
    dl_bytes_destroy(&training);
    return result;
}

#undef TRY
```

Cleanup destroys both tapes and the optimizer before the borrowed model. The dataset and batch buffers are independent owners. A final output-stream failure can make the process exit nonzero even after a checkpoint was saved; examine the reported operation and the checkpoint rather than assuming that every nonzero exit means the last file was lost.

A restored run already at `STOP` performs validation and exits without another update. `STOP` less than the restored completed-update count is an error. The checkpoint's total schedule is not extended implicitly.

## Four configurations, one decoder

All configuration files use the strict twelve-line format introduced in Chapter 12. The final newline is required; do not add comments or reorder fields. Model seed 42 initializes weights. The separately stored training generator starts at 2026, so changing model construction does not silently consume data draws.

The small diagnostic configuration is large enough to use actual attention and SwiGLU, but short enough for repeated executable tests:

```ini title="configs/byte-check.conf"
vocab=256
layers=1
width=16
query_heads=4
kv_heads=2
ffn_width=40
batch=1
time=8
epsilon=0.00001
rope_theta=10000
init_std=0.125
seed=42
```

For the bounded text pilot, use two blocks, width 32, and two length-16 sequences per microbatch. This configuration has **29,856 unique scalar parameters**. It exercises the same manually differentiated mechanisms as the larger model.

```ini title="configs/byte-pilot.conf"
vocab=256
layers=2
width=32
query_heads=4
kv_heads=2
ffn_width=80
batch=2
time=16
epsilon=0.00001
rope_theta=10000
init_std=0.125
seed=42
```

The planned byte model has four blocks, width 256, eight query heads sharing two key/value heads, and feed-forward width 704. Its unique parameter count is **2,885,888**. `B=2,T=128` with `MICRO=4` gives 1024 scored targets per update.

```ini title="configs/byte-small.conf"
vocab=256
layers=4
width=256
query_heads=8
kv_heads=2
ffn_width=704
batch=2
time=128
epsilon=0.00001
rope_theta=10000
init_std=0.02
seed=42
```

A short-context probe keeps **all the same model dimensions and initialization settings** while reducing the batch to one eight-target sequence. It tests full parameter ownership and optimizer state without first paying for the larger context. It is a different batching experiment, not a checkpoint-compatible continuation of the previous configuration.

```ini title="configs/byte-small-probe.conf"
vocab=256
layers=4
width=256
query_heads=8
kv_heads=2
ffn_width=704
batch=1
time=8
epsilon=0.00001
rope_theta=10000
init_std=0.02
seed=42
```

The count follows the established directory formula. With width $D$, feed-forward width $F$, layer count $L$, vocabulary $V$, and shared-key width $K=H_{kv}(D/H_q)$,

$$
P=VD+L\left(2D^2+2KD+3FD+2D\right)+D.
$$

The three feed-forward matrices contribute $3FD$; attention contributes two full-width projections and two shared-key/value projections. The embedding/output table contributes $VD$ once, not twice.

## A known local text source

Create `scripts/make_training_fixture.py`. It writes distinct training and validation sentences with deliberately similar vocabulary. These texts are **original diagnostic fixtures**, not a downloaded corpus, not TinyStories, and not a representative language-quality benchmark. Their modest size makes the exact training inputs inspectable.

The script performs asset preparation only. UTF-8 encoding produces raw bytes; it does not run any tokenizer model, tensor computation, or optimizer. It refuses to reuse an existing output directory. Unlike Chapter 13's more general staged preparer, this short local-fixture script can leave its newly created directory partially populated after a disk error; remove that failed fixture directory before rerunning.

```python title="scripts/make_training_fixture.py"
#!/usr/bin/env python3
"""Create separate, author-written diagnostic texts; no download or model code."""
from __future__ import annotations

import argparse
from pathlib import Path

TRAIN = """A red fox sat beside the gate. The gate was open and the fox went in.
A blue bird flew over the tree. The bird saw a small seed on the path.
Mina put a cup on the table. She filled the cup with cool water.
The sun came out after the rain. The wet leaves shone in the light.
A small dog ran to the pond. It stopped at the edge and looked at a frog.
The frog jumped into the water. The dog sat down and waited.
Ravi found a green stone near the hill. He put it back beside the path.
A child opened a book in the quiet room. The book told a story about a boat.
The boat moved slowly on the river. Two ducks followed it under a bridge.
At night the family closed the gate. The fox slept under a tall tree.
In the morning a bell rang. Mina took her bag and walked to the garden.
She saw the blue bird again. This time it carried a leaf in its beak.
The old cat watched from the wall. It stretched its legs and went to sleep.
Ravi made a little boat from paper. He placed it in a bowl of water.
The paper boat floated for a while. Then Ravi lifted it out to dry.
A warm wind moved the branches. The leaves made a soft sound above the path.
The child read one more page. Then she closed the book and turned off the lamp.
"""

VALID = """A little fox found a bowl beside the wall. It drank some water and walked away.
Mina sat under the tree with a book. A green leaf fell onto the open page.
The dog watched two birds near the pond. They flew over the gate into the garden.
Ravi took his paper boat to the table. He told a story about a long river.
When the rain stopped, the child opened the door. The sun shone on the wet path.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    for name, text in (("train.bin", TRAIN), ("valid.bin", VALID)):
        payload = text.encode("utf-8")
        (args.directory / name).write_bytes(payload)
        print(f"{name}: {len(payload)} bytes")
    (args.directory / "SOURCE.txt").write_text(
        "Author-written diagnostic fixture. Training and validation contain distinct "
        "sentences but intentionally similar vocabulary. Not TinyStories or a "
        "representative language benchmark. UTF-8, LF, no normalization.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

The files contain 1214 and 393 bytes respectively. The validation sentences are not training records, but shared grammar and vocabulary make this an easy diagnostic distribution. A falling score here shows that the actual training path changes predictive behavior; it does not establish broad language ability.

For a larger prepared corpus, reuse Chapter 13's bounded preparation program and its recorded provenance. Do not silently replace a failed remote download with these local sentences. Use a fresh checkpoint path for the new data. Changing the exact bytes, even with an unchanged file length, is rejected by this checkpoint format.

## Build and run the fixed-batch check first

Replace the Makefile with the following complete version. It includes the new trainer/checkpoint modules in `CORE`, adds the two C tests from Part 4, and builds `train` in all three configurations. `test-training-cli` runs Python only as an executable/file comparison harness.

```makefile title="Makefile"
CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2 -fno-lto
LDLIBS = -lm
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/rng.c src/model.c src/data.c src/optim.c src/train.c src/checkpoint.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/rng.h include/dl/model.h include/dl/data.h include/dl/optim.h include/dl/train.h include/dl/checkpoint.h tests/tiny.h tests/training_fixture.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim test_training test_checkpoint

.PHONY: all debug sanitize release test test-assets test-training-cli clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data build/debug/train
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data build/sanitize/train
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data build/release/train

build/debug/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/%: tests/%.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/compare: app/compare.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/debug/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/inspect_data: app/inspect_data.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test: debug sanitize release
	@set -e; for mode in debug sanitize release; do \
	    for name in $(PROGRAMS); do ./build/$$mode/$$name; done; \
	    ./build/$$mode/compare kernel both 9 33 65 1 nt; \
	done

test-assets:
	python3 tests/test_prepare.py

clean:
	rm -rf build

build/debug/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/train: app/train.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-training-cli: build/release/train
	python3 tests/test_train_cli.py
```

On the native Mac, from the implementation directory:

```sh
make build/debug/test_training build/sanitize/test_training build/release/test_training
./build/debug/test_training
./build/sanitize/test_training
./build/release/test_training
make build/release/train
mkdir -p runs
python3 scripts/make_training_fixture.py data/training-fixture
```

A compatible compiler on another platform can be selected with `CC=clang`. The source requires C17, the math library, and the POSIX file/time facilities used explicitly in the listings. No exact compiler release or invented operating-system requirement is imposed.

Part 4's fixed-batch test uses a still smaller 256-token-vocabulary model and performs 120 real optimizer updates on the same eight targets. The observed debug, sanitizer, and optimized runs with Clang 17 on x86_64 Linux all reported:

```text
overfit ref: 5.692190972 -> 0.078423635 on 8 fixed targets after 120 updates
overfit blocked: 5.692190972 -> 0.078423635 on 8 fixed targets after 120 updates
```

The test's criterion is a large reduction relative to the initial loss, not those exact decimals on every machine. It verifies unchanged random state and the exact processed-target count as well. This is memorization of one fixed batch, deliberately not held-out validation.

For a fixed-batch application experiment rather than the unit test, use:

```sh
./build/release/train new configs/byte-check.conf \
  data/training-fixture/train.bin data/training-fixture/valid.bin \
  runs/fixed.bin 160 160 1 fixed blocked
```

That application command is a runnable variation, not the source of the eight-target test numbers above: it uses a different width and learning-rate schedule. In fixed mode the printed validation still comes from the separate validation file. A low fixed training loss need not imply a good validation score.

## Run a text pilot and interrupt at a clean boundary

Use a single intended 160-update schedule, first stopping after ten updates:

```sh
./build/release/train new configs/byte-pilot.conf \
  data/training-fixture/train.bin data/training-fixture/valid.bin \
  runs/pilot.bin 160 10 2 random blocked
./build/release/train resume configs/byte-pilot.conf \
  data/training-fixture/train.bin data/training-fixture/valid.bin \
  runs/pilot.bin 160 160 2 random blocked
```

These commands were executed with Clang 17's optimized build on x86_64 Linux. They process 64 targets per update and 10,240 target uses altogether. The fixed validation prefix consists of four $2\cdot16$-target batches, or **128 scored targets**. Of the validation file's 392 possible next-byte targets, **264 are unscored** under this limit.

| Completed updates | Validation-prefix mean loss, nats per byte target |
| --- | ---: |
| 0 | 5.725389095 |
| 10, before stopping | 4.831238939 |
| 160, after resuming | 2.270204153 |

The scalar loss uses natural logarithms, as in Chapter 5. An exactly uniform 256-way prediction would have loss $\log256\approx5.54518$; the random initialized model is not exactly uniform. These measurements are a bounded text-training demonstration, not a TinyStories result or a claim about a native M4 run.

A single printed training loss can rise from one update to the next because random windows change. For example, the final pilot update's pre-update training mean was 2.317264589, while the fixed validation-prefix score after that update was 2.270204153. Those numbers use different examples and different parameter states; there is no contradiction to diagnose merely because they differ.

To compare a resumed application run against an uninterrupted one, Part 4 provides a complete executable-level test. It compares every non-timing update log field and the final checkpoint bytes, not just a similar loss curve.

## Probe the full parameter count, then the intended batching

The following short-context command was also executed on x86_64 Linux:

```sh
./build/release/train new configs/byte-small-probe.conf \
  data/training-fixture/train.bin data/training-fixture/valid.bin \
  runs/small-probe.bin 8 8 1 random blocked
```

It trains all 2,885,888 parameters for eight updates, scoring eight targets per update. The 32-target validation prefix moved from 5.481445505 to 3.985547996. That probe is too small to assess model quality, but it verifies that the complete parameter and optimizer allocations participate in real training rather than only in a construction test.

Then run the intended batch/context setting:

```sh
./build/release/train new configs/byte-small.conf \
  data/training-fixture/train.bin data/training-fixture/valid.bin \
  runs/small.bin 8 8 4 random blocked
```

This command was executed too: eight updates, **1024 targets per update**, **8192 target uses**. The 393-byte validation file supplies only one full 256-target batch at this geometry. The application therefore scores 256 targets and explicitly reports the other 136 as unscored.

| Observation | Measured value in the Linux run |
| --- | ---: |
| Validation prefix before training | 5.474827494 nats/target |
| Validation prefix after update 8 | 3.006250581 nats/target |
| Update 1 training mean, before its step | 5.544080418 |
| Update 8 training mean, before its step | 3.002654058 |
| Median whole-update interval, eight samples | 2.940115 seconds |
| Smallest / largest update interval | 2.801424 / 3.225819 seconds |

The timing is only an observation from this container, with the complete update scope described above. It is not an M4 throughput prediction, a kernel-only result, or a machine comparison. The corpus is intentionally tiny and repeatedly sampled. Much of such an early loss reduction can come from learning easy byte-frequency and short-context regularities; do not label it general language competence.

## Account for payloads before choosing a longer run

The full configuration printed the following logical payload counts from its constructed objects:

```text
payload_bytes model=23087104 optimizer=46174208 train_tape=34603008 eval_tape=34603008 data_capacity=41943040
```

These correspond to 22.0176 MiB for weights plus gradients, 44.0352 MiB for optimizer arrays, 33 MiB for each float tape arena, and 40 MiB of reserved data capacity. Together they are about **172.053 MiB**. Loading temporarily adds about **33.026 MiB** of candidate values, bringing those counted payloads to about **205.079 MiB**. Metadata, runtime/library buffers, allocator overhead, code pages, and the operating system are excluded. No resident-memory measurement is being claimed.

The data loaders reserve their configured 32 MiB and 8 MiB caps even for this small fixture. That is why the printed capacity is not 1607 bytes. The caps are conservative, explicit implementation choices, not an estimate of the text's actual size.

For the 16 GB M4 target, this chosen model does not need GPU offload merely to fit these payloads. Runtime is the practical uncertainty. Measure a short **native** pilot on the intended data, then estimate a longer run from your own median complete-update interval, adding validation and checkpoint intervals separately. The single-threaded, checked reference arithmetic is not a tuned accelerator implementation. A large amount of free memory does not guarantee fast training.

To score more validation data, change the two application calls' batch limit from four to zero for all full batches, or to a larger positive number for a longer fixed prefix. Keep evaluation geometry and coverage fixed when comparing losses. That change does not alter update state, but it changes what the reported metric measures. Part 4's isolation test makes that separation executable.

Continue with [Part 4 — Checking continuation and failure recovery](https://25d-228.github.io/gpt-c/chapter-15-part-4.html), including the complete test sources and controlled wrong implementations.
