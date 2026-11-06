# Chapter 19 — Whole-model, training, and memory checks

This concludes the chapter begun with [tiled backward](https://25d-228.github.io/gpt-c/chapter-19.html). The operator checks are necessary, but they do not prove that a multi-block training tape keeps the right statistics alive or that a checkpoint preserves the new execution policy.

## Reuse a named, complete fixture

Add `tests/test_tiled_training.c`. It includes the unchanged [Chapter 15 training fixture](https://25d-228.github.io/gpt-c/media/chapter-19/reference/tests/training_fixture.h), also reproduced and explained in [Chapter 15, Part 4](https://25d-228.github.io/gpt-c/chapter-15-part-4.html). That header supplies the complete `TrainingFixture` owner, its create/destroy functions, assertion macros, and exact persistent-state comparison. It is part of both the prerequisite and current companion snapshots, not a helper the reader must invent.

The new test creates independent models from the same explicit seed. Each has its own weights, gradients, optimizer, tapes, batches, and datasets. Matching model values do not permit sharing mutable scratch between the two executions.

### Keep counts separate from correctness assertions

The maximum disagreement variables are diagnostic summaries. Actual acceptance uses explicit finite comparisons at every checked coordinate. A printed small maximum is not a substitute for visiting all parameters.

```c title="tests/test_tiled_training.c"
#define _POSIX_C_SOURCE 200809L
#include "training_fixture.h"

#include <unistd.h>

static size_t model_differences = 0;
static double gradient_error = 0.0, update_error = 0.0;

static int close_value(double a, double b, double tolerance)
{
    return isfinite(a) && isfinite(b) && fabs(a-b) <= tolerance*(1.0+fabs(b));
}
```

## Check the complete decoder and the allocation formula

Use one- and two-block models, all three sharing ratios, and both matrix backends. The dimensions are small enough for selected coordinate differences but large enough to exercise both residual paths, final normalization, and both uses of the tied embedding/output table.

First compare the actual arena count against Part 2's linear expression, and verify that all score/probability views are empty. Construct a second tape at twice the sequence length. If $C$ is the tile-only constant in **bytes**, the expected relation is $P(2T)+C=2P(T)$, because only the sequence-dependent part doubles. Then check a budget equal to the exact payload succeeds and a budget one byte smaller fails without publishing a partial owner.

After comparing logits, every parameter gradient, and the embedding-input gradient, perturb the first and last coordinate of **each unique parameter**. This includes all block scales and projections; it does not pretend to check every large tensor coordinate. Across the six configurations this produces 204 numerical differences.

Finally corrupt only block zero's scaled sum after a successful forward. In two-block models, block one's reverse work has already run when the error is reached. The test requires global gradient invalidation and consumed-tape state, then checks fresh SET recovery and another ADD accumulation.

```c title="tests/test_tiled_training.c"
static int model_case(size_t layers, size_t heads, DlBackend backend)
{
    int result=0;
    DlModel full={0}, tiled={0};
    DlTrainTape a={0}, b={0}, doubled={0}, exact={0}, rejected={0};
    DlModelConfig c={.vocab=17,.layers=layers,.width=16,.query_heads=4,
        .kv_heads=heads,.ffn_width=20,.epsilon=1e-5f,.init_std=0.125f,
        .rope_theta=10000,.seed=42};
    enum { B=2, T=5, M=B*T, R=3, S=2 };
    size_t ids[M],targets[M];
    for (size_t i=0;i<M;++i) { ids[i]=(i*3+2)%17; targets[i]=(i*5+1)%17; }
    OK(dl_model_create(&full,c));
    OK(dl_model_create(&tiled,c));
    OK(dl_train_tape_create(&a,&full,B,T,1024*1024));
    OK(dl_train_tape_create_tiled(&b,&tiled,B,T,R,S,1024*1024));
    OK(dl_train_tape_create_tiled(&doubled,&tiled,B,2*T,R,S,1024*1024));
    size_t bytes=0,twice=0,reference_bytes=0;
    OK(dl_train_tape_payload_bytes(&b,&bytes));
    OK(dl_train_tape_payload_bytes(&doubled,&twice));
    OK(dl_train_tape_payload_bytes(&a,&reference_bytes));
    size_t linear=(layers+1)*M*(9*16+3*heads*4+3*20)+2*M*17;
    size_t constant=layers*(R*S+R*4+2*R)+(2*R*S+R*4+2*S*4+R);
    CHECK(b.arena.count==linear);
    CHECK(a.arena.count==linear+2*(layers+1)*B*4*T*T);
    CHECK(bytes==linear*sizeof(float)+(3*layers*M*4+constant)*sizeof(double));
    CHECK(twice+constant*sizeof(double)==2*bytes);
    OK(dl_train_tape_create_tiled(&exact,&tiled,B,T,R,S,bytes));
    STATUS(dl_train_tape_create_tiled(&rejected,&tiled,B,T,R,S,bytes-1),DL_RANGE);
    CHECK(rejected.model==NULL && rejected.arena.data==NULL && rejected.tiled==NULL);
    STATUS(dl_train_tape_create_tiled(&rejected,&tiled,B,T,SIZE_MAX,S,SIZE_MAX),DL_OVERFLOW);
    CHECK(rejected.model==NULL && rejected.tiled==NULL);
    for (size_t l=0;l<layers;++l) {
        CHECK(b.blocks[l].attention.scores.base==NULL && b.blocks[l].attention.probs.base==NULL);
        if (l!=0) CHECK(b.tiled[l].data!=b.tiled[l-1].data);
    }
    CHECK(b.attention_scratch.scores.base==NULL && b.attention_scratch.probs.base==NULL);
    double fa=0,fb=0;
    OK(dl_model_forward(&full,&a,backend,ids,targets,M,&fa));
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    CHECK(close_value(fa,fb,2e-6));
    for (size_t i=0;i<M*17;++i)
        CHECK(close_value(a.logits.base[a.logits.offset+i],b.logits.base[b.logits.offset+i],3e-6));
    OK(dl_model_backward(&full,&a,DL_SET));
    OK(dl_model_backward(&tiled,&b,DL_SET));
    for (size_t p=0;p<full.param_count;++p)
        for (size_t i=0;i<full.params[p].grad.count;++i) {
            double x=full.params[p].grad.data[i],y=tiled.params[p].grad.data[i];
            gradient_error=fmax(gradient_error,fabs(x-y));
            CHECK(close_value(y,x,7e-6));
        }
    for (size_t i=0;i<M*16;++i)
        CHECK(close_value(a.embedding_grad.base[a.embedding_grad.offset+i],
                          b.embedding_grad.base[b.embedding_grad.offset+i],7e-6));
    STATUS(dl_model_backward(&tiled,&b,DL_SET),DL_STATE);
    /* Two positions in EACH unique parameter, including the tied table and
     * every block norm/projection. Preserve analytic values before changing. */
    for (size_t p=0;p<tiled.param_count;++p) {
        size_t coordinates[]={0,tiled.params[p].value.count-1};
        for (size_t j=0;j<2;++j) {
            size_t i=coordinates[j];
            float *v=&tiled.params[p].value.data[i],old=*v;
            float plus=old+0.002f,minus=old-0.002f;
            double analytic=tiled.params[p].grad.data[i],losses[2]={0};
            for (size_t side=0;side<2;++side) {
                OK(dl_model_mark_changed(&tiled));
                *v=side==0?plus:minus;
                OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&losses[side]));
            }
            OK(dl_model_mark_changed(&tiled)); *v=old;
            double numeric=(losses[0]-losses[1])/((double)plus-(double)minus);
            CHECK(close_value(numeric,analytic,4e-4));
            ++model_differences;
        }
    }
    /* Corrupt only block0's statistic: earlier block1 reverse work can have
     * written gradients. The model must invalidate the WHOLE result. */
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    b.tiled[0].data[b.tiled[0].row_capacity]=0;
    STATUS(dl_model_backward(&tiled,&b,DL_SET),DL_NUMERIC);
    CHECK(!tiled.gradients_valid && !b.ready);
    STATUS(dl_model_sgd(&tiled,0.01),DL_STATE);
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    OK(dl_model_backward(&tiled,&b,DL_SET));
    OK(dl_model_forward(&full,&a,backend,ids,targets,M,&fa));
    OK(dl_model_backward(&full,&a,DL_ADD));
    OK(dl_model_forward(&tiled,&b,backend,ids,targets,M,&fb));
    OK(dl_model_backward(&tiled,&b,DL_ADD));
    for (size_t p=0;p<full.param_count;++p)
        for (size_t i=0;i<full.params[p].grad.count;++i)
            CHECK(close_value(full.params[p].grad.data[i],tiled.params[p].grad.data[i],1.4e-5));
    printf("tiled model: L=%zu Hkv=%zu backend=%d full=%zu tiled=%zu doubled=%zu bytes\n",
           layers,heads,(int)backend,reference_bytes,bytes,twice);
    result=1;
done:
    dl_train_tape_destroy(&rejected); dl_train_tape_destroy(&exact);
    dl_train_tape_destroy(&doubled); dl_train_tape_destroy(&b); dl_train_tape_destroy(&a);
    dl_model_destroy(&tiled); dl_model_destroy(&full);
    return result;
}
```

### Change strategy only by reconstructing owners

The fixture initially creates the legacy materialized tapes. Destroy them before creating tiled ones; do not edit a live tape's mode while its memory still has another layout. The helper keeps the model/optimizer attached and changes only empty tape owners and the matching scalar run policy. Partial construction remains safe for the fixture destructor.

```c title="tests/test_tiled_training.c"
static DlStatus use_tiled(TrainingFixture *f, size_t qr, size_t kc)
{
    dl_train_tape_destroy(&f->tape);
    dl_train_tape_destroy(&f->evaluation);
    f->state.attention=DL_ATTENTION_TILED;
    f->state.query_tile=qr; f->state.key_tile=kc;
    DlStatus s=dl_train_tape_create_tiled(&f->tape,&f->model,1,8,qr,kc,1024*1024);
    if (s!=DL_OK) return s;
    return dl_train_tape_create_tiled(&f->evaluation,&f->model,1,8,qr,kc,1024*1024);
}
```

### Compare updates, not just gradients

Full and tiled runs must consume the same data RNG and target counts. Their weights and both optimizer moment arrays are compared with stated tolerances, because the attention reduction boundaries differ. Exact equality is reserved for two executions that use the same saved strategy, capacities, backend, inputs, and continuation state.

```c title="tests/test_tiled_training.c"
static int persistent_close(const TrainingFixture *a, const TrainingFixture *b)
{
    if (a->state.rng.state!=b->state.rng.state || a->state.tokens_seen!=b->state.tokens_seen ||
        a->optim.updates!=b->optim.updates) return 0;
    for (size_t p=0;p<a->model.param_count;++p)
        for (size_t i=0;i<a->model.params[p].value.count;++i) {
            double x=a->model.params[p].value.data[i],y=b->model.params[p].value.data[i];
            update_error=fmax(update_error,fabs(x-y));
            if (!close_value(x,y,3e-5)) return 0;
        }
    for (size_t i=0;i<2*a->optim.count;++i)
        if (!close_value(a->optim.moments[i],b->optim.moments[i],3e-6)) return 0;
    return 1;
}
```

## Check seven updates and an exact continuation boundary

For each backend and fixed/random mode, run four matched updates, save the tiled state, and load it into a separately constructed matching fixture. Wrong strategy and wrong tile capacity must reject without changing live state. The weight-only loader must copy the exact trained weights without requiring the trainer's execution policy.

Then execute three more updates in full, tiled, and restored-tiled fixtures. Full versus tiled is a tolerance comparison; tiled versus restored-tiled is exact, including loss, persistent weights/moments, counters, RNG, and gradients where the helper requests them. There are 28 full/tiled update pairs and 12 exact future continuation groups across the four configurations.

```c title="tests/test_tiled_training.c"
static int training_resume(DlBackend backend, int fixed)
{
    int result=0,fd=-1;
    char path[]="/tmp/dl-tiled-XXXXXX";
    TrainingFixture full={0},tiled={0},restored={0};
    DlWeights weights={0};
    DlTrainStats a={0},b={0},r={0};
    OK(fixture_create(&full,backend,fixed));
    OK(fixture_create(&tiled,backend,fixed));
    OK(fixture_create(&restored,backend,fixed));
    OK(use_tiled(&tiled,3,2)); OK(use_tiled(&restored,3,2));
    /* A correct mode in the state paired with the wrong tape is rejected
     * before the trainer opens a gradient group or consumes random numbers. */
    STATUS(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,
                           &full.tape,&b),DL_SHAPE);
    CHECK(tiled.optim.updates==0 && tiled.state.rng.state==2026);
    for (size_t i=0;i<4;++i) {
        OK(dl_train_update(&full.optim,&full.state,&full.training,&full.batch,&full.tape,&a));
        OK(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,&tiled.tape,&b));
        CHECK(close_value(a.loss,b.loss,3e-6));
        CHECK(persistent_close(&full,&tiled));
    }
    fd=mkstemp(path); CHECK(fd!=-1); CHECK(close(fd)==0); fd=-1;
    OK(dl_checkpoint_save(path,&tiled.optim,&tiled.state,&tiled.training,&tiled.valid,1024*1024));
    /* Wrong strategy and wrong tile size cannot silently resume the run. */
    STATUS(dl_checkpoint_load(path,&full.optim,&full.state,&full.training,&full.valid,1024*1024),DL_SHAPE);
    restored.state.query_tile=4;
    STATUS(dl_checkpoint_load(path,&restored.optim,&restored.state,&restored.training,&restored.valid,1024*1024),DL_SHAPE);
    CHECK(restored.optim.updates==0 && restored.state.tokens_seen==0);
    restored.state.query_tile=3;
    OK(dl_checkpoint_load(path,&restored.optim,&restored.state,&restored.training,&restored.valid,1024*1024));
    CHECK(fixture_equal(&tiled,&restored,0));
    OK(dl_checkpoint_load_weights(path,&weights,tiled.model.config,1024*1024,1024*1024));
    for (size_t p=0;p<weights.param_count;++p) {
        DlMatrix w=weights.params[p].value;
        CHECK(memcmp(w.base+w.offset,tiled.model.params[p].value.data,w.rows*w.cols*sizeof(float))==0);
    }
    for (size_t i=0;i<3;++i) {
        OK(dl_train_update(&full.optim,&full.state,&full.training,&full.batch,&full.tape,&a));
        OK(dl_train_update(&tiled.optim,&tiled.state,&tiled.training,&tiled.batch,&tiled.tape,&b));
        OK(dl_train_update(&restored.optim,&restored.state,&restored.training,&restored.batch,&restored.tape,&r));
        CHECK(persistent_close(&full,&tiled));
        CHECK(b.loss==r.loss && fixture_equal(&tiled,&restored,1));
    }
    printf("tiled train/resume: backend=%d fixed=%d seven matched updates, three exact continuations\n",
           (int)backend,fixed);
    result=1;
done:
    if (fd!=-1) (void)close(fd);
    (void)unlink(path);
    dl_weights_destroy(&weights);
    fixture_destroy(&restored); fixture_destroy(&tiled); fixture_destroy(&full);
    return result;
}
```

## Print a controlled sequence-length sweep

Hold vocabulary, layer count, all widths, batch size, and tile capacities fixed. Only sequence length changes from 32 through 256. Count both float and double owners through the public accessor and destroy each pair before allocating the next. These allocations are deliberately small; the purpose is to expose the shape of memory growth, not fill the machine.

```c title="tests/test_tiled_training.c"
static int storage_scale(void)
{
    int result=0;
    DlModel m={0}; DlTrainTape full={0},tiled={0};
    DlModelConfig c={.vocab=256,.layers=2,.width=32,.query_heads=4,.kv_heads=2,
        .ffn_width=64,.epsilon=1e-5f,.init_std=0.125f,.rope_theta=10000,.seed=42};
    OK(dl_model_create(&m,c));
    for (size_t time=32;time<=256;time*=2) {
        OK(dl_train_tape_create(&full,&m,1,time,64*1024*1024));
        OK(dl_train_tape_create_tiled(&tiled,&m,1,time,16,32,64*1024*1024));
        size_t f=0,t=0; OK(dl_train_tape_payload_bytes(&full,&f));
        OK(dl_train_tape_payload_bytes(&tiled,&t));
        printf("training payload: T=%zu materialized=%zu tiled=%zu\n",time,f,t);
        dl_train_tape_destroy(&tiled); dl_train_tape_destroy(&full);
    }
    result=1;
done:
    dl_train_tape_destroy(&tiled); dl_train_tape_destroy(&full); dl_model_destroy(&m);
    return result;
}
```

### Run every model and continuation configuration

All failure exits still destroy tapes before their models. The final summary reports finite-difference count and maximum observed full/tiled disagreements, after all assertions have passed.

```c title="tests/test_tiled_training.c"
int main(void)
{
    const size_t heads[]={1,2,4};
    for (int backend=0;backend<2;++backend) {
        for (size_t h=0;h<3;++h)
            if (!model_case(h==0?1:2,heads[h],(DlBackend)backend)) return EXIT_FAILURE;
        for (int fixed=0;fixed<2;++fixed)
            if (!training_resume((DlBackend)backend,fixed)) return EXIT_FAILURE;
    }
    if (!storage_scale()) return EXIT_FAILURE;
    printf("tiled model differences=%zu max gradient disagreement=%.6g max weight disagreement=%.6g\n",
           model_differences,gradient_error,update_error);
    puts("all tiled training checks passed");
    return EXIT_SUCCESS;
}
```

## Exercise real application files and operating-system failure

Add `tests/test_tiled_cli.py`. Python launches the C executables, creates or mutates small file fixtures, and compares output bytes. It does no model computation.

The existing [fixture script](https://25d-228.github.io/gpt-c/media/chapter-19/reference/scripts/make_training_fixture.py) writes original diagnostic training and validation text into a temporary directory. The test uses a twelve-update schedule, stops at update five, and resumes to twelve. Compare the final files byte-for-byte and compare seven resumed update records after removing only elapsed seconds. Matching logs without matching moments would not be enough; matching checkpoint bytes checks the saved persistent state too.

`struct.pack_into("<Q", ...)` writes one unsigned 64-bit little-endian test field at a known offset. The test deliberately corrupts version, policy, capacities, trailer, and file length. This is not a format parser used in model computation.

On POSIX systems where Python supplies `resource`, the child process receives a 1,024-byte file-size limit and ignores `SIGXFSZ` so the C writer observes a write/flush failure. `preexec_fn` applies this only to the child. The test verifies that the existing checkpoint bytes and directory cleanup survive. Where that API is absent it prints a narrow skip rather than claiming the failure path was exercised; it was exercised on the authoring Linux host.

```python title="tests/test_tiled_cli.py"
#!/usr/bin/env python3
"""Train/resume/load C executables. Python only edits file fixtures and compares bytes."""
from __future__ import annotations

import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

def main() -> None:
    train = Path(sys.argv[1] if len(sys.argv) > 1 else "build/release/train").resolve()
    generate = Path(sys.argv[2] if len(sys.argv) > 2 else "build/release/generate").resolve()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/byte-check.conf"
    with tempfile.TemporaryDirectory(prefix="dl-tiled-") as directory:
        work = Path(directory)
        text = work / "text"
        subprocess.run([sys.executable, str(root / "scripts/make_training_fixture.py"), str(text)],
                       check=True, capture_output=True, text=True)
        prompt = work / "prompt.bin"
        prompt.write_bytes(b"A")

        def execute(command: list[str], *, fail: bool = False, preexec_fn=None):
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=90, preexec_fn=preexec_fn)
            if (result.returncode != 0) != fail:
                raise AssertionError(f"{command}\n{result.stdout}\n{result.stderr}")
            return result

        def run(mode: str, path: Path, stop: int, *, engine: str = "tiled",
                fail: bool = False, preexec_fn=None):
            return execute([str(train), mode, str(config), str(text / "train.bin"),
                            str(text / "valid.bin"), str(path), "12", str(stop),
                            "2", "random", "blocked", engine],
                           fail=fail, preexec_fn=preexec_fn)

        def infer(path: Path, output: Path, engine: str, *, fail: bool = False):
            return execute([str(generate), str(config), str(path), str(prompt), str(output),
                            "16", "15", "sample", "0.8", "42", "blocked", engine], fail=fail)

        split, whole = work / "split.bin", work / "whole.bin"
        run("new", split, 5)
        original = split.read_bytes()
        if original[:8] != b"DLCKPT02" or original[-8:] != b"ENDCKP02" or \
                struct.unpack_from("<QQQ", original, 256) != (1, 16, 32):
            raise AssertionError("wrong v2 format/policy extension")
        run("resume", split, 6, engine="full", fail=True)
        run("new", split, 6, fail=True)
        if split.read_bytes() != original:
            raise AssertionError("rejected mode/new call changed the checkpoint")
        try:
            import resource
        except ImportError:
            print("tiled cli: OS write-failure injection unavailable")
        else:
            def limited_file() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            failed = run("resume", split, 6, fail=True, preexec_fn=limited_file)
            if "dl_checkpoint_save" not in failed.stderr or split.read_bytes() != original or \
                    list(work.glob("split.bin.tmp.*")):
                raise AssertionError("failed write did not preserve original file/cleanup")
            print("tiled cli: actual OS short write preserved v2 checkpoint")

        continued = run("resume", split, 12)
        uninterrupted = run("new", whole, 12)
        def updates(output: str) -> list[str]:
            return [line.split(" seconds=", 1)[0] for line in output.splitlines()
                    if line.startswith("update=")]
        if split.read_bytes() != whole.read_bytes() or \
                updates(continued.stdout) != updates(uninterrupted.stdout)[5:]:
            raise AssertionError("v2 resume did not reproduce future state")
        print("tiled cli: seven resumed logs and final checkpoint bytes match")
        transcripts = []
        for engine in ("full", "cache", "tiled"):
            output = work / f"{engine}.bin"
            infer(whole, output, engine)
            transcripts.append(output.read_bytes())
        if len(set(transcripts)) != 1:
            raise AssertionError("v2 weight-only loading strategies disagree")
        print("tiled cli: v2 weights generate identical full/cache/tiled fixture bytes")

        damaged = work / "damaged.bin"
        rejected = work / "rejected.bin"
        bad_cases: list[bytes] = []
        for offset, value in ((8, 3), (256, 2), (264, 0), (272, 0)):
            data = bytearray(original)
            struct.pack_into("<Q", data, offset, value)
            bad_cases.append(bytes(data))
        bad_cases.extend(original[:cut] for cut in (256, 260, 264, 272, 279, 280))
        bad_cases.append(original[:-8] + b"ENDCKP01")
        bad_cases.append(original + b"x")
        for content in bad_cases:
            damaged.write_bytes(content)
            run("resume", damaged, 6, fail=True)
            infer(damaged, rejected, "full", fail=True)
            if damaged.read_bytes() != content or rejected.exists():
                raise AssertionError("failed loader changed file or produced output")
        # A different positive tile policy is legal metadata for inference, but
        # does not describe the same continuation arithmetic for training.
        data = bytearray(original)
        struct.pack_into("<Q", data, 264, 8)
        damaged.write_bytes(data)
        run("resume", damaged, 6, fail=True)
        infer(damaged, work / "other-policy.bin", "full")
        print(f"tiled cli: {len(bad_cases)} damaged/truncated files rejected by both loaders; policy mismatch rejected")
    print("all tiled training CLI checks passed")

if __name__ == "__main__":
    main()
```

## Integrate the build targets

Replace the root `Makefile`. `CORE` and headers already include tiled forward from Chapter 18; the new C program is added to `PROGRAMS`, so `make test` runs it in all three build modes. `test-tiled-cli` depends on the release trainer and generator. Earlier numerical, training, and generation tests remain.

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
CORE = src/common.c src/tensor.c src/kernels.c src/loss.c src/norm.c src/ffn.c src/rope.c src/attention_ref.c src/attention_tiled.c src/rng.c src/model.c src/data.c src/optim.c src/train.c src/checkpoint.c src/inference.c src/sample.c src/cache.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h \
          include/dl/loss.h include/dl/norm.h include/dl/ffn.h include/dl/rope.h \
          include/dl/attention.h include/dl/attention_tiled.h include/dl/rng.h include/dl/model.h include/dl/data.h include/dl/optim.h include/dl/train.h include/dl/checkpoint.h include/dl/inference.h include/dl/sample.h include/dl/cache.h tests/tiny.h tests/training_fixture.h
PROGRAMS = test_kernels test_loss gradcheck test_blocked test_ops test_ffn test_rope test_attention test_attention_backward test_model test_data test_optim test_training test_checkpoint test_inference test_sample test_cache test_tiled test_tiled_training

.PHONY: all debug sanitize release test test-assets test-training-cli test-generation-cli clean

all: debug

debug: $(addprefix build/debug/,$(PROGRAMS)) build/debug/compare build/debug/inspect_data build/debug/train build/debug/generate
sanitize: $(addprefix build/sanitize/,$(PROGRAMS)) build/sanitize/compare build/sanitize/inspect_data build/sanitize/train build/sanitize/generate
release: $(addprefix build/release/,$(PROGRAMS)) build/release/compare build/release/inspect_data build/release/train build/release/generate

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

build/debug/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/sanitize/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

build/release/generate: app/generate.c $(CORE) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(CORE) $< -o $@ $(LDLIBS)

test-generation-cli: build/release/train build/release/generate
	python3 tests/test_generate_cli.py build/release/train build/release/generate full
	python3 tests/test_generate_cli.py build/release/train build/release/generate cache
	python3 tests/test_generate_cli.py build/release/train build/release/generate tiled

.PHONY: test-tiled-cli
test-tiled-cli: build/release/train build/release/generate
	python3 tests/test_tiled_cli.py build/release/train build/release/generate
```

## Reproduce the source state and run the checks

The complete companion is `book/media/chapter-19/reference/` in the cumulative package. Its 73 files consist of the complete Chapter 18 prerequisite state plus these changes:

| Change | Paths |
|---|---|
| Replace attention implementation | `include/dl/attention_tiled.h`, `src/attention_tiled.c` |
| Replace tape and run-state integration | `include/dl/model.h`, `src/model.c`, `include/dl/train.h`, `src/train.c` |
| Replace checkpoint/application integration | `include/dl/checkpoint.h`, `src/checkpoint.c`, `app/train.c` |
| Replace the existing test/build files | `tests/test_tiled.c`, `Makefile` |
| Add complete tests | `tests/test_tiled_training.c`, `tests/test_tiled_cli.py` |

All other 60 files are unchanged. The file-titled code blocks in Parts 1–5 supply the complete replacements/additions, in order. Concatenate consecutive blocks for the same file with one blank line. Do not insert explanatory prose into the source. The complete earlier chapter remains available when tracing a retained dependency.

From the updated implementation directory on the native Mac:

```sh
make debug sanitize release
make test
make test-assets test-training-cli test-generation-cli test-tiled-cli
```

The default `CC = xcrun clang` is the Chapter 1 convention, not a requirement for an exact compiler release. With a compatible compiler directly on `PATH`, `make CC=clang ...` overrides it. The following focused commands are useful before running the complete suite:

```sh
./build/debug/test_tiled
./build/debug/test_tiled_training
./build/sanitize/test_tiled
./build/sanitize/test_tiled_training
python3 tests/test_tiled_cli.py build/release/train build/release/generate
```

A successful run has no `FAILED:` lines or sanitizer diagnostics. On the tested Clang 17 x86_64 Linux build, the whole-model summary was:

```text
tiled model differences=204 max gradient disagreement=4.84288e-08 max weight disagreement=4.47035e-07
all tiled training checks passed
```

Together with Part 4, this is **1,236 new numerical-difference comparisons per build configuration**: 624 core, 408 complete sublayer, and 204 selected complete-model coordinates. The full debug, AddressSanitizer/UndefinedBehaviorSanitizer, and optimized suites passed, including the inherited tests. The new CLI test passed with the corresponding trainer/generator executables from all three modes. Leak-enabled runs of the two new/extended C test programs reported no leaks on that host.

These observations are not M4 execution results. Floating-point differences can vary with the platform's elementary functions and compiler. Investigate failed numerical bounds; do not erase them merely to reproduce a printed example.

## Read the allocation sweep without calling it process memory

For $B=1$, $L=2$, $D=32$, $H_q=4$, $H_{kv}=2$, $d=8$, $F=64$, $V=256$, $R=16$, and $S=32$, the executable reported:

| Sequence length | Materialized tape payload, bytes | Tiled tape payload, bytes |
|---:|---:|---:|
| 32 | 366,592 | 298,624 |
| 64 | 929,792 | 573,056 |
| 128 | 2,646,016 | 1,121,920 |
| 256 | 8,437,760 | 2,219,648 |

The tiled column is affine in sequence length at these fixed capacities. The materialized column includes the square attention term. The table counts successfully constructed numerical arrays; it excludes model/optimizer owners, metadata, data buffers, stacks, executable/runtime mappings, and allocator behavior. It is **not resident set size** and is not a measured 16 GB system limit.

Very short sequences or oversized tiles can make tiling consume more storage than materialization. The purpose is the controlled growth and absence of square attention state, not a universal claim that one method wins at every size. Both implementations still visit a quadratic number of causal query/key pairs, and the two-sweep backward deliberately recomputes derivatives. No speedup follows from this table.

## Run actual training through the tiled backward

Use a fresh run directory. `scripts/make_training_fixture.py` refuses to overwrite its output directory; `train new` refuses existing checkpoints. The following commands reproduce the small, inspectable experiment without a network download:

```sh
mkdir -p runs/chapter-19
python3 scripts/make_training_fixture.py runs/chapter-19/text
./build/release/train new configs/byte-pilot.conf \
  runs/chapter-19/text/train.bin runs/chapter-19/text/valid.bin \
  runs/chapter-19/full.bin 160 160 2 random blocked full
./build/release/train new configs/byte-pilot.conf \
  runs/chapter-19/text/train.bin runs/chapter-19/text/valid.bin \
  runs/chapter-19/tiled.bin 160 160 2 random blocked tiled
```

The pilot has 29,856 unique parameters, $B=2$, $T=16$, and two microbatches per update: 64 targets per committed update, or 10,240 target uses over 160 updates. Training samples repeatedly from 1,214 bytes of diagnostic text. The separate validation file has 393 bytes; the application scores 128 targets per report and explicitly reports 264 unscored targets because it limits evaluation to four full batches.

Both commands were run in C on the authoring Linux host. Their actual validation loss in natural-log units was:

| Attention strategy | Before training | After 160 updates |
|---|---:|---:|
| Materialized | 5.725389095 | 2.270204153 |
| Tiled | 5.725389094 | 2.270204138 |

The close values support this bounded integration check, not bitwise equivalence between strategies, broad language quality, or a performance result. The corpus is tiny and deliberately repetitive. Do not interpret its loss as a generalization benchmark. The version-2 checkpoint from tiled training is loadable by the existing generator; the CLI tests separately verify full/cached/tiled generation on a deterministic fixture.

To inspect continuation manually, choose a **new** checkpoint filename, keep total at 160, first run with `STOP=40`, and then use `resume` with `STOP=160` and the same `tiled` argument. Changing `TOTAL` changes the schedule and is correctly rejected. The twelve-update CLI test performs the exact-byte comparison automatically.

## Deliberately break one mechanism at a time

Work in a disposable copy or save the current file first. Change only the described expression, rebuild the affected test, observe its failure, and restore it before the next experiment. Sanitizers remain enabled; none of these mistakes requires an out-of-bounds access.

**Omit the softmax correction.** In `derivative_tile`, replace `double g = p * (dp - center[r]);` with `double g = p * dp;`. The hand-derived key gradients and independent backward oracle fail. The missing correction enforces competition between keys; without it, adding the same constant to all scores would spuriously change the result.

**Keep only one sharing head.** In the key-owned sweep of `backward_compute`, called by `dl_attention_tiled_core_backward`, change the local-head loop from `gh < group` to `gh < 1`. Query derivatives can remain plausible because their sweep is unchanged, but shared K/V gradients lose contributions. The two-head hand fixture fails.

**Drop the key scale.** At the final key-gradient store, omit multiplication by `scale` while leaving the query store unchanged. The hand fixture and oracle reject it. Scaling only Q is not the derivative of a scaled dot product.

**Reconstruct from log-sum-exp only.** In `derivative_tile`, replace `double p = exp(delta) / f->data[f->row_capacity + sr];` with `double p = exp((double)score - f->data[2 * f->row_capacity + sr]);`. Ordinary inputs may still look correct. The enormous equal-score backward fixture rejects its value gradient, showing why extreme numerical tests complement finite differences.

**Reuse one block's statistics.** In the model's tiled forward and backward dispatch, replace `&t->tiled[layer]` by `&t->tiled[0]` while leaving the allocation code unchanged. The last block overwrites the first block's normalization state. One-block tests may still pass; the two-block model checks fail. Separate forward storage is a correctness requirement, not merely a memory accounting detail.

All five defects were compiled and executed as separate sanitizer builds during authoring. Each failed an explicit numerical/state assertion without an AddressSanitizer or UndefinedBehaviorSanitizer report. A memory-safe program can still implement the wrong derivative or associate the wrong saved state with a layer.

## Completion checks and references

The implemented result is a selectable, matched tiled training path with no quadratic score/probability/score-gradient tape, explicit per-block saved statistics, bounded double backward tiles, complete manual derivatives, and tested continuation. Before moving on, explain the two reduction ownership orders, calculate the payload for a small configuration, and identify exactly which earlier contributions would be lost by an incorrect SET. Record actual hands-on time separately from reading and unattended runs; the chapter's estimate remains **7.00 hours** until learner timings support a recalibration.

The source consulted for the recomputation mechanism was [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, version 2, Appendix B.2 and B.4](https://arxiv.org/html/2205.14135v2). The dense softmax derivation, head-sharing rules, and projection orientations are developed locally in [Chapter 11](https://25d-228.github.io/gpt-c/chapter-11.html); the numerical normalization representation is developed in [Chapter 18](https://25d-228.github.io/gpt-c/chapter-18.html). This implementation is a CPU teaching kernel, not a reproduction of a CUDA launch schedule or a measured claim about that paper's performance.

No native M4/macOS execution, resident-memory measurement, attention-speed comparison, live TinyStories preparation, or pretrained-model comparison is claimed by this chapter's authoring runs. The next chapter adds the explicit pretrained configuration and weight-import boundary while preserving these C computation paths.
