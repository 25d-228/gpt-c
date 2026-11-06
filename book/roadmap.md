# Decoder Laboratory in C

Build a decoder whose computation can be followed from a byte in a text file to a predicted token, and from a scalar training loss back to every parameter. The implementation is a native C17 program: explicit array layouts, owned buffers, handwritten derivatives, a training loop, and a stateful inference engine. No tensor framework executes the model on its behalf.

The same engine has two uses. A **2,885,888-parameter byte model** learns from a bounded text corpus. An inference-only configuration loads **SmolLM2-135M** and connects the same operators to pretrained weights and their tokenizer. The additional memory-efficiency mechanism is **exact tiled attention, including backward computation**. It removes the full attention-probability matrix from the optimized path without changing the intended attention function.

> **Learning objectives**
> Trace shapes and ownership through a modern decoder; derive and implement its gradients; train, evaluate, checkpoint, and resume a small model; distinguish full-prefix evaluation, prefill, and incremental decoding; and explain the numerical and memory tradeoffs of tiled attention using working C code.

## What the completed program does

The finished laboratory contains a scalar reference path and a blocked C matrix-product path. Both remain usable for diagnosis. A command-line training program performs forward computation, loss evaluation, backward computation, gradient accumulation, clipping, and optimizer updates in C. A generation program uses either full-prefix recomputation or a persistent key/value cache. A comparison program checks numerical agreement before measuring memory or elapsed time.

The final demonstration crosses these components rather than presenting isolated operators. It trains the byte model, evaluates held-out text, writes a checkpoint, resumes the next updates, and generates a continuation. It compares full-prefix and cached logits on identical token prefixes. It compares materialized and tiled attention outputs **and all three input gradients**, then repeats the comparison through a model update. Finally, it imports the pretrained configuration and compares short C-computed continuations and logits with an independent reference.

A training run is not replaced by an overfit-one-batch exercise. The latter is an early diagnostic. The former uses separate training and validation inputs, reports the number of processed tokens, and preserves enough state to resume. Output quality depends on the actual data and compute used; the small byte model is not presented as an instruction-following assistant.

## Starting knowledge and native tools

Basic C expressions, loops, functions, and elementary Python are sufficient starting points. The foundations introduce translation units, pointer contracts, ownership, numerical types, and differentiation before the decoder build. Matrix multiplication, probability normalization, the chain rule, attention, and tokenization boundaries are explained when first needed; transformer expertise is not a prerequisite.

The target is an **M4 Mac mini with 16 GB unified memory**, running native macOS. Use an installed compatible Apple Clang/Command Line Tools toolchain, a shell, `make`, and Python 3 for preparation and checks. C17 is the source-language baseline, not a requirement for one exact compiler release. The required features are ordinary C compilation and linking, the C mathematical library, debug symbols, and usable address/undefined-behavior sanitizer builds. Chapter 1 tests the installed toolchain rather than inferring capabilities from its version string. See [Apple's tool resources][apple-tools] and the [Clang sanitizer instructions][asan].

These observation commands do not install or upgrade anything:

```sh title="Terminal — inspect the local environment"
sw_vers
uname -m
xcrun clang --version
xcrun lldb --version
make --version
python3 --version
```

Keep debug/sanitized and optimized builds separate. Correctness checks do not use `-ffast-math`: its assumptions about non-finite values and algebraic transformations are inappropriate for those checks, as documented in the [Clang floating-point guide][clang-fp]. No particular macOS version is assumed. A compatible alternative compiler is acceptable; it must pass the same small feature checks.

Python dependencies enter only when an asset or reference exercise needs them. Dataset streaming, Safetensors conversion, tokenization, and an isolated model checker are separate from the C executable. The C engine requires neither a Python interpreter nor a framework runtime once inputs have been prepared. The optional final Accelerate adapter substitutes only matrix products already implemented in C. There is no required Metal, CUDA, virtual machine, GPU cluster, or architecture-specific assembly.

## One architecture, two configurations

A decoder predicts the next token from preceding tokens. Here a token is an integer indexing an embedding table. A training window supplies an input sequence and a target sequence shifted forward by one position. The hidden state is a vector for each input token. Each block updates those vectors using earlier positions, never future ones.

The block ordering is pre-normalization, attention, residual addition, pre-normalization, a gated feed-forward network, and a second residual addition. A residual addition adds a sublayer's result to its input. [LLaMA Section 2.2][llama] provides an architectural reference for pre-normalization, RMSNorm, SwiGLU, and rotary positions; [GQA Section 2.2][gqa] describes sharing key/value heads among groups of query heads.

The named operators have specific jobs. RMSNorm rescales each token's coordinates using their root mean square and learned per-coordinate scales. RoPE rotates query and key coordinates according to position. Grouped-query attention forms weighted sums of earlier value vectors. SwiGLU multiplies a SiLU-activated projection by a second projection before projecting back to the model width. Each operator receives both a forward implementation and a derived backward implementation; the component papers are linked in their chapter plans below.

Let $X$ be the input to a block, $A$ the attention sublayer including its projections, and $F$ the feed-forward sublayer. The subscripts distinguish the two independently learned normalization scales:

$$
U = X + A(\operatorname{RMSNorm}_1(X)),
$$

$$
Y = U + F(\operatorname{RMSNorm}_2(U)).
$$

The output $Y$ becomes the next block's input. After the last block, a final RMSNorm produces $Z$. The embedding matrix $E$ is also the output-projection weight:

$$
\text{logits} = Z E^{\mathsf T}.
$$

A logit is an unnormalized score for one vocabulary entry. Sharing $E$ means one parameter allocation, not two copies that happen to start equal. Its gradient receives contributions from both embedding lookup and output projection. The required model has no biases, dropout, learned position table, or general-purpose automatic-differentiation engine.

### Dimensions and tensor conventions

Use $B$ for the number of sequences in a microbatch, $T$ for tokens per sequence, $V$ for vocabulary size, $d$ for model width, $L$ for block count, $H_q$ for query heads, $H_{kv}$ for key/value heads, $d_h=d/H_q$ for coordinates per head, and $f$ for feed-forward width. A microbatch is the group processed by one forward/backward pair; several microbatches can contribute to one update.

| Quantity | Byte-model training configuration | Pretrained inference configuration |
|---|---:|---:|
| Vocabulary $V$ | 256 byte IDs | 49,152 tokenizer IDs |
| Blocks $L$ | 4 | 30 |
| Width $d$ | 256 | 576 |
| Query heads $H_q$ | 8 | 9 |
| Key/value heads $H_{kv}$ | 2 | 3 |
| Head width $d_h$ | 32 | 64 |
| Feed-forward width $f$ | 704 | 1,536 |
| RMSNorm epsilon | $10^{-5}$ | $10^{-5}$ |
| RoPE frequency base | 10,000 | 100,000 |
| Initial operating size | $B=2$, $T=128$ | $B=1$, cache capacity 1,024 |
| Parameter training | All unique parameters | None |

The pretrained values come from the [official SmolLM2-135M configuration][smollm-config]. Its upstream position limit is 8,192; **1,024 is this laboratory's initial cache limit**, not a claim about the model's maximum context. Conversion must validate the actual downloaded configuration and tensor inventory. The byte-model values are laboratory design choices, not a reconstruction of upstream pretraining.

All required weights, activations, gradients, and optimizer arrays are stored as FP32. Small numerical checks can use double-precision evaluation or accumulation as explicitly identified by the relevant chapter. Matrices use row-major storage. A linear layer stores its weight as `[output_features, input_features]` and computes $XW^{\mathsf T}$; it does not silently transpose the checkpoint's meaning.

| Buffer | Logical shape | Storage or ownership rule |
|---|---|---|
| Input and target IDs | `[B, T]` each | Integer IDs; validate before lookup |
| Embedding table | `[V, d]` | Also owns the tied output weight |
| Hidden state | `[B, T, d]` | May be viewed as `[B*T, d]` without copying |
| Queries | `[B, T, Hq, dh]` | Query-head axis is not the key/value-head axis |
| Keys and values | `[B, T, Hkv, dh]` each | Shared heads are not physically repeated |
| Reference attention probabilities | `[B, Hq, T, T]` | Stored only by the materialized training path |
| Tiled attention row statistics | `[B, Hq, T]` | Per-row log-sum-exp, plus saved operator inputs/output |
| Inference key and value caches | `[L, B, Hkv, capacity, dh]` each | Only entries before committed length are valid |

Validate positive dimensions, divisibility of $d$ by $H_q$, divisibility of $H_q$ by $H_{kv}$, and even $d_h$. Query head `h` uses key/value head `h / (Hq / Hkv)`. RoPE uses **split-half coordinate pairing** throughout, matching the selected pretrained configuration. Position zero, later positions, and backward rotations are checked before cache integration.

### Ownership, state, and dependency boundaries

The model owns unique parameter arrays and a stable parameter directory. Training state separately owns gradients, Adam moments, and saved forward intermediates needed by backward computation. Inference owns a reusable workspace and, when requested, a cache. Loading pretrained weights must not allocate training moments or retain every layer's training intermediates.

Kernel inputs are borrowed buffers with declared shapes, strides, and overlap rules. Outputs explicitly distinguish overwrite from accumulation. Allocation happens at controlled boundaries, not in each inner product. Failed shape checks and allocation attempts return errors with cleanup; input validation is not replaced by assertions that disappear in optimized builds. Chapter 2 establishes these conventions with a standalone program before they become model infrastructure.

Python may prepare a bounded corpus, convert named checkpoint tensors to the documented course format, tokenize/detokenize pretrained inputs and outputs, and emit independent reference values. It does not perform the C program's projections, normalization, attention, loss, updates, sampling, or cache management. A reference framework is a separate checker, not a hidden fallback. Safetensors parsing and the upstream tokenizer are peripheral dependencies; C validates and owns the converted weights it executes.

## Resource envelope and useful measurements

The bias-free architecture with a tied embedding/output table has the following parameter count:

$$
N = Vd + L\left(2d^2 + 2dH_{kv}d_h + 3df + 2d\right) + d.
$$

The terms count the shared table, query/output projections, key/value projections, three feed-forward projections, two normalization scales per block, and the final scale. This gives 2,885,888 parameters for training and 134,515,008 for pretrained inference. The latter is calculated from the published configuration, not inferred from the rounded model name.

One MiB means $2^{20}$ bytes. The following are **array-size calculations, not observed process memory**:

| Allocation | Calculated size | What remains outside it |
|---|---:|---|
| Byte-model FP32 weights | 11.01 MiB | Gradients, moments, saved activations |
| Byte-model weights + gradients + two Adam moments | 44.04 MiB | Data and all activation/workspace storage |
| One attention matrix per training block, all four blocks, $B=2,T=128$ | 4.00 MiB | Any additional score/gradient matrices |
| Pretrained FP32 weights | 513.13 MiB | Cache, workspace, loader intermediates |
| Pretrained key/value cache, capacity 1,024 | 45.00 MiB | Weights and current-token workspace |
| One pretrained layer's materialized attention matrix at $T=1024$ | 36.00 MiB | Other layer buffers |
| Full pretrained logits for all 1,024 positions | 192.00 MiB | Avoided when only final-position logits are needed |

The cache calculation is $2LBH_{kv}Cd_h\cdot4$ bytes, where $C$ is allocated capacity and the factor two counts keys and values. Normal generation computes the needed final-position vocabulary scores rather than retaining all prompt-position logits. Training and short reference comparisons have different output requirements.

Initial **working budgets to verify** are below 512 MiB for the default byte-model training process and below 2 GiB for the pretrained C inference process. They are neither measured peak resident sizes nor whole-system guarantees. Report accounted live arrays separately from operating-system process measurements. Run weight conversion and the independent Python reference separately from the C memory measurement; include temporary conversion copies in their own memory accounting.

CPU runtime remains an empirical question. Start training with a short pilot and measure the optimized C path before committing to a longer run. Aim initially for roughly **one to two million processed byte tokens**, with the achieved token count stated alongside loss. Reduce the run size or the number of repeated measurements when necessary, not the implementation of backward computation or cache correctness. Accumulation increases tokens per update without multiplying simultaneously live activation storage. Reading time and unattended compute are separate from implementation effort.

Exact tiled attention means the same mathematical attention as the materialized reference, not identical last bits or guaranteed speedup. The online-normalization and recomputation ideas are described in the [online-normalizer paper][online] and [FlashAttention][flash]. The laboratory implements CPU loops and buffers, not that paper's tuned GPU kernel. Quadratic query/key arithmetic remains; the removed allocation is the full score/probability matrix in the optimized path.

## Implementation layout

The learner project lives on `work/decoder-lab`. Its files are introduced progressively; this is the intended application layout, not a requirement to create empty files in advance. The course index remains on `main`, and the book remains separate from application source.

| Location | Responsibility |
|---|---|
| `Makefile` | Debug, sanitizer, optimized, test, and measurement targets |
| `examples/01-build/` | Complete multi-file dot-product program |
| `examples/02-storage/` | Complete owned-buffer and borrowed-view program |
| `examples/03-numerics/` | Complete floating-point and finite-difference program |
| `include/dl/` | Public C headers for the modules below |
| `src/common.c`, `src/tensor.c` | Status, checked sizes, storage, views, and shape checks |
| `src/kernels.c`, `src/loss.c` | Scalar/blocked products, embeddings, cross-entropy, and their derivatives |
| `src/norm.c`, `src/ffn.c`, `src/rope.c` | RMSNorm, SwiGLU, and positional rotations |
| `src/attention_ref.c`, `src/attention_tiled.c` | Materialized and bounded-tile attention paths |
| `src/model.c`, `src/rng.c` | Configuration, unique parameters, decoder composition, and explicit random state |
| `src/data.c`, `src/optim.c`, `src/checkpoint.c` | Byte windows, training updates, and validated persistence |
| `src/inference.c`, `src/cache.c`, `src/sample.c` | Inference workspace, cache state, and token selection |
| `src/accelerate.c` | Optional matrix-product adapter introduced at the end |
| `app/train.c`, `app/generate.c`, `app/compare.c` | Small command-line drivers, not a service |
| `tests/` | Deterministic fixtures and executable checks introduced with each subsystem |
| `scripts/` | Bounded data preparation, weight conversion, tokenization, and independent checks |
| `configs/` | Tiny diagnostic, byte-training, and validated imported configurations |
| `data/`, `weights/`, `runs/`, `build/` | Local assets and generated outputs; not committed course content |

A C source module's public interface is declared in the matching header under `include/dl/`. The three foundation examples remain independently runnable: they do not depend on this later project tree. Tests and drivers acquire their interfaces in the chapter that first uses them, not through unexplained future helpers.

## Chapter plan and implementation hours

The plan contains **82.0 hours of scoped implementation plus 8.0 hours of debugging/integration reserve: 90.0 hours total**. Chapter estimates already include ordinary exercises, local debugging, tests, and integration of that chapter's code. The reserve covers cross-cutting problems; it is not another charge for every test. Reading, course authoring, publication, downloads, and unattended compute are excluded. Active investigation of a failed run counts as implementation work.

| Chapter | Topic | Hours |
|---|---|---:|
| 1 | Native C builds and a debuggable numerical program | 1.75 |
| 2 | Owned buffers, borrowed views, and failure cleanup | 2.50 |
| 3 | Floating-point arithmetic and numerical derivatives | 2.25 |
| 4 | Tensor layouts and reference matrix products | 4.00 |
| 5 | Embeddings, stable loss, and elementary backward passes | 3.50 |
| 6 | Blocked C kernels and meaningful timing | 2.00 |
| 7 | RMSNorm and residual gradient paths | 3.00 |
| 8 | SwiGLU and the two-branch backward pass | 3.50 |
| 9 | Rotary positions as explicit coordinate transformations | 2.50 |
| 10 | Causal grouped-query attention, forward | 4.00 |
| 11 | Attention backward and shared-head accumulation | 4.50 |
| 12 | Assemble and differentiate the complete decoder | 4.50 |
| 13 | Byte text, shifted targets, and bounded data preparation | 2.00 |
| 14 | AdamW, accumulation, and gradient clipping | 4.00 |
| 15 | Train, evaluate, checkpoint, and resume | 6.00 |
| 16 | Inference workspaces and token sampling | 3.00 |
| 17 | Prefill and incremental key/value caching | 4.50 |
| 18 | Online normalization and tiled attention forward | 5.00 |
| 19 | Recomputed tiled backward without a quadratic tape | 7.00 |
| 20 | Import and validate pretrained weights | 4.00 |
| 21 | Tokenizer boundaries and pretrained reference agreement | 3.50 |
| 22 | End-to-end demonstration and measured tradeoffs | 5.00 |
| | **Scoped implementation** | **82.00** |
| | **Cross-cutting debugging/integration reserve** | **8.00** |
| | **Total** | **90.00** |

Each chapter below names its starting code, section outline, runnable result, and an indicative split of its hours. These are implementation estimates, not reading assignments. Record actual hands-on time as work proceeds and revise the remaining estimates from those observations. A long explanation does not make a short implementation task take longer.

### Chapter 1 — Native C builds and a debuggable numerical program

**Prerequisites:** Basic C expressions, loops, and functions. No existing project files.

**Sections.** Follow a source file through preprocessing, compilation, and linking; separate a declaration in `dot.h` from a definition in `dot.c`; use include guards, `const` pointer parameters, `size_t`, and correct formatted output; build a small program with `make`; inspect a value in LLDB; distinguish a compiler warning, a linker failure, and a runtime memory error. Introduce the flags before using them.

**Build and run.** Create the complete `examples/01-build/{dot.h,dot.c,main.c,Makefile}` program. Its normal case computes the dot product of `[1,2,3]` and `[4,5,6]` and prints `dot = 32`. A zero-length case and a deliberately isolated bad-access exercise expose the difference between an interface contract and a sanitizer diagnostic. Restore a clean passing build after the fault exercise; intentional undefined behavior is not left in the normal test target.

**Outcome and effort.** A runnable multi-file program with debug, sanitizer, and optimized configurations, plus a reproducible debugger session. Allocate 0.50 hours to setup/builds, 0.75 to the program, and 0.50 to diagnosis and checks. Technical reference: [Clang's sanitizer usage][asan].

### Chapter 2 — Owned buffers, borrowed views, and failure cleanup

**Prerequisites:** Chapter 1's build and pointer interface.

**Sections.** Distinguish an array from a pointer and a pointer from an allocation; represent an owning buffer with a struct; express a non-owning view with pointer, extent, and stride; check `size_t` multiplication before allocation; initialize structs before cleanup; use status returns and a single cleanup path; explain aliasing and the limited promise made by `restrict` before any later optimization uses it.

**Build and run.** Create `examples/02-storage/{buffer.h,buffer.c,main.c,Makefile}`. Allocate six floats, fill a two-by-three matrix with `1` through `6`, borrow its second row, and print a row sum of `15`. Reject an overflowing size request without attempting that allocation. Exercise an injected allocation failure and free every successfully acquired buffer. A view never frees its owner, and the program does not read a view after owner destruction.

**Outcome and effort.** A complete storage example that establishes checked construction, unchanged output on failed construction, and cleanup conventions for the later engine. Allocate 1.50 hours to implementation and 1.00 to boundary cases, failure injection, and sanitizer checks.

### Chapter 3 — Floating-point arithmetic and numerical derivatives

**Prerequisites:** Chapters 1–2; no calculus assumed beyond ordinary algebra.

**Sections.** Contrast `float` storage and `double` accumulation; inspect limits with `<float.h>`; explain rounding, cancellation, infinity, and NaN; introduce a derivative as a small-change rate and the chain rule through a two-step scalar calculation; derive centered finite differences; combine absolute and relative error rather than divide blindly by a nearly zero reference.

**Build and run.** Create `examples/03-numerics/numerics.c` and its build target. Compare summation orders, demonstrate a shifted exponential sum, and sweep finite-difference steps for $f(x)=x^3-2x$ at $x=1.5$, whose exact derivative is $4.75$. Print the step, estimated derivative, and error so an excessively small step is visibly different from an incorrect derivative. A non-finite comparison is a failure, not a passing tolerance check.

**Outcome and effort.** A runnable numerical diagnostic and a reusable vocabulary for later gradient checks. Allocate 1.25 hours to the program and 1.00 to experiments and explanation-driven changes. Use the [Clang floating-point documentation][clang-fp] to connect compiler assumptions to observed behavior.

### Chapter 4 — Tensor layouts and reference matrix products

**Prerequisites:** The three complete foundation examples.

**Sections.** Derive a flat offset from a shape; distinguish contiguous storage from a strided view; implement checked element/byte counts; flatten token rows without copying; derive matrix multiplication from a scalar dot product; fix weight orientation and leading dimensions; make overlap and overwrite rules explicit.

**Build and run.** Start `include/dl/`, `src/common.c`, `src/tensor.c`, `src/kernels.c`, the project `Makefile`, and `tests/test_kernels.c`. Implement the scalar forward product and the transpose combinations needed by later derivatives. A two-by-three input multiplied by a known two-by-three weight produces a two-by-two result that can be checked by hand. Include non-square shapes, an invalid dimension request, and guard values around output storage.

**Outcome and effort.** Owned tensors, borrowed views, and complete scalar products with checked public interfaces. These are purpose-built numerical routines, not a generic tensor-expression system. Allocate 2.75 hours to storage and kernels and 1.25 to layout traces, tests, and integration.

### Chapter 5 — Embeddings, stable loss, and elementary backward passes

**Prerequisites:** Chapters 3–4 and the reference kernel test target.

**Sections.** Gather token rows from a table; accumulate repeated-token gradients; turn logits into probabilities with a maximum shift; compute mean next-token cross-entropy through log-sum-exp; derive the combined softmax/loss derivative; derive both linear-layer gradients and distinguish assigning a new gradient from adding another contribution.

**Build and run.** Extend `src/kernels.c`, add `src/loss.c`, and create `tests/test_loss.c` and `tests/gradcheck.c`. A tiny embedding–projection–loss program has completely specified weights and IDs. Repeated IDs must add into the same embedding row. Zero logits over four vocabulary entries have loss $\log 4$. Large finite logits must not overflow simply because exponentiation was performed before subtracting the maximum.

**Outcome and effort.** Embedding, projection, and loss forward/backward routines with finite-difference checks. The checker perturbs actual stored values, restores them after each trial, and reports problematic coordinates. Allocate 2.25 hours to routines/derivations in code and 1.25 to gradient and boundary checks. The [stable-softmax construction][online] is a supporting reference.

### Chapter 6 — Blocked C kernels and meaningful timing

**Prerequisites:** Chapter 5's correct scalar forward and backward products.

**Sections.** Relate loop ordering to contiguous loads; tile output and reduction dimensions; handle tail tiles; preserve accumulation across reduction tiles; use an explicit backend enum; separate correctness builds from measurement builds; measure repeated work without timing printing or setup.

**Build and run.** Add blocked implementations to `src/kernels.c` and a kernel mode to `app/compare.c`. Check forward products, input-gradient products, and weight-gradient products against the scalar versions. Use awkward dimensions that are not multiples of a block size. A test of nonzero initial output catches accidentally clearing an accumulation target on every tile.

**Outcome and effort.** A fully C optimized baseline that remains interchangeable with the reference. Print dimensions, backend, iteration count, elapsed time, and numerical error; do not assume blocking is faster for every shape. No Accelerate dependency enters here. Allocate 1.25 hours to blocked kernels and dispatch and 0.75 to equivalence checks and timing.

### Chapter 7 — RMSNorm and residual gradient paths

**Prerequisites:** Chapters 3–6, including additive gradient interfaces.

**Sections.** Normalize one token row without subtracting its mean; place epsilon inside the square root; separate the direct scaling derivative from the derivative of the denominator; accumulate scale gradients across rows; trace an identity residual path alongside a learned path; decide which forward intermediates must remain alive.

**Build and run.** Add `src/norm.c`, its header, and normalization/residual cases in `tests/test_ops.c`. Work a two-coordinate row by hand, then check input and scale gradients numerically. Test a zero row with positive epsilon and two callers contributing to the same gradient buffer. Removing the denominator's derivative must produce a failing check rather than a superficially plausible result.

**Outcome and effort.** Complete RMSNorm forward/backward and residual-addition gradient rules that can be composed without losing a contribution. Allocate 2.00 hours to implementation and 1.00 to traces and checks. Technical basis: [RMSNorm Sections 4 and 4.2][rmsnorm]; the decoder uses a scale-only, epsilon-stabilized version.

### Chapter 8 — SwiGLU and the two-branch backward pass

**Prerequisites:** Chapters 5–7 and the linear-gradient routines.

**Sections.** Define sigmoid and SiLU with numerically stable evaluation; form independent gate and value projections; multiply their coordinates; project the product back to model width; derive both branch gradients and the SiLU derivative; choose saved buffers instead of destroying values needed in backward.

**Build and run.** Add `src/ffn.c` and feed-forward tests. A tiny row passes through all three weight matrices, with intermediate gate, value, and product arrays visible. Check gradients with respect to the input and each matrix. Use positive, negative, and near-zero gate inputs; a mistaken derivative that differentiates only the gate branch must fail.

**Outcome and effort.** A complete bias-free SwiGLU sublayer, not merely a SiLU activation routine. Allocate 2.25 hours to connected forward/backward code and 1.25 to finite differences, buffer-lifetime checks, and integration. Technical basis: [GLU Variants, equations 5–6][swiglu].

### Chapter 9 — Rotary positions as explicit coordinate transformations

**Prerequisites:** Chapters 3–5 and tensor views from Chapter 4. Chapter 8 is the current sequential code state.

**Sections.** Derive a two-dimensional rotation; connect relative position to rotated query/key dot products; generalize to an even head width; distinguish split-half pairing from adjacent-coordinate pairing; generate frequencies and position-dependent angles; derive backward as multiplication by the rotation transpose.

**Build and run.** Add `src/rope.c` and rotation tests. Check position zero, a known quarter-turn pair, norm preservation within floating-point tolerance, and a backward finite difference. Compare a dot product after shifting both positions by the same offset. Reject an odd head width rather than silently leaving a coordinate untreated.

**Outcome and effort.** Explicit forward and backward rotations with an absolute position argument, ready for both full-sequence attention and cached decoding. Values are not rotated. Allocate 1.50 hours to routines and 1.00 to position/layout checks. Technical basis: [RoFormer Sections 3.1–3.2][rope]; the [checkpoint configuration][smollm-config] specifies the selected pairing convention.

### Chapter 10 — Causal grouped-query attention, forward

**Prerequisites:** Chapters 4–9; stable normalization and positional rotations are already implemented.

**Sections.** Form query, key, and value projections; map each query head to a shared key/value head; scale query/key dot products; mask future positions before normalization; accumulate a weighted value vector; concatenate head results and apply the output projection. Follow a two-token example before introducing longer sequences.

**Build and run.** Add the materialized path in `src/attention_ref.c` and `tests/test_attention.c`. Keep its score/probability arrays visible for inspection. A one-token row has probability one. Changing a future token cannot change an earlier output. Check several query heads sharing one key/value head without duplicating the actual key/value storage.

**Outcome and effort.** A complete causal attention sublayer with declared saved buffers for backward. Invalid shapes and empty permitted-key sets are diagnosed explicitly; diagonal self-attention gives every normal row at least one allowed key. Allocate 2.75 hours to projections, masking, and layout plumbing and 1.25 to numerical traces and tests. Technical basis: [GQA Section 2.2][gqa].

### Chapter 11 — Attention backward and shared-head accumulation

**Prerequisites:** Chapter 10's complete materialized forward path and Chapters 5–9's derivative interfaces.

**Sections.** Reverse the weighted sum; derive the softmax row derivative without allocating its Jacobian; reverse scaled dot products; sum key/value contributions from every sharing query head; reverse rotations and projection matrices; identify where a gradient buffer is zeroed once and where it is subsequently incremented.

**Build and run.** Extend `src/attention_ref.c` and the attention/gradient tests. Compare query, key, value, input, and projection-weight gradients on a small causal sequence. Check that masked score gradients vanish. Exercise the one-key/value-head and one-key/value-head-per-query-head endpoints using matched effective tensors, not unrelated randomly initialized models.

**Outcome and effort.** The materialized attention implementation becomes an executable derivative reference for later memory optimization. Allocate 3.00 hours to the complete backward path and 1.50 to shared-head, mask, and finite-difference checks. Supporting derivation: [FlashAttention Appendix B.2 and its standard backward algorithm][flash].

### Chapter 12 — Assemble and differentiate the complete decoder

**Prerequisites:** All operator code through Chapter 11 and the gradient checker.

**Sections.** Validate model configuration; enumerate unique parameters in a stable order; initialize weights with explicit random state and norm scales; compose residual blocks and the final projection; allocate a training tape of saved intermediates; reverse the block sequence; merge the two gradients of the tied embedding table.

**Build and run.** Add `src/model.c`, `src/rng.c`, matching headers, `configs/tiny.conf`, and `tests/test_model.c`. The diagnostic configuration uses one block, width 16, four query heads, two key/value heads, feed-forward width 40, vocabulary 16, and four tokens. Introduce unsigned random-state operations and the initialization rule before relying on them. Check selected coordinates from every parameter family and a whole-model directional derivative, not millions of finite differences.

**Outcome and effort.** One C forward/backward call produces loss and all unique parameter gradients. A small, fully implemented plain-gradient update is a smoke check; the sustained optimizer follows in Chapter 14. Allocate 3.00 hours to composition, initialization, and parameter ownership and 1.50 to gradient integration and diagnostics. Do not use a plausible scalar loss as the only check of the assembled model.

### Chapter 13 — Byte text, shifted targets, and bounded data preparation

**Prerequisites:** Chapter 12's model interface and explicit random state.

**Sections.** Distinguish bytes, Unicode characters, and tokenizer IDs; open/read/close files with checked results; construct input/target windows with a one-byte shift; keep validation separate from sampled training windows; preserve raw bytes while displaying text safely; bound preparation by bytes as well as record count.

**Build and run.** Add `src/data.c`, `tests/test_data.c`, and `scripts/prepare_data.py`. Begin with a tiny local ASCII fixture whose windows are printed and checked. Then prepare at most 16 MiB of training text and 1 MiB of validation text from their respective [TinyStories splits][tinystories], stopping at record boundaries and retaining source/revision and license information. Use [streaming preparation][streaming] rather than requiring the full corpus. The C loader samples and batches the resulting local bytes.

**Outcome and effort.** Bounded train/validation files and a deterministic C batch iterator. Byte IDs are exactly 0–255: there is no invented byte-model BOS/EOS or padding token. The file must contain at least $T+1$ bytes. Allocate 1.25 hours to preparation and the loader and 0.75 to target alignment, short-file failures, and split checks.

### Chapter 14 — AdamW, accumulation, and gradient clipping

**Prerequisites:** Chapter 12's unique parameter directory and Chapter 13's batches.

**Sections.** Track first and second moments; apply bias corrections using update count; separate weight decay from the loss gradient; accumulate a token-weighted mean across microbatches; clip the global gradient norm after accumulation; define schedule state and the precise point at which gradients are cleared.

**Build and run.** Add `src/optim.c` and `tests/test_optim.c`. Verify scalar updates against hand-calculated values. Compare one combined batch with equivalent microbatches before and after an update. A repeated embedding/output parameter must appear once in optimizer iteration. Matrix weights, including the shared table, receive the configured decay once; RMSNorm scales do not. Detect non-finite gradients before modifying optimizer state.

**Outcome and effort.** A checked update step whose meaning does not depend accidentally on the number of accumulation calls. The initial training setting uses four microbatches per update: $2\times128\times4=1024$ target tokens. Allocate 2.50 hours to state/update logic and 1.50 to normalization, clipping, and reference checks. Technical basis: [AdamW Algorithm 2][adamw]; the implementation states its learning-rate/decay convention explicitly.

### Chapter 15 — Train, evaluate, checkpoint, and resume

**Prerequisites:** Chapters 12–14, including a complete backward path and checked file I/O.

**Sections.** Integrate a complete update loop; distinguish training and held-out evaluation; first overfit one fixed batch, then run a text pilot; implement a simple warmup/decay schedule; define a versioned little-endian checkpoint rather than dumping C structs; serialize all state required for the next update; validate before replacing live state; handle partial reads/writes and failed saves.

**Build and run.** Add `app/train.c`, `src/checkpoint.c`, `configs/byte-small.conf`, and checkpoint/training tests. Save unique weights, both optimizer moments, update count, schedule/hyperparameters, data-position/random state, and configuration. Checkpoints are written at completed update boundaries, so partially accumulated gradients are not an implicit missing dependency. Resume tests compare the next batches and updates with uninterrupted execution on the same build/backend. A truncated or incompatible file must be rejected without destroying the currently usable model.

**Outcome and effort.** Actual C text training, a held-out loss measurement, and restartable progress. Run a short pilot before selecting the longer token budget; do not promise a particular loss or throughput. Allocate 2.00 hours to the loop/evaluation, 2.00 to checkpoint/resume, and 2.00 to active debugging and tests. Unattended training is additional elapsed machine time.

### Chapter 16 — Inference workspaces and token sampling

**Prerequisites:** Chapter 15's trained weights, validated loader, and model configuration.

**Sections.** Execute a forward pass without training gradients, moments, or a full training tape; reuse layer workspace; compute only the vocabulary scores required by the caller; establish full-prefix generation as a baseline; implement greedy and temperature-based categorical selection; separate random state from model state.

**Build and run.** Add `src/inference.c`, `src/sample.c`, `app/generate.c`, and sampling/inference checks. Compare inference logits with the training forward path on fixed weights and IDs. With a prescribed uniform draw, a small categorical distribution must select a known interval. Define positive-temperature validation, greedy tie handling, and a maximum generated-token count. Write generated bytes unchanged to a file; escaped terminal display is only a presentation layer.

**Outcome and effort.** A functioning C generator before any cache optimization, with a baseline against which incremental decoding can be checked. Allocate 2.00 hours to workspace/logit selection/sampling and 1.00 to probability boundaries and end-to-end tests. Top-k, top-p, beam search, and interactive serving are not required features.

### Chapter 17 — Prefill and incremental key/value caching

**Prerequisites:** Chapters 9–11 and Chapter 16's inference-only full-prefix baseline.

**Sections.** Distinguish processing an initial prompt from appending one token; allocate per-layer shared-head caches; rotate a key once at its absolute position; attend over the committed prefix and the new token; define reset/capacity behavior; separate tentative writes from the public committed length.

**Build and run.** Add `src/cache.c` and `tests/test_cache.c`; extend inference and generation. A full-prompt prefill initializes every layer's key/value history. Each decode step consumes one token and returns logits for the following token. Compare the logits at every tested prefix against full-prefix recomputation. Test one-token prompts, independent resets, exact capacity, and an attempted append past capacity.

**Outcome and effort.** Correct stateful decoding with no recomputation of old keys and values. Validate and provision workspace before mutation; advance length only after all layers complete. A failed step can leave an unreachable tentative suffix, but must not corrupt committed entries or claim a new valid length. Allocate 3.00 hours to prefill/decode/state transitions and 1.50 to equivalence and failure checks. No sliding-window eviction or batched variable-length serving is added.

### Chapter 18 — Online normalization and tiled attention forward

**Prerequisites:** Chapter 10's materialized attention and Chapter 17's working inference path.

**Sections.** Maintain a running maximum, scaled normalization sum, and weighted-value numerator; rescale old contributions when the maximum changes; merge a score tile into that state; generate scores from query/key tiles rather than accept a prebuilt square matrix; handle causal diagonal tiles and incomplete tiles; save each row's log-sum-exp.

**Build and run.** Add the forward path in `src/attention_tiled.c` and `tests/test_tiled.c`. First reproduce a three-key worked row, then compare complete causal grouped-query outputs. Start with query tiles of 16 rows and key tiles of 32 columns; vary both, including tails. Skip wholly masked tiles without forming an undefined difference of negative infinities. Count live temporary storage instead of reporting only the size of one tile.

**Outcome and effort.** An inference-capable bounded-tile attention forward pass with no full score/probability allocation. Training still uses the materialized forward/backward pair until Chapter 19. Allocate 3.25 hours to online state and tiled computation and 1.75 to equivalence, masking, and allocation checks. Technical basis: [online normalization, Section 3][online], and [FlashAttention Section 3.1][flash].

### Chapter 19 — Recomputed tiled backward without a quadratic tape

**Prerequisites:** Chapter 11's materialized derivatives, Chapter 18's tiled outputs/statistics, and Chapter 15's trainer.

**Sections.** Reconstruct probabilities from a score tile and saved row normalization; obtain the softmax reduction from the output/output-gradient dot product; accumulate value, query, and key gradients in bounded tiles; retain the score-scale factor; combine contributions from shared heads; remove the quadratic attention tape from the tiled training mode.

**Build and run.** Complete `src/attention_tiled.c`, extend `tests/test_tiled.c`, and connect the paired tiled backend to training. Compare all three input gradients with the materialized reference, then use tiny finite differences and a whole-model update comparison. Test single-token rows, lengths crossing tile boundaries, several tile sizes, and several head-sharing ratios. The tests intentionally distinguish a missing shared-head contribution from an ordinary rounding difference.

**Outcome and effort.** Exact attention training without retained full probability or score-gradient matrices. Saved query/key/value arrays and output still occupy linear storage; “no quadratic tape” does not mean “no saved activations.” Allocate 4.50 hours to the backward algorithm and tape integration and 2.50 to gradient, memory, and training regression checks. Technical basis: [FlashAttention Appendix B.2–B.4][flash].

### Chapter 20 — Import and validate pretrained weights

**Prerequisites:** Chapter 15's file format/loader, Chapter 17's inference path, and the complete operators through Chapter 19.

**Sections.** Inspect a specific model revision and tensor inventory; map named tensors to the engine's parameter directory; expand BF16 storage to FP32 without changing tensor meaning; preserve matrix orientation, tied weights, normalization epsilon, and RoPE settings; distinguish a weight-only file from a resumable training checkpoint; reject unsupported architecture features.

**Build and run.** Add `scripts/export_smollm2.py`, an imported configuration, and loader fixtures. Reuse and extend `src/checkpoint.c` rather than introduce an unrelated weight-loading system. Convert incrementally, accounting for temporary copies. Before a large load, exercise small malformed files with wrong shapes, duplicated/missing tensors, invalid offsets/counts, and truncation. The imported model allocates no gradients or Adam moments.

**Outcome and effort.** SmolLM2-135M weights owned and executable by the C inference engine, with an explicit tensor mapping and bounded loader. No claim of correct model output is made until the next chapter's independent comparison. Allocate 2.50 hours to conversion/mapping and 1.50 to validation and load-memory checks. References: [official configuration][smollm-config] and [Safetensors metadata format][safetensors].

### Chapter 21 — Tokenizer boundaries and pretrained reference agreement

**Prerequisites:** Chapter 20's imported weights and Chapter 17's cache checks.

**Sections.** Fix the tokenizer revision and special-token policy; distinguish byte-model IDs from pretrained vocabulary IDs; export prompt IDs without hidden preprocessing; align reference and C precision/layout; compare logits before samples; inspect the first divergent intermediate; compare cached and uncached continuations under identical token prefixes.

**Build and run.** Add `scripts/tokenize.py`, `scripts/reference_smollm2.py`, and comparison support in `app/compare.c`. Python emits IDs and independent expected values; C performs every model operation and selects generated IDs; Python decodes the resulting IDs afterward. The checker uses the same source weights expanded to FP32 and a compatible unfused/eager reference path where supported. Record its actual library version and settings rather than requiring the version named in historical checkpoint metadata.

**Outcome and effort.** Short pretrained continuations with measured logit error and localized intermediate checks. A matching greedy token is insufficient evidence when the rest of the vocabulary scores disagree; conversely, near-tied logits can change a sampled continuation. Allocate 2.00 hours to tokenizer/reference plumbing and 1.50 to numerical comparison and diagnosis. References: [model configuration][smollm-config] and [the Llama reference API][llama-api].

### Chapter 22 — End-to-end demonstration and measured tradeoffs

**Prerequisites:** The complete program through Chapter 21, including the trained byte-model checkpoint and imported pretrained weights.

**Sections.** Run the training/resume/generation demonstration from documented inputs; compare scalar and blocked products; measure prefill separately from one-token decode; compare materialized and tiled attention across sequence lengths; account for arrays and process memory separately; distinguish numerical error, overhead, and useful work; document the actual machine/build/run settings.

**Build and run.** Finish `app/compare.c`, a small `scripts/run_demo.sh`, and a concise `runs/README.md` describing local commands and generated result files. Measure each memory mode in a separate process so retained reference buffers do not contaminate the tiled result. Use deterministic input fixtures, warmups, repeated timings, and numerical checks before interpreting a speed difference. Keep results small: loss by processed tokens, representative output/error checks, and a compact timing/memory table.

**Outcome and effort.** A repeatable local demonstration of both model configurations and the implemented memory tradeoff. Allocate 2.00 hours to integration, 1.50 to measurements, and 0.75 to diagnosis/documented reruns. Use the remaining 0.75 hours for either an optional `src/accelerate.c` adapter comparison or a deeper C tile/shape sweep. The optional adapter uses [Apple's single-precision matrix product][accelerate] only after the C kernels and their derivatives are understood. No particular speedup is a completion condition.

## Milestones and scope boundaries

The dependency path is cumulative: foundation programs precede tensor kernels; verified operators precede the model; a working materialized trainer precedes cache and tiled optimizations; correct C inference precedes imported-model comparison. The narrower prerequisites named above identify interfaces that need rereading, not permission to assume unwritten future infrastructure.

| After | Runnable capability |
|---|---|
| Chapter 3 | Three independent C foundation programs with memory/numerical checks |
| Chapter 6 | Tensor primitives and scalar/blocked forward/backward products |
| Chapter 12 | A complete decoder with manually checked parameter gradients |
| Chapter 15 | Actual byte-text training, held-out evaluation, and checkpoint resume |
| Chapter 17 | Full-prefix, prefill, and incremental decoding with matching logits |
| Chapter 19 | Tiled attention forward/backward and a comparable model update |
| Chapter 21 | Imported pretrained inference with independent numerical comparison |
| Chapter 22 | The combined demonstration and actual timing/memory observations |

The required implementation is deliberately single-process and CPU-first. It includes neither mixed-precision training nor hand-written vector intrinsics, a general tokenizer, automatic differentiation, distributed training, serving infrastructure, quantization, low-rank adapters, beam search, or sliding-window cache eviction. This leaves room to implement and diagnose the central forward/backward, ownership, optimizer, and cache mechanisms rather than replace them with dependencies.

Correctness checks move from hand-calculated scalars to operator finite differences, complete-model directional checks, state/resume checks, and independent pretrained comparisons. Tolerances are chosen and explained with each check; one unexplained global threshold is not used for every tensor. Training loss, token counts, and generated examples are interpreted with their tokenizer and data boundaries. In particular, raw perplexities from the byte model and the pretrained tokenizer are not treated as directly comparable quality scores.

## Technical references

These sources support the named mechanisms and interfaces. The implementation chapters derive the necessary equations and supply their code; reading a linked paper is not a substitute for a missing implementation step.

**Architecture and operators.** Touvron et al., [*LLaMA*, version 1, Section 2.2][llama]; Zhang and Sennrich, [*Root Mean Square Layer Normalization*, version 1, Sections 4 and 4.2][rmsnorm]; Su et al., [*RoFormer*, version 5, Sections 3.1–3.2][rope]; Shazeer, [*GLU Variants Improve Transformer*, version 1, equations 5–6][swiglu]; Ainslie et al., [*GQA*, version 3, Section 2.2][gqa].

**Optimization and attention memory.** Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*, version 3, Algorithm 2][adamw]; Milakov and Gimelshein, [*Online normalizer calculation for softmax*, version 1, Section 3][online]; Dao et al., [*FlashAttention*, version 2, Section 3.1 and Appendix B][flash].

**Model and data interfaces.** HuggingFaceTB's [SmolLM2-135M configuration][smollm-config]; the authors' [TinyStories dataset card][tinystories]; Hugging Face's [streaming guide][streaming], [Safetensors metadata documentation][safetensors], and [Llama reference API][llama-api]. Model/data revisions and installed checker versions are recorded during asset preparation rather than assumed from a mutable web page.

**Native implementation tools.** [Apple's Xcode and command-line tool resources][apple-tools]; [Clang AddressSanitizer usage][asan]; [Clang floating-point semantics][clang-fp]; [Apple Accelerate `cblas_sgemm`][accelerate]. These are interface and feature references, not mandates to install a specific toolchain release.

[llama]: https://arxiv.org/html/2302.13971v1
[rmsnorm]: https://arxiv.org/html/1910.07467v1
[rope]: https://arxiv.org/html/2104.09864v5
[swiglu]: https://arxiv.org/html/2002.05202v1
[gqa]: https://arxiv.org/html/2305.13245v3
[adamw]: https://arxiv.org/html/1711.05101v3
[online]: https://arxiv.org/html/1805.02867v1
[flash]: https://arxiv.org/html/2205.14135v2
[smollm-config]: https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json
[tinystories]: https://huggingface.co/datasets/roneneldan/TinyStories/raw/main/README.md
[streaming]: https://huggingface.co/docs/datasets/stream
[safetensors]: https://huggingface.co/docs/safetensors/metadata_parsing
[llama-api]: https://huggingface.co/docs/transformers/main/model_doc/llama
[apple-tools]: https://developer.apple.com/xcode/resources/
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage
[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[accelerate]: https://developer.apple.com/documentation/accelerate/cblas_sgemm(_:_:_:_:_:_:_:_:_:_:_:_:_:_:)
