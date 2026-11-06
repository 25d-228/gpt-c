# Modern Language Models in C

A native C17 decoder laboratory with explicit tensor layouts, handwritten gradients, small-model training, cached generation, and exact tiled attention.

Build one engine in two configurations: a 2.9-million-parameter byte model trained from scratch and a SmolLM2-135M pretrained inference path. Implement RMSNorm, rotary positions, SwiGLU, grouped-query attention, stable loss, AdamW, checkpoints, prefill, and one-token decoding. Python prepares assets and independent checks; it does not replace C model computation.

## Read the book

[Start with the complete roadmap](https://25d-228.github.io/gpt-c/index.html). The **22 chapters and 90 implementation hours** include three runnable C/numerical foundations chapters and eight hours of debugging/integration reserve. Reading and unattended computation are separate.

<details>
<summary>Chapter index — complete lessons and their linked parts</summary>

<!-- course-links:start -->
- [Roadmap — Decoder Laboratory in C](https://25d-228.github.io/gpt-c/index.html)
- [Chapter 1 — Native C builds and a debuggable numerical program](https://25d-228.github.io/gpt-c/chapter-01.html)
- [Chapter 2 — Owned buffers, borrowed views, and failure cleanup](https://25d-228.github.io/gpt-c/chapter-02.html)
- [Chapter 3 — Floating-point arithmetic and numerical derivatives](https://25d-228.github.io/gpt-c/chapter-03.html)
- [Chapter 4 — Tensor layouts and reference matrix products](https://25d-228.github.io/gpt-c/chapter-04.html)
- [Chapter 5 · Part 1 — Embeddings, stable loss, and elementary backward passes](https://25d-228.github.io/gpt-c/chapter-05.html)
- [Chapter 5 · Part 2 — Checking gradients and numerical failures](https://25d-228.github.io/gpt-c/chapter-05-part-2.html)
- [Chapter 6 · Part 1 — Blocked C kernels and meaningful timing](https://25d-228.github.io/gpt-c/chapter-06.html)
- [Chapter 6 · Part 2 — Timing complete kernel calls](https://25d-228.github.io/gpt-c/chapter-06-part-2.html)
- [Chapter 7 · Part 1 — RMSNorm and residual gradient paths](https://25d-228.github.io/gpt-c/chapter-07.html)
- [Chapter 7 · Part 2 — Checking normalization and residual graphs](https://25d-228.github.io/gpt-c/chapter-07-part-2.html)
- [Chapter 8 · Part 1 — SwiGLU and the two-branch backward pass](https://25d-228.github.io/gpt-c/chapter-08.html)
- [Chapter 8 · Part 2 — Checking feed-forward branches and saved state](https://25d-228.github.io/gpt-c/chapter-08-part-2.html)
- [Chapter 9 · Part 1 — Rotary positions as explicit coordinate transformations](https://25d-228.github.io/gpt-c/chapter-09.html)
- [Chapter 9 · Part 2 — Checking coordinate conventions and absolute positions](https://25d-228.github.io/gpt-c/chapter-09-part-2.html)
- [Chapter 10 · Part 1 — Causal grouped-query attention, forward](https://25d-228.github.io/gpt-c/chapter-10.html)
- [Chapter 10 · Part 2 — Checking causal attention and shared heads](https://25d-228.github.io/gpt-c/chapter-10-part-2.html)
- [Chapter 11 · Part 1 — Attention backward and shared-head accumulation](https://25d-228.github.io/gpt-c/chapter-11.html)
- [Chapter 11 · Part 2 — Checking attention gradients and accumulation](https://25d-228.github.io/gpt-c/chapter-11-part-2.html)
- [Chapter 12 · Part 1 — Assemble and differentiate the complete decoder](https://25d-228.github.io/gpt-c/chapter-12.html)
- [Chapter 12 · Part 2 — Training tape and reverse scheduling](https://25d-228.github.io/gpt-c/chapter-12-part-2.html)
- [Chapter 12 · Part 3 — Checking the assembled model](https://25d-228.github.io/gpt-c/chapter-12-part-3.html)
- [Chapter 13 · Part 1 — Byte text, shifted targets, and bounded data preparation](https://25d-228.github.io/gpt-c/chapter-13.html)
- [Chapter 13 · Part 2 — Bounded preparation and end-to-end checks](https://25d-228.github.io/gpt-c/chapter-13-part-2.html)
- [Chapter 14 · Part 1 — AdamW, accumulation, and gradient clipping](https://25d-228.github.io/gpt-c/chapter-14.html)
- [Chapter 14 · Part 2 — Checking optimizer state and equivalent batches](https://25d-228.github.io/gpt-c/chapter-14-part-2.html)
- [Chapter 15 · Part 1 — Train, evaluate, checkpoint, and resume](https://25d-228.github.io/gpt-c/chapter-15.html)
- [Chapter 15 · Part 2 — Versioned checkpoints and coherent restoration](https://25d-228.github.io/gpt-c/chapter-15-part-2.html)
- [Chapter 15 · Part 3 — Running the training application](https://25d-228.github.io/gpt-c/chapter-15-part-3.html)
- [Chapter 15 · Part 4 — Checking continuation and failure recovery](https://25d-228.github.io/gpt-c/chapter-15-part-4.html)
- [Chapter 16 · Part 1 — Inference workspaces and token sampling](https://25d-228.github.io/gpt-c/chapter-16.html)
- [Chapter 16 · Part 2 — Weight-only loading and categorical selection](https://25d-228.github.io/gpt-c/chapter-16-part-2.html)
- [Chapter 16 · Part 3 — Full-prefix generation and executable checks](https://25d-228.github.io/gpt-c/chapter-16-part-3.html)
- [Chapter 17 · Part 1 — Prefill and incremental key/value caching](https://25d-228.github.io/gpt-c/chapter-17.html)
- [Chapter 17 · Part 2 — Computing and committing a cached step](https://25d-228.github.io/gpt-c/chapter-17-part-2.html)
- [Chapter 17 · Part 3 — Checking cache equivalence and failure recovery](https://25d-228.github.io/gpt-c/chapter-17-part-3.html)
- [Chapter 18 · Part 1 — Online normalization and tiled attention forward](https://25d-228.github.io/gpt-c/chapter-18.html)
- [Chapter 18 · Part 2 — Integrating bounded attention into inference](https://25d-228.github.io/gpt-c/chapter-18-part-2.html)
- [Chapter 18 · Part 3 — Checking tiled outputs, storage, and generation](https://25d-228.github.io/gpt-c/chapter-18-part-3.html)
- [Chapter 19 · Part 1 — Recomputed tiled backward without a quadratic tape](https://25d-228.github.io/gpt-c/chapter-19.html)
- [Chapter 19 · Part 2 — A linear training tape and paired dispatch](https://25d-228.github.io/gpt-c/chapter-19-part-2.html)
- [Chapter 19 · Part 3 — Training policy and versioned continuation](https://25d-228.github.io/gpt-c/chapter-19-part-3.html)
- [Chapter 19 · Part 4 — Checking tiled core and sublayer gradients](https://25d-228.github.io/gpt-c/chapter-19-part-4.html)
- [Chapter 19 · Part 5 — Whole-model, training, and memory checks](https://25d-228.github.io/gpt-c/chapter-19-part-5.html)
- [Chapter 20 · Part 1 — Import and validate pretrained weights](https://25d-228.github.io/gpt-c/chapter-20.html)
- [Chapter 20 · Part 2 — Bounded C loading and imported position limits](https://25d-228.github.io/gpt-c/chapter-20-part-2.html)
- [Chapter 20 · Part 3 — Executable import fixtures and the C inspection path](https://25d-228.github.io/gpt-c/chapter-20-part-3.html)
- [Chapter 20 · Part 4 — Cross-language validation and separate memory measurements](https://25d-228.github.io/gpt-c/chapter-20-part-4.html)
- [Chapter 21 · Part 1 — Tokenizer boundaries and pretrained reference agreement](https://25d-228.github.io/gpt-c/chapter-21.html)
- [Chapter 21 · Part 2 — Observing C execution and selecting vocabulary IDs](https://25d-228.github.io/gpt-c/chapter-21-part-2.html)
- [Chapter 21 · Part 3 — Independent eager equations and numerical disagreement](https://25d-228.github.io/gpt-c/chapter-21-part-3.html)
- [Chapter 21 · Part 4 — Executable checks and the real-asset workflow](https://25d-228.github.io/gpt-c/chapter-21-part-4.html)
- [Chapter 22 · Part 1 — End-to-end demonstration and measured tradeoffs](https://25d-228.github.io/gpt-c/chapter-22.html)
- [Chapter 22 · Part 2 — Measure forward/backward pairs and test the harness](https://25d-228.github.io/gpt-c/chapter-22-part-2.html)
- [Chapter 22 · Part 3 — A repeatable training, restart, and generation run](https://25d-228.github.io/gpt-c/chapter-22-part-3.html)
- [Chapter 22 · Part 4 — Interpret the measurements and complete the demonstration](https://25d-228.github.io/gpt-c/chapter-22-part-4.html)
<!-- course-links:end -->

</details>

The final demonstration trains a bounded text model, resumes the same trajectory, generates in C, compares full-prefix/cached logits and materialized/tiled gradients, and separates prefill, decoding, live arrays, and process memory. Executed byte-model and synthetic-reference results are documented with their limits. Genuine pretrained/tokenizer agreement remains unexecuted in the provided environment; its complete local workflow is included rather than reported as a passed test.

## Workspace

Target: a native M4 Mac mini with 16 GB memory, a compatible C17 compiler, `make`, and Python 3 for assets/checks. No specific macOS release, GPU backend, or exact compiler version is required. Recorded Linux measurements are not M4 predictions.

`main` is this concise entry point. `gh-pages` holds the full book and generated website. Implement the application on persistent **`work/decoder-lab`**; keep datasets, weights, build outputs, and local runs out of documentation branches. See [AGENTS.md](AGENTS.md) for repository working rules.
