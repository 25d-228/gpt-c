# Chapter 21, Part 3 — Independent eager equations and numerical disagreement

[Part 2](https://25d-228.github.io/gpt-c/chapter-21-part-2.html) emits C intermediate tensors and fixed-prefix logits. This part builds an independent CPU FP32 equation reference and a comparator. It does not call a Python text-generation pipeline, perform training, or choose continuation tokens.

## What is independent, and what is shared?

The C path reads Chapter 20's converted `weights.dlw`. The reference reads the **original** `model.safetensors`, asks the Safetensors/PyTorch interface for each named tensor, and expands BF16 to FP32 through the library conversion. Its arithmetic uses explicit eager tensor operations. Thus the reference does not inherit a bug in our manual BF16 bit expansion, C weight offsets, C matrix loops or cache state machine.

We share the source-configuration validator and the checked ID/trace format. This is useful agreement about what experiment is being run, not complete independence from every possible specification error. The source tensor inventory is built again by name and shape. Hand calculations and the earlier operator tests remain necessary: two implementations can consistently follow the same mistaken convention.

This is an **explicit PyTorch equation reference**, not a claim to have executed `LlamaForCausalLM` or the model provider's generation program. The supported computation is the bias-free, tied-output, RMSNorm/rotary/SwiGLU grouped-query profile established in Chapter 20 and the [Llama model documentation](https://huggingface.co/docs/transformers/model_doc/llama). Unsupported configuration features fail validation rather than being silently approximated.

Create this optional dependency file. Compatible maintained releases are acceptable; the numerical report records the versions actually used. The old version string inside a pretrained configuration is metadata, not a mandatory installation gate.

```text title="requirements-reference.txt"
# Install compatible maintained releases for your Python/platform.
# CPU FP32 reference uses torch + safetensors; tokenization uses transformers.
# No model generation or training is delegated to these Python packages.
torch
safetensors
transformers
```

Use a separate environment for these independent checks:

```sh
python3 -m venv .venv-reference
. .venv-reference/bin/activate
python -m pip install -r requirements-reference.txt
```

On the M4, use a native arm64 Python environment. This reference deliberately runs on **CPU**, not Metal/MPS or CUDA, so it isolates FP32 eager equations from device-specific reduced-precision or fused-attention choices. C and Python runs are separate processes; do not keep an unnecessary C process holding the full weight set while the reference loads another copy. The reference itself retains expanded FP32 weights and small-prefix temporaries. Its memory behavior must still be measured locally rather than equated with a file's byte count.

## Load original tensors with an independent conversion

Create `scripts/reference_smollm2.py` from the next four listings. `safe_open` supplies named tensors without loading Python pickle code; see its [documented PyTorch interface](https://huggingface.co/docs/safetensors/en/api/torch). The loader verifies the entire tensor inventory, every shape, the allowed source dtypes, and finite expanded values.

Weights are stored as `[output_features, input_features]`, matching [PyTorch's linear operation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.linear.html). The embedding table also supplies the final projection; no independent `lm_head` weight is invented. The source file and configuration have explicit size limits. The small-fixture bypass requires an explicit flag and is never automatically selected because the real assets are absent.

```python title="scripts/reference_smollm2.py"
#!/usr/bin/env python3
"""Independent CPU/FP32 eager equation reference. Never selects generated IDs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

import torch
import torch.nn.functional as functional
import safetensors
from safetensors import safe_open

from export_smollm2 import MODEL, REVISION, configuration, json_object
from probe_io import exclusive_file, read_ids, trace_header, write_json


def load_source(directory, *, fixture=False):
    directory = Path(directory)
    if (directory/"config.json").stat().st_size > 65536 or \
       (directory/"model.safetensors").stat().st_size > 300*1024*1024:
        raise ValueError("source asset exceeds budget")
    raw = json_object((directory/"config.json").read_bytes())
    c = configuration(raw, fixture=fixture)  # Interpretation only, not conversion or computation.
    v, layers, width, hq, hk, hidden = c["dims"]
    kv = hk * (width//hq)
    expected = {"model.embed_tokens.weight": (v, width), "model.norm.weight": (width,)}
    for i in range(layers):
        prefix = f"model.layers.{i}."
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            expected[prefix+name] = (width,)
        for name, shape in (("q_proj", (width, width)), ("k_proj", (kv, width)),
                            ("v_proj", (kv, width)), ("o_proj", (width, width))):
            expected[prefix+"self_attn."+name+".weight"] = shape
        for name, shape in (("gate_proj", (hidden, width)), ("up_proj", (hidden, width)),
                            ("down_proj", (width, hidden))):
            expected[prefix+"mlp."+name+".weight"] = shape
    weights = {}
    with safe_open(str(directory/"model.safetensors"), framework="pt", device="cpu") as source:
        if set(source.keys()) != set(expected):
            raise ValueError("unexpected source tensor inventory")
        for name, shape in expected.items():
            tensor = source.get_tensor(name)
            if tuple(tensor.shape) != shape or tensor.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"unexpected tensor shape/dtype: {name}")
            # This library conversion is independent of Chapter 20's bit expansion.
            value = tensor.to(dtype=torch.float32, device="cpu", copy=True)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"nonfinite source weight: {name}")
            weights[name] = value
    return c, weights


```

## Visible equations rather than an attention convenience function

For row $x\in\mathbb{R}^D$, scale vector $g$, and epsilon $\epsilon$, the normalization below computes

$$
y_j=x_j g_j\left(\frac{1}{D}\sum_{k=0}^{D-1}x_k^2+\epsilon\right)^{-1/2}.
$$

`keepdim=True` retains the last dimension as size one so that the inverse root broadcasts across each row. All these operations use FP32 tensors. Our C implementation's wider reductions will therefore not generally agree bit-for-bit.

The rotary input is shaped `[time, heads, head_width]`. `[..., :half]` and `[..., half:]` select the two coordinate halves, not alternating coordinates. A frequency row broadcasts across heads. Absolute positions start at zero because this reference recomputes the complete supplied prefix. We will test cached nonzero continuation positions by feeding that same complete prefix to the reference, not by restarting the C cache's rotary positions.

```python title="scripts/reference_smollm2.py"
def normalize(x, scale, epsilon):
    return (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)) * scale


def rotate(x, theta):
    time, heads, width = x.shape
    half = width//2
    frequency = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32)*2.0/width))
    angles = torch.arange(time, dtype=torch.float32)[:, None] * frequency[None, :]
    cosine, sine = angles.cos()[:, None, :], angles.sin()[:, None, :]
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left*cosine - right*sine, right*cosine + left*sine), dim=-1)


```

Let $T$ be prefix length, $H_q$ query-head count, $H_{kv}$ shared key/value-head count, and $d=D/H_q$ the head width. The projections first produce `Q[T,Hq,d]` and `K,V[T,Hkv,d]`. The independent reference physically repeats key/value heads for simplicity; C does not. With four query heads and two shared heads, `repeat_interleave(2)` gives the mapping `[K0,K0,K1,K1]`, not `[K0,K1,K0,K1]`.

After transposition, scores have shape `[Hq,T,T]`. The strictly upper-triangular mask sets future positions to negative infinity before row softmax. The diagonal remains visible, so no causal row has all positions masked. Attention context is transposed back and flattened into `[T,D]` before the output projection.

The feed-forward branch uses the documented [SiLU operation](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.silu.html), `x * sigmoid(x)`, followed by multiplication with the independent value projection. Both residual additions are explicit. There is no fused scaled-dot-product attention call, hidden cache, generation API, dropout, or autograd computation.

```python title="scripts/reference_smollm2.py"
def forward(c, weights, ids, observe=None):
    """Single dense causal sequence, no padding, cache, dropout, autograd, or sampling."""
    _, layers, width, hq, hk, _ = c["dims"]
    time, head_dim = len(ids), width//hq
    if not 1 <= time <= c["positions"] or any(not 0 <= x < c["dims"][0] for x in ids):
        raise ValueError("invalid reference prefix")
    def emit(name, index, value):
        if observe is not None:
            observe(name, index, value.reshape(time if name not in ("final_norm", "logits") else 1, -1))
    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        table = weights["model.embed_tokens.weight"]
        x = functional.embedding(torch.tensor(ids, dtype=torch.long), table)
        emit("embedding", 0, x)
        for layer in range(layers):
            prefix = f"model.layers.{layer}."
            u = normalize(x, weights[prefix+"input_layernorm.weight"], c["epsilon"])
            emit("attn_norm", layer, u)
            q = functional.linear(u, weights[prefix+"self_attn.q_proj.weight"]).reshape(time, hq, head_dim)
            k = functional.linear(u, weights[prefix+"self_attn.k_proj.weight"]).reshape(time, hk, head_dim)
            v = functional.linear(u, weights[prefix+"self_attn.v_proj.weight"]).reshape(time, hk, head_dim)
            emit("query", layer, q); emit("key", layer, k); emit("value", layer, v)
            q, k = rotate(q, c["theta"]), rotate(k, c["theta"])
            emit("rotated_query", layer, q); emit("rotated_key", layer, k)
            # Explicitly expand KV heads here: a different implementation from C's index map.
            keys = k.repeat_interleave(hq//hk, dim=1).transpose(0, 1)
            values = v.repeat_interleave(hq//hk, dim=1).transpose(0, 1)
            scores = (q.transpose(0, 1) @ keys.transpose(1, 2)) / math.sqrt(head_dim)
            mask = torch.ones(time, time, dtype=torch.bool).triu(1)
            probabilities = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
            context = (probabilities @ values).transpose(0, 1).contiguous().reshape(time, width)
            emit("context", layer, context)
            branch = functional.linear(context, weights[prefix+"self_attn.o_proj.weight"])
            emit("attention", layer, branch)
            middle = x + branch
            emit("middle", layer, middle)
            u = normalize(middle, weights[prefix+"post_attention_layernorm.weight"], c["epsilon"])
            emit("ffn_norm", layer, u)
            gate = functional.linear(u, weights[prefix+"mlp.gate_proj.weight"])
            up = functional.linear(u, weights[prefix+"mlp.up_proj.weight"])
            hidden = functional.silu(gate) * up
            emit("gate", layer, gate); emit("up", layer, up); emit("ffn_hidden", layer, hidden)
            branch = functional.linear(hidden, weights[prefix+"mlp.down_proj.weight"])
            emit("ffn", layer, branch)
            x = middle + branch
            emit("hidden", layer, x)
        norm = normalize(x[-1:], weights["model.norm.weight"], c["epsilon"])
        emit("final_norm", layers, norm)
        logits = functional.linear(norm, table)
        emit("logits", layers, logits)
        return logits


```

`torch.inference_mode()` disables autograd bookkeeping; it is not a substitute for choosing evaluation behavior in a stateful neural-network module. Here there is no such module and no training-only operator. The nested CPU autocast context explicitly disables automatic dtype changes, even if a caller happened to enable them outside this function. The [inference-mode documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html) describes that distinction.

The export layer fixes one CPU thread, disables the MKLDNN path, and records actual versions and settings. `set_float32_matmul_precision("highest")` documents the desired precision policy; PyTorch's [precision-setting documentation](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html) notes that the setting currently affects CUDA operations. It does not establish a CPU bitwise-equivalence guarantee.

For intermediate mode, the same 16-ID diagnostic bound applies as in C. For prefix mode, the reference evaluates every supplied prefix from `start` onward and emits all vocabulary logits for its last row. It never selects a token: the transcript is an input, including when C produced that transcript in a previous run.

```python title="scripts/reference_smollm2.py"
def export_reference(directory, ids_file, output, *, mode="trace", start=1, fixture=False):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = False
    vocab, ids = read_ids(ids_file)
    if mode not in ("trace", "prefix") or not 1 <= start <= len(ids):
        raise ValueError("invalid reference mode/start")
    if mode == "trace" and len(ids) > 16:
        raise ValueError("trace limited to 16 input tokens")
    c, weights = load_source(directory, fixture=fixture)
    if vocab != c["dims"][0] or len(ids) > c["positions"]:
        raise ValueError("token vocabulary or position limit mismatch")
    settings = dict(engine="torch-eager-equations", model="synthetic fixture" if fixture else MODEL,
        expected_revision=None if fixture else REVISION, source=str(directory),
        acquisition="caller-supplied local files", python=platform.python_version(),
        torch=torch.__version__, safetensors=safetensors.__version__, device="cpu",
        dtype="float32 expanded from source", threads=torch.get_num_threads(),
        fused_attention=False, compile=False, autocast=False, mkldnn=False,
        float32_matmul_precision=torch.get_float32_matmul_precision())
    count, total = 0, 0
    with exclusive_file(output) as file:
        write_json(file, trace_header(mode, ids, c["dims"], len(ids) if mode == "trace" else start, settings,
                                      epsilon=c["epsilon"], theta=c["theta"]))
        def emit(name, index, value):
            nonlocal count, total
            if value.ndim != 2 or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid reference intermediate")
            total += value.numel()
            if total > 8*1024*1024:
                raise ValueError("trace value budget exceeded")
            write_json(file, dict(kind="tensor", name=name, index=index,
                rows=value.shape[0], cols=value.shape[1], values=value.flatten().tolist()))
            count += 1
        if mode == "trace":
            forward(c, weights, ids, emit)
        else:
            for n in range(start, len(ids)+1):
                emit("logits", n, forward(c, weights, ids[:n]))
        write_json(file, dict(kind="end", records=count))
    return dict(records=count, values=total, settings=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("ids", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("--mode", choices=("trace", "prefix"), default="trace")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--fixture", action="store_true", help="accept a tiny synthetic profile, NOT real pretrained validation")
    args = parser.parse_args()
    try:
        report = export_reference(args.source, args.ids, args.output, mode=args.mode,
                                  start=args.start, fixture=args.fixture)
        print(json.dumps(report, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"reference failed: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
```

## A numerical agreement criterion is not a string comparison

For a C value $c_i$ and reference value $r_i$, choose a positive absolute tolerance $a$ and a nonnegative relative tolerance $b$. A coordinate passes when

$$
|c_i-r_i|\le a+b|r_i|.
$$

Absolute tolerance handles values near zero; relative tolerance scales with reference magnitude. The criterion is asymmetric by design: $r_i$ is the declared reference. The report additionally gives maximum absolute error, root-mean-square error, and maximum normalized error ratio:

$$
E_{\max}=\max_i|c_i-r_i|,\qquad
E_{\mathrm{rms}}=\sqrt{\frac{1}{n}\sum_i(c_i-r_i)^2},\qquad
R_{\max}=\max_i\frac{|c_i-r_i|}{a+b|r_i|}.
$$

Here $n$ is the number of compared tensor coordinates. A value of $R_{\max}>1$ fails the selected criterion. Tolerances must be reported, not silently increased until a broken implementation passes.

Compare **every logit**, not only the largest one. Changing an unselected logit by 0.25 can leave the greedy token unchanged while badly changing the distribution. Conversely, a tiny allowed perturbation can reverse a nearly tied pair. Numerical acceptance and greedy-token agreement are separate fields in the report.

For a particular logit row, let the reference's largest and second-largest values be $r_{(1)}$ and $r_{(2)}$, and define gap $\Delta=r_{(1)}-r_{(2)}$. Let $E$ be the maximum actual absolute error across this row. If

$$
\Delta>2E,
$$

the reference winner must remain the winner: its C value is at least $r_{(1)}-E$, while every competitor is at most $r_{(2)}+E$. This is a certificate based on observed error, not a promise for a different prompt.

For example, reference logits `[2, 1.999, 0]` and C logits `[1.9996, 1.9994, 0]` have gap 0.001 and maximum error 0.0004; the gap exceeds twice that error. But reference `[1, 1.0000001]` and C `[1.0000001, 1]` can pass an absolute tolerance of $10^{-6}$ while selecting different IDs. In both engines, exact equal maxima select the lowest ID.

Once independently generated transcripts choose different tokens, their following logits describe different conditional inputs. Comparing those subsequent rows as though they were the same experiment is invalid. Replay one fixed transcript through both engines first; only then discuss the differing token choices and their margins.

## A streaming comparator that also rejects incomplete evidence

Create `scripts/compare_traces.py`. Headers must match in IDs, mode, geometry, start length, epsilon and rotary base. Record names, indices and shapes must match in order. Engine settings may differ and are preserved in the report. The iterator walks both files to completion even when an earlier value fails, ensuring that truncated traces do not pass by matching a valid prefix.

The first failed coordinate is reported along with its stage. The maximum normalized ratio can overflow for an absurdly small user tolerance; it is emitted as JSON `null` in that case while the failed-coordinate count still records failure. Nonfinite tensor inputs are rejected before comparison and are never accepted through this reporting convention.

```python title="scripts/compare_traces.py"
#!/usr/bin/env python3
"""Compare fixed-prefix numerical traces; never retokenize or generate."""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sys

from probe_io import trace_records


def compare(actual, reference, *, atol, rtol):
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol <= 0 or rtol < 0:
        raise ValueError("require finite atol>0, rtol>=0")
    left, right = trace_records(actual), trace_records(reference)
    a, b = next(left), next(right)
    for key in ("version", "mode", "ids", "geometry", "start", "epsilon", "theta"):
        if a[key] != b[key]:
            raise ValueError(f"incomparable headers: {key}")
    values = records = failed = greedy_rows = same_greedy = certified = 0
    largest = squared = scaled = 0.0
    first = None
    greedy_details = []
    for x, y in itertools.zip_longest(left, right):
        if x is None or y is None:
            raise ValueError("missing or extra tensor record")
        for key in ("name", "index", "rows", "cols"):
            if x[key] != y[key]:
                raise ValueError(f"different tensor identity/shape: {key}")
        row_errors = [0.0] * x["rows"]
        for i, (c, r) in enumerate(zip(x["values"], y["values"])):
            error = abs(c-r)
            tolerance = atol + rtol*abs(r)
            largest = max(largest, error)
            scaled = max(scaled, error/tolerance)
            squared += error*error
            row_errors[i//x["cols"]] = max(row_errors[i//x["cols"]], error)
            if error > tolerance:
                failed += 1
                if first is None:
                    first = dict(name=x["name"], index=x["index"], row=i//x["cols"],
                        column=i % x["cols"], actual=c, reference=r, absolute=error, tolerance=tolerance)
        if x["name"] == "logits":
            for row in range(x["rows"]):
                start, stop = row*x["cols"], (row+1)*x["cols"]
                cr, rr = x["values"][start:stop], y["values"][start:stop]
                top = heapq.nlargest(2, range(len(rr)), key=lambda k: (rr[k], -k))
                ci = max(range(len(cr)), key=lambda k: (cr[k], -k))
                gap = rr[top[0]] - rr[top[1]]
                greedy_rows += 1
                same_greedy += ci == top[0]
                certified += gap > 2*row_errors[row]
                greedy_details.append(dict(index=x["index"], row=row, actual_id=ci,
                    reference_id=top[0], reference_gap=gap, max_abs=row_errors[row],
                    margin_certifies=gap > 2*row_errors[row]))
        values += len(x["values"]); records += 1
    if values == 0 or greedy_rows == 0:
        raise ValueError("trace lacks numerical logits")
    return dict(passed=failed == 0, records=records, values=values,
        failed_values=failed, max_abs=largest, rms=math.sqrt(squared/values),
        max_scaled_error=scaled if math.isfinite(scaled) else None, first_failure=first, atol=atol, rtol=rtol,
        greedy_matches=same_greedy, greedy_rows=greedy_rows, certified_rows=certified,
        greedy=greedy_details, actual_settings=a["settings"], reference_settings=b["settings"])


```

The CLI requires explicit tolerances. Status 0 means numerical acceptance; status 1 means a complete comparable experiment failed the tolerance; status 2 means the files or arguments do not define a valid comparison. A near-tied greedy disagreement can therefore appear in a status-0 report. Inspect both results rather than redefining the numeric check around a desired string.

```python title="scripts/compare_traces.py"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual"); parser.add_argument("reference")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.actual, args.reference, atol=args.atol, rtol=args.rtol)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError, OverflowError, StopIteration) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
```

## Diagnose the earliest divergence

The trace order follows data dependencies. Start at the first failed stage, not at a visually impressive final logit error.

| First disagreement | Check before changing tolerances |
|---|---|
| `embedding` | Exact IDs, vocabulary, source revision, BF16 conversion and embedding-row offsets |
| `attn_norm` / `ffn_norm` | Epsilon, scale tensor, row width and reduction arithmetic |
| `query`, `key`, `value` | Named projection weights, transpose orientation and head widths |
| `rotated_query` / `rotated_key` | Split-half pairing, absolute positions, base and frequency arithmetic |
| `context` | Causal mask, head-sharing map, score scale, softmax and value accumulation |
| `attention`, `ffn` | Output projection orientation and intermediate branch values |
| `middle`, `hidden` | Residual source, lifetime and accidental overwrite |
| `final_norm`, `logits` | Requested row, final scale, tied output table and output projection |
| Only cached prefixes | Prefill length, committed length, append position and per-layer key/value state |

A small discrepancy appearing after many operations may reflect a different legitimate reduction order. A large discrepancy first appearing exactly at rotation suggests a convention or position error. These are hypotheses to test against the intermediate values, not automatic verdicts based on magnitude alone.

Continue to [Part 4 — Executable checks and the real-asset workflow](https://25d-228.github.io/gpt-c/chapter-21-part-4.html). It gives runnable fixtures, measured agreement for those fixtures, and a separate path for the genuine assets.
