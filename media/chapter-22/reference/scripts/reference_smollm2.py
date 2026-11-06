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
