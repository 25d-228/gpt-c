# Chapter 21, Part 4 — Executable checks and the real-asset workflow

The earlier parts defined [token boundaries](https://25d-228.github.io/gpt-c/chapter-21.html), [C execution](https://25d-228.github.io/gpt-c/chapter-21-part-2.html), and [independent comparison](https://25d-228.github.io/gpt-c/chapter-21-part-3.html). This part verifies them together. Start from the Chapter 20 source tree and apply all additions and replacements in order, or use the complete [Chapter 21 companion source](https://25d-228.github.io/gpt-c/media/chapter-21/reference/Makefile).

Keep two claims separate throughout: a synthetic fixture can establish that file boundaries, tensor mapping and reference equations agree for that fixture; only a run on the genuine selected assets can establish pretrained-model agreement. The tests below do not relabel synthetic weights as pretrained weights.

## Check the observer's lifetime and failure behavior in C

Create `tests/test_trace.c`. The test constructs a small model, publishes immutable weights, and compares ordinary and observed forward calls in all four full/tiled and scalar/blocked combinations. It copies the last observed logits while their view is live; it never retains workspace addresses.

Injected callback failures occur at the first record, within blocks, at final normalization and after the final logits. A subsequent clean call must reproduce the ordinary output exactly. An invalid token ID is a preflight error: no callback runs and output sentinels remain unchanged. The `CHECK` macro jumps to the single cleanup path so a failed assertion does not leak the model or its borrowed workspaces.

```c title="tests/test_trace.c"
#include "dl/inference.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #condition); goto done; \
} } while (0)
#define OK(call) CHECK((call) == DL_OK)

typedef struct {
    size_t calls, fail_at, layers, time, width, vocab;
    int invalid;
    float last[16];
} Observer;

static DlStatus observe(void *context, const char *name, size_t index, DlConstMatrix a)
{
    Observer *o = context;
    ++o->calls;
    size_t end = 0;
    if (dl_matrix_end(a, &end) != DL_OK) o->invalid = 1;
    if (strcmp(name, "embedding") == 0 &&
        (index != 0 || a.rows != o->time || a.cols != o->width)) o->invalid = 1;
    if (strcmp(name, "logits") == 0) {
        if (index != o->layers || a.rows != 1 || a.cols != o->vocab) o->invalid = 1;
        for (size_t j = 0; j < a.cols && j < 16; ++j) o->last[j] = a.base[a.offset+j];
    }
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (!isfinite(a.base[a.offset+i*a.ld+j])) o->invalid = 1;
    return o->calls == o->fail_at ? DL_STATE : DL_OK;
}

static int check(DlBackend backend, int tiled)
{
    int result = 0;
    DlModel model = {0};
    DlWeights weights = {0};
    DlInference workspace = {0};
    DlModelConfig c = {.vocab=16, .layers=2, .width=8, .query_heads=4,
        .kv_heads=2, .ffn_width=12, .epsilon=1e-5f, .rope_theta=10000,
        .init_std=0.125f, .seed=42};
    size_t ids[] = {0, 3, 9, 3, 15};
    float baseline[16], actual[16];
    DlMatrix expected = {baseline, 16, 0, 1, 16, 16};
    DlMatrix output = {actual, 16, 0, 1, 16, 16};
    OK(dl_model_create(&model, c));
    OK(dl_weights_from_model(&weights, &model, 1024*1024));
    if (tiled) { OK(dl_inference_create_tiled(&workspace, &weights, 5, 2, 3, 1024*1024)); }
    else { OK(dl_inference_create(&workspace, &weights, 5, 1024*1024)); }
    OK(dl_inference_forward(&workspace, backend, ids, 5, 4, expected));
    Observer o = {.layers=2, .time=5, .width=8, .vocab=16};
    OK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output, observe, &o));
    CHECK(o.calls == 33 && !o.invalid);
    CHECK(memcmp(baseline, actual, sizeof baseline) == 0);
    CHECK(memcmp(o.last, actual, sizeof actual) == 0);
    const size_t faults[] = {1, 2, 17, 32, 33};
    for (size_t i = 0; i < sizeof faults/sizeof faults[0]; ++i) {
        o.calls=0; o.fail_at=faults[i];
        CHECK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output,
                                         observe, &o) == DL_STATE);
        CHECK(o.calls == o.fail_at);
        /* A failed sink invalidates the result, even after final logits were written. */
        OK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output, NULL, NULL));
        CHECK(memcmp(baseline, actual, sizeof baseline) == 0);
    }
    o.calls=0; ids[4]=16;
    for (size_t i=0; i<16; ++i) actual[i]=77.0f;
    CHECK(dl_inference_forward_trace(&workspace, backend, ids, 5, 4, output,
                                     observe, &o) == DL_RANGE);
    CHECK(o.calls == 0);
    for (size_t i=0; i<16; ++i) CHECK(actual[i] == 77.0f);
    result=1;
done:
    dl_inference_destroy(&workspace);
    dl_weights_destroy(&weights);
    dl_model_destroy(&model);
    return result;
}

int main(void)
{
    for (int backend=0; backend<2; ++backend)
        for (int tiled=0; tiled<2; ++tiled)
            if (!check((DlBackend)backend, tiled)) return 1;
    puts("trace: four dispatch cases, callback failures and clean retries pass");
    return 0;
}
```

Append `test_trace` to the existing `PROGRAMS` assignment in the Makefile. The existing generic `build/<mode>/%` rules compile it; no new compilation rule is needed. Add this optional Python-reference target at the end of the Makefile:

```makefile title="Makefile: reference target"
.PHONY: test-reference
test-reference: build/release/compare
	python3 tests/test_reference.py build/release/compare
```

The base C build does not import Python numerical libraries. Running `test-reference` requires the optional environment from Part 3. This keeps a missing tokenizer package from obstructing ordinary C compilation.

## The independent integration checks

Create `tests/test_reference.py` from the following consecutive listings. It imports `tests/import_fixture.py` from Chapter 20; that complete named prerequisite creates original tiny Safetensors bytes and a matching source configuration. `scripts/export_smollm2.py` converts them for C, while `reference_smollm2.py` reads the original source through a separate tensor library.

`FakeTokenizer` is intentionally **not** a byte-pair tokenizer or an imitation of the model's vocabulary. It is a controllable adapter that records the arguments passed at the boundary. This proves that our script preserves UTF-8 text and requests the intended options; it does not prove that a real tokenizer library assigns the correct IDs. Real assets and the real library remain necessary for that separate check.

`setUpClass` constructs one disposable fixture shared by the tests. `run_c` launches the actual compiled executable, checks its exit status and preserves its stdout/stderr for diagnosis. `alter_trace` makes an intentionally damaged copy rather than corrupting the reference used by other cases.

```python title="tests/test_reference.py"
#!/usr/bin/env python3
"""Offline protocol, tokenizer-policy and independent eager-reference checks."""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from import_fixture import write_source, fixture_source, raw_config
import export_smollm2 as exporter
import probe_io
from compare_traces import compare
import reference_smollm2 as reference
import tokenizer_boundary as tokens
C_APP = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


class FakeTokenizer:
    """Policy fixture, not SmolLM2's tokenizer or a BPE implementation."""
    bos_token_id = eos_token_id = 0
    all_special_ids = [0]
    is_fast = True
    def __len__(self): return 49152
    def encode(self, text, **kwargs):
        if kwargs != dict(add_special_tokens=False, padding=False, truncation=False):
            raise AssertionError(kwargs)
        self.received = text
        return [0] if text == "<|endoftext|>" else [x+1 for x in text.encode("utf-8")]
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        if clean_up_tokenization_spaces:
            raise AssertionError("cleanup was requested")
        return self._decode(ids, skip_special_tokens)
    @staticmethod
    def _decode(ids, skip):
        # One decode of the complete byte sequence, not one Unicode decode per ID.
        parts, buffer = [], bytearray()
        for x in ids:
            if x == 0:
                parts.append(buffer.decode("utf-8")); buffer.clear()
                if not skip: parts.append("<|endoftext|>")
            else: buffer.append(x-1)
        parts.append(buffer.decode("utf-8"))
        return "".join(parts)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = write_source(cls.root/"source")
        cls.converted = cls.root/"converted"
        exporter.export(cls.source, cls.converted, fixture=True)
        cls.ids = cls.root/"prompt.ids"
        probe_io.write_ids(cls.ids, 16, [0, 3, 9, 3, 15])
        cls.expected = cls.root/"expected.jsonl"
        reference.export_reference(cls.source, cls.ids, cls.expected, fixture=True)
        cls.serial = 0
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def new(self, suffix=""):
        type(self).serial += 1
        return self.root/f"output-{self.serial}{suffix}"
    def run_c(self, operation, strategy="full", backend="ref", extra=(), ids=None, output=None, ok=True):
        if C_APP is None: self.skipTest("supply the built compare executable")
        output = output or self.new()
        command = [str(C_APP), "pretrained", operation, str(self.converted/"model.cfg"),
                   str(self.converted/"weights.dlw"), str(ids or self.ids), str(output),
                   strategy, backend, *map(str, extra)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode == 0, ok, run.stdout+run.stderr)
        return output, run
    def alter_trace(self, change):
        rows = [json.loads(line) for line in self.expected.read_text().splitlines()]
        change(rows)
        path = self.new(".jsonl")
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        return path

```

The first group tests the binary ID boundary, exact text policy and pinned acquisition arguments. It includes ID 256 and ID 49,151, which a byte-oriented interface would corrupt. A modified vocabulary can be a valid token file while still being invalid for this particular loaded model; the test distinguishes those cases.

The fetch test substitutes an in-memory HTTP response and checks that all three requested URLs include the selected revision. The loader test substitutes the tokenizer adapter and inspects its keyword arguments. These are explicitly offline interface checks, not successful network downloads or executions of Transformers' tokenizer.

```python title="tests/test_reference.py"
    def test_token_wire_roundtrip_and_rejections(self):
        for vocab, ids in ((16,[0]), (49152,[0,255,256,49151]), (16,[15]*256)):
            path=self.new(); probe_io.write_ids(path,vocab,ids)
            self.assertEqual(probe_io.read_ids(path),(vocab,ids))
        raw = self.ids.read_bytes()
        bad = [raw[:n] for n in (0,7,8,23,24,len(raw)-1)] + [raw+b"x"]
        for offset, value, fmt in ((0,b"NOTIDS01","8s"),(8,256,"Q"),(16,0,"Q"),
                                  (16,257,"Q"),(24,16,"I"),(24,2**32-1,"I")):
            data=bytearray(raw); struct.pack_into("<"+fmt,data,offset,value); bad.append(bytes(data))
        for data in bad:
            path=self.new(); path.write_bytes(data)
            # A changed vocab=256 is valid as a file, but NOT as this C model's input.
            if data[8:16] != struct.pack("<Q",256):
                with self.assertRaises(ValueError): probe_io.read_ids(path)
            self.run_c("trace",ids=path,ok=False)
        for ids in ([],[True],[-1],[16],[0]*257):
            with self.assertRaises(ValueError): probe_io.write_ids(self.new(),16,ids)

    def test_exact_utf8_and_special_policy(self):
        tokenizer=FakeTokenizer(); raw="  café\r\n茶\n".encode()
        ids=tokens.encode_text(tokenizer,raw)
        self.assertEqual(tokenizer.received.encode(),raw)
        self.assertEqual(tokens.decode_ids(tokenizer,49152,ids),raw)
        self.assertEqual(tokens.encode_text(tokenizer,b"",bos=True),[0])
        self.assertEqual(tokens.encode_text(tokenizer,b"a",bos=True),[0,98])
        for raw in (b"",b"\xff",b"<|endoftext|>",b"a"*257,b"a"*65537):
            with self.assertRaises((ValueError,UnicodeError)):
                tokens.encode_text(tokenizer,raw)
        self.assertEqual(tokens.encode_text(tokenizer,b"<|endoftext|>",allow_special=True),[0])
        self.assertEqual(tokens.decode_ids(tokenizer,49152,[0,98]),b"<|endoftext|>a")
        self.assertEqual(tokens.decode_ids(tokenizer,49152,[0,98],skip_special=True),b"a")
        with self.assertRaises(ValueError): tokens.decode_ids(tokenizer,16,[0])

    def test_tokenizer_loading_and_pinned_fetch(self):
        calls=[]
        def load(path,**kwargs):
            calls.append((path,kwargs)); return FakeTokenizer()
        fake=types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=load))
        with patch.dict(sys.modules,{"transformers":fake}): tokens.load_tokenizer(self.root)
        self.assertEqual(calls[0][1],dict(local_files_only=True,trust_remote_code=False,use_fast=True))
        urls=[]
        def request(req,timeout):
            self.assertEqual(timeout,60); urls.append(req.full_url); return io.BytesIO(b"{}")
        output=self.new()
        with patch.object(tokens.urllib.request,"urlopen",side_effect=request): tokens.fetch(output)
        self.assertEqual(len(urls),3)
        self.assertTrue(all(exporter.REVISION in url for url in urls))
        failed=self.new()
        with patch.object(tokens.urllib.request,"urlopen",side_effect=OSError("offline")):
            with self.assertRaises(OSError): tokens.fetch(failed)
        self.assertFalse(failed.exists())
        self.assertFalse(list(self.root.glob(".tokenizer-*")))

```

The next group compares all intermediate values across two attention strategies and two C matrix backends. Prefix tests vary the initial cache-prefill split: lengths one, three and five must produce the same fixed-prefix logits as independent recomputation. This catches an implementation that works only when every request begins with one token.

Generation is then performed by C under all six strategy/backend combinations. Python reads the resulting IDs without choosing replacements, and the reference evaluates the **same** transcript. The EOS test uses the first ID actually selected by C as the stopping ID; it checks a genuine state transition rather than hoping that arbitrary random weights will select a particular number.

```python title="tests/test_reference.py"
    def test_four_intermediate_paths_against_independent_reference(self):
        for strategy in ("full","tiled"):
            for backend in ("ref","blocked"):
                path,_=self.run_c("trace",strategy,backend)
                report=compare(path,self.expected,atol=1e-5,rtol=2e-5)
                self.assertTrue(report["passed"],report)
                self.assertEqual((report["records"],report["values"]),(33,1264))
                self.assertEqual(report["greedy_matches"],1)
                print(f"fixture trace {strategy}/{backend}: values={report['values']} max_abs={report['max_abs']:.9g}")

    def test_identical_prefixes_across_prefill_splits(self):
        for start in (1,3,5):
            expected=self.new(); reference.export_reference(self.source,self.ids,expected,
                mode="prefix",start=start,fixture=True)
            for strategy in ("full","tiled","cache"):
                for backend in ("ref","blocked"):
                    path,_=self.run_c("prefix",strategy,backend,extra=(start,))
                    report=compare(path,expected,atol=1e-5,rtol=2e-5)
                    self.assertTrue(report["passed"],report)
                    self.assertEqual(report["values"],16*(6-start))

    def test_c_generation_then_fixed_prefix_reference(self):
        outputs=[]
        prompt=self.new(); probe_io.write_ids(prompt,16,[0,3])
        for strategy in ("full","tiled","cache"):
            for backend in ("ref","blocked"):
                path,run=self.run_c("generate",strategy,backend,extra=(5,"none"),ids=prompt)
                self.assertIn("consumed=6",run.stderr)
                outputs.append(path.read_bytes())
        self.assertTrue(all(x==outputs[0] for x in outputs))
        transcript=self.new(); transcript.write_bytes(outputs[0])
        vocab,ids=probe_io.read_ids(transcript)
        self.assertEqual((vocab,len(ids)),(16,7))
        expected=self.new(); reference.export_reference(self.source,transcript,expected,
            mode="prefix",start=2,fixture=True)
        for strategy in ("full","tiled","cache"):
            path,_=self.run_c("prefix",strategy,extra=(2,),ids=transcript)
            self.assertTrue(compare(path,expected,atol=1e-5,rtol=2e-5)["passed"])
        # Stop on the actual FIRST selected ID; the prompt's 0 does not stop generation.
        stopped,run=self.run_c("generate","cache",extra=(5,ids[2]),ids=prompt)
        self.assertEqual(probe_io.read_ids(stopped)[1],ids[:3])
        self.assertIn("eos=selected",run.stderr)
        last=self.new(); probe_io.write_ids(last,16,[0]*8)
        self.run_c("generate","cache",extra=(1,"none"),ids=last)  # One selection at max position.
        self.run_c("generate","cache",extra=(2,"none"),ids=last,ok=False)
        print("synthetic C-selected IDs:",ids)


```

The original tiny fixture has head width two. That is insufficient to distinguish all rotary pairing mistakes: adjacent and split-half pairing coincide when there is only one coordinate pair. The second fixture has head width four, a nondefault rotary base, a 300-row vocabulary, nonuniform BF16 weights, and inputs above 255.

It also constructs its source inventory directly from independent shapes and writes it through `safetensors.torch.save_file`. Thus the reference comparison does not depend exclusively on the hand-built Chapter 20 wire fixture. The fixed seed makes the fixture reproducible; its values are synthetic, not trained parameters.

```python title="tests/test_reference.py"
    def test_wider_heads_high_ids_and_independent_weight_loading(self):
        import torch
        from safetensors.torch import save_file
        source=self.new(); source.mkdir()
        raw=raw_config(); raw.update(vocab_size=300,hidden_size=16,intermediate_size=24,
            max_position_embeddings=16,rope_theta=100000)
        (source/"config.json").write_text(json.dumps(raw))
        generator=torch.Generator().manual_seed(2026)
        weights={}
        shapes={"model.embed_tokens.weight":(300,16),"model.norm.weight":(16,)}
        for layer in range(2):
            prefix=f"model.layers.{layer}."
            shapes[prefix+"input_layernorm.weight"]=(16,)
            shapes[prefix+"post_attention_layernorm.weight"]=(16,)
            for name,shape in (("q_proj",(16,16)),("k_proj",(8,16)),
                               ("v_proj",(8,16)),("o_proj",(16,16))):
                shapes[prefix+"self_attn."+name+".weight"]=shape
            for name,shape in (("gate_proj",(24,16)),("up_proj",(24,16)),("down_proj",(16,24))):
                shapes[prefix+"mlp."+name+".weight"]=shape
        for name,shape in shapes.items():
            value=torch.randn(shape,generator=generator)*0.12
            if len(shape)==1: value+=1.0
            weights[name]=value.to(torch.bfloat16)
        save_file(weights,str(source/"model.safetensors"))
        converted=self.new(); exporter.export(source,converted,fixture=True)
        ids=self.new(); probe_io.write_ids(ids,300,[0,299,2,255,256,299,1])
        expected=self.new(); reference.export_reference(source,ids,expected,fixture=True)
        old=self.converted
        try:
            self.converted=converted
            for strategy in ("full","tiled"):
                for backend in ("ref","blocked"):
                    actual,_=self.run_c("trace",strategy,backend,ids=ids)
                    report=compare(actual,expected,atol=1e-5,rtol=2e-5)
                    self.assertTrue(report["passed"],report)
                    self.assertEqual(report["records"],33)
                    print(f"wide-head trace {strategy}/{backend}: values={report['values']} max_abs={report['max_abs']:.9g}")
            prefix=self.new(); reference.export_reference(source,ids,prefix,mode="prefix",start=3,fixture=True)
            for strategy in ("full","tiled","cache"):
                actual,_=self.run_c("prefix",strategy,"blocked",ids=ids,extra=(3,))
                self.assertTrue(compare(actual,prefix,atol=1e-5,rtol=2e-5)["passed"])
        finally: self.converted=old

```

Finally, the comparison tool is tested against misleading success conditions. A large change to a losing logit preserves the greedy choice but must fail numerically. A near tie can switch the greedy choice while passing the explicit numerical tolerance. A malformed header, missing trailer, missing record or nonfinite value must be rejected, not counted as an approximate agreement.

Existing-output sentinels check that an unsuccessful run does not destroy earlier results. The injected Python write exception checks cleanup of a newly created partial file. These are distinct from a real disk-full or power-loss experiment; the tests below do not claim to simulate either. The source-loader tests reject unsupported geometry features, misuse of the real-profile path for fixtures, and nonfinite source weights.

```python title="tests/test_reference.py"
    def test_numerical_mismatch_not_hidden_by_matching_argmax(self):
        def change(rows):
            row=rows[-2]; winner=max(range(row["cols"]),key=lambda i:row["values"][i])
            row["values"][(winner+1)%row["cols"]]-=0.25
        damaged=self.alter_trace(change)
        report=compare(damaged,self.expected,atol=1e-5,rtol=2e-5)
        self.assertFalse(report["passed"])
        self.assertEqual(report["greedy_matches"],1)
        self.assertEqual(report["first_failure"]["name"],"logits")
        early=self.alter_trace(lambda rows:rows[3]["values"].__setitem__(0,10.0))
        self.assertEqual(compare(early,self.expected,atol=1e-5,rtol=2e-5)["first_failure"]["name"],"query")

    def test_close_argmax_disagreement_and_ties(self):
        path=self.new(); other=self.new()
        header=probe_io.trace_header("prefix",[0],(2,1,2,1,1,2),1,{})
        for out,values in ((path,[1.0000001,1.0]),(other,[1.0,1.0000001])):
            with probe_io.exclusive_file(out) as file:
                probe_io.write_json(file,header)
                probe_io.write_json(file,dict(kind="tensor",name="logits",index=1,rows=1,cols=2,values=values))
                probe_io.write_json(file,dict(kind="end",records=1))
        report=compare(path,other,atol=1e-6,rtol=0)
        self.assertTrue(report["passed"]); self.assertEqual(report["greedy_matches"],0)
        self.assertEqual(report["certified_rows"],0)
        # The same numerical check with a near-zero reference relies on absolute tolerance.
        self.assertRaises(ValueError,compare,path,other,atol=float("nan"),rtol=0)

    def test_header_shape_nonfinite_and_truncation(self):
        mutations=[lambda r:r[0]["ids"].__setitem__(0,1),lambda r:r[0]["geometry"].__setitem__(2,16),
            lambda r:r[1].__setitem__("cols",9),lambda r:r[1]["values"].__setitem__(0,float("nan")),
            lambda r:r.pop(),lambda r:r.pop(2),lambda r:r.append(r[1]),
            lambda r:r[-1].__setitem__("records",32),lambda r:r[0].__setitem__("start",1)]
        for mutate in mutations:
            with self.assertRaises(ValueError): compare(self.alter_trace(mutate),self.expected,atol=1e-5,rtol=2e-5)
        for atol,rtol in ((0,0),(-1,0),(1e-5,-1),(float("inf"),0)):
            with self.assertRaises(ValueError): compare(self.expected,self.expected,atol=atol,rtol=rtol)

    def test_output_safety_and_argument_boundaries(self):
        sentinel=self.new(); sentinel.write_bytes(b"keep")
        self.run_c("trace",output=sentinel,ok=False)
        self.assertEqual(sentinel.read_bytes(),b"keep")
        for mode,strategy,extra in (("trace","cache",()),("prefix","cache",(0,)),
                                   ("prefix","cache",(6,)),("generate","full",(0,"none")),
                                   ("generate","cache",(1,16)),("generate","full",(1,"-1"))):
            target=self.new(); self.run_c(mode,strategy,extra=extra,output=target,ok=False)
            self.assertFalse(target.exists())
        path=self.new(); path.write_bytes(b"keep")
        with self.assertRaises(FileExistsError): probe_io.write_ids(path,16,[0])
        failed=self.new()
        with self.assertRaises(OSError):
            with probe_io.exclusive_file(failed) as file:
                file.write("partial"); raise OSError("injected failure")
        self.assertFalse(failed.exists())

    def test_reference_source_rejection(self):
        config=raw_config(); config["rope_interleaved"]=True
        bad=write_source(self.new(),config=config)
        with self.assertRaises(ValueError): reference.load_source(bad,fixture=True)
        with self.assertRaises(ValueError): reference.load_source(self.source)
        header,payload=fixture_source(); payload=bytearray(payload)
        offset=header["model.norm.weight"]["data_offsets"][0]
        struct.pack_into("<H",payload,offset,0x7fc0)
        bad=write_source(self.new(),header,payload)
        with self.assertRaises(ValueError): reference.load_source(bad,fixture=True)

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]],verbosity=2)
```

Run the cumulative C suite and the new independent suite:

```sh
make test
python3 tests/test_reference.py build/debug/compare
python3 tests/test_reference.py build/sanitize/compare
python3 tests/test_reference.py build/release/compare
make test-reference
```

The source uses the existing native Mac compiler default. For the Linux commands observed below, each `make` invocation instead used `CC=clang`. The Python script must receive an executable path: omitting it deliberately skips C-dependent tests and cannot establish cross-language agreement. Twelve completed tests, not a run containing skipped C tests, is the expected integration result.

## A complete small experiment without external assets

This experiment uses a new directory. Do not reuse a path containing earlier outputs: scripts intentionally refuse replacement. Activate the optional environment first and build `build/release/compare`.

```sh
mkdir -p runs
python3 tests/import_fixture.py runs/chapter21-fixture/source
python3 - <<'PYFIX'
import sys
sys.path.insert(0, 'scripts')
from export_smollm2 import export
from probe_io import write_ids
export('runs/chapter21-fixture/source',
       'runs/chapter21-fixture/converted', fixture=True)
write_ids('runs/chapter21-fixture/prompt.ids', 16, [0, 3, 9, 3, 15])
PYFIX
python3 scripts/reference_smollm2.py \
  runs/chapter21-fixture/source runs/chapter21-fixture/prompt.ids \
  runs/chapter21-fixture/reference.jsonl --fixture
./build/release/compare pretrained trace \
  runs/chapter21-fixture/converted/model.cfg \
  runs/chapter21-fixture/converted/weights.dlw \
  runs/chapter21-fixture/prompt.ids runs/chapter21-fixture/c.jsonl \
  full blocked
python3 scripts/compare_traces.py \
  runs/chapter21-fixture/c.jsonl runs/chapter21-fixture/reference.jsonl \
  --atol 1e-5 --rtol 2e-5
```

There are 33 tensor records and 1,264 compared values. The report must declare numerical acceptance and identify both engines' actual settings. It must not say that these files contain SmolLM2 pretrained weights: the source profile is an explicit fixture.

### Observed fixture results

The following are actual x86_64 Linux observations using Clang 17, Python 3.13.5, CPU PyTorch 2.10.0 and Safetensors 0.7.0. No M4 execution is implied. The numerical limits were $a=10^{-5}$ and $b=2\times10^{-5}$.

| Fixture | Attention strategy | Values per trace | Maximum absolute error against eager FP32 |
|---|---|---:|---:|
| Width 8, head width 2, vocabulary 16 | Full materialized | 1,264 | $1.49011612\times10^{-8}$ |
| Same fixture | Tiled | 1,264 | $1.49011612\times10^{-8}$ |
| Width 16, head width 4, vocabulary 300 | Full materialized | 3,788 | $4.76837158\times10^{-7}$ |
| Same wider fixture | Tiled | 3,788 | $4.17232513\times10^{-7}$ |

Both C matrix backends gave the reported bound in each row. Across the eight trace comparisons, 20,208 intermediate values were compared per build configuration. Fixed-prefix checks additionally cover all vocabulary logits across multiple prefill lengths and the generated transcript. These differences are nonzero because independent eager FP32 arithmetic and our wider C reductions do not have identical rounding paths.

With synthetic prompt IDs `[0,3]`, all six C strategy/backend combinations selected the five-ID continuation `[15,15,15,15,15]`. The resulting transcript has seven IDs, while the largest consumed prefix has six. This is a state/consistency result, **not text quality** and not a pretrained continuation.

The cumulative 21 C test programs and the new 12-test Python suite passed debug, AddressSanitizer/UndefinedBehaviorSanitizer and optimized builds. The new observer test and sanitizer-driven reference integration also passed with leak detection enabled. Prior preparation, import, training and generation checks remain in the snapshot and passed their invoked checks. No compiler warnings or new sanitizer diagnostics were observed.

## Localize two deliberate implementation errors

Make each change only in a disposable working copy, keep the correct reference file unchanged, and restore the implementation before the next experiment.

First, in `dl_inference_forward_trace`, change the attention normalization's epsilon argument from `c.epsilon` to `c.epsilon * 1000.0f`. The metadata still describes the correct configuration; the computation is wrong. On the small fixture, the first failed stage is `attn_norm`, block zero, row zero, column zero. The observed values were approximately -0.01634425 in C and -0.05320407 in the reference. Do not compensate by giving the reference the same wrong epsilon.

Second, restore epsilon and change the workspace-construction loop from `next.positions[i] = i` to `next.positions[i] = 0`. Embeddings, normalization and the unrotated projections still agree. The first failure is `rotated_query`, block zero, row one, column zero: approximately -0.04290363 versus -0.03279041. Position zero itself is unchanged, which explains why testing a one-token prompt would miss the defect.

Both variants ran under AddressSanitizer/UndefinedBehaviorSanitizer without a memory diagnostic and failed the explicit numerical comparison. They access valid storage and compute the wrong function. This is why neither a sanitizer nor agreement of selected tokens can replace intermediate numerical tests.

## Run the genuine-asset experiment separately

The commands below are the real-model workflow, **not a transcript of a completed pretrained run**. In the authoring environment, DNS prevented acquisition of the genuine weight/tokenizer payloads, and Transformers/tokenizers were not installed. The executed numerical results above use genuine PyTorch arithmetic but synthetic source weights. The fake tokenizer tests do not establish real-tokenizer output.

Use the same pinned revision for model and tokenizer. When Chapter 20's genuine source and converted directories already exist, reuse those paths; do not invoke their download/conversion again or replace them with fixtures. Otherwise, run the existing asset-only converter and then fetch the tokenizer:

```sh
python3 scripts/export_smollm2.py \
  --download assets/smollm2-source --output assets/smollm2-f32
python3 scripts/tokenize.py fetch assets/smollm2-tokenizer
```

Install the optional packages from Part 3 before encoding. Create a short prompt without an implicit shell-added newline. The base model does not require applying an instruction/chat template to this experiment.

```sh
mkdir -p runs/pretrained-check
printf 'Once upon a time,' > runs/pretrained-check/prompt.utf8
python3 scripts/tokenize.py encode assets/smollm2-tokenizer \
  runs/pretrained-check/prompt.utf8 runs/pretrained-check/prompt.ids \
  > runs/pretrained-check/tokenization.json
python3 scripts/tokenize.py inspect runs/pretrained-check/prompt.ids
python3 scripts/tokenize.py decode assets/smollm2-tokenizer \
  runs/pretrained-check/prompt.ids runs/pretrained-check/roundtrip.utf8
cmp runs/pretrained-check/prompt.utf8 runs/pretrained-check/roundtrip.utf8
```

Inspect the actual IDs and count; do not assume a count from the number of words. The intermediate trace needs at most 16 IDs. Choose an explicitly shorter prompt if needed rather than adding hidden truncation. The round-trip command is a check of these particular prompt bytes and actual assets. A failure needs investigation, not an invented expected ID list. Adding `--bos` intentionally changes the round trip if special tokens are retained; keep the policy consistent and documented.

Generate C intermediates and the independent reference in **separate processes**:

```sh
./build/release/compare pretrained trace \
  assets/smollm2-f32/model.cfg assets/smollm2-f32/weights.dlw \
  runs/pretrained-check/prompt.ids runs/pretrained-check/c-full.jsonl \
  full blocked
python3 scripts/reference_smollm2.py \
  assets/smollm2-source runs/pretrained-check/prompt.ids \
  runs/pretrained-check/reference.jsonl
python3 scripts/compare_traces.py \
  runs/pretrained-check/c-full.jsonl runs/pretrained-check/reference.jsonl \
  --atol 1e-4 --rtol 2e-4
./build/release/compare pretrained trace \
  assets/smollm2-f32/model.cfg assets/smollm2-f32/weights.dlw \
  runs/pretrained-check/prompt.ids runs/pretrained-check/c-tiled.jsonl \
  tiled blocked
python3 scripts/compare_traces.py \
  runs/pretrained-check/c-tiled.jsonl runs/pretrained-check/reference.jsonl \
  --atol 1e-4 --rtol 2e-4
```

The displayed real-model tolerances are **initial diagnostic acceptance targets**, not observed agreement bounds or guaranteed pass criteria for the genuine model. Keep the first failure, per-row margins, maximum error and actual settings. Investigate an early localized discrepancy before considering a justified tolerance change for accumulated rounding. Do not report a pass without executing the full comparison to its trailer.

Now let C select four new greedy IDs and decode them only afterward:

```sh
./build/release/compare pretrained generate \
  assets/smollm2-f32/model.cfg assets/smollm2-f32/weights.dlw \
  runs/pretrained-check/prompt.ids runs/pretrained-check/transcript.ids \
  cache blocked 4 0
python3 scripts/tokenize.py inspect runs/pretrained-check/transcript.ids
python3 scripts/tokenize.py decode assets/smollm2-tokenizer \
  runs/pretrained-check/transcript.ids runs/pretrained-check/transcript.utf8
```

The last argument is the explicitly chosen EOS ID 0. Early EOS is an allowed outcome, not an excuse to remove a selected ID from the numeric transcript. The decoded file includes the original prompt and, by default, visible special tokens. Nothing here asserts a particular sentence before the genuine run has taken place.

Finally, compare all strategies on that **same C-selected transcript**. Obtain the actual prompt count for the initial prefill, not its byte count:

```sh
P=$(python3 -c 'import sys; sys.path.insert(0,"scripts"); from probe_io import read_ids; print(len(read_ids(sys.argv[1])[1]))' runs/pretrained-check/prompt.ids)
python3 scripts/reference_smollm2.py \
  assets/smollm2-source runs/pretrained-check/transcript.ids \
  runs/pretrained-check/reference-prefix.jsonl --mode prefix --start "$P"
for strategy in full tiled cache; do
  ./build/release/compare pretrained prefix \
    assets/smollm2-f32/model.cfg assets/smollm2-f32/weights.dlw \
    runs/pretrained-check/transcript.ids \
    "runs/pretrained-check/c-prefix-$strategy.jsonl" "$strategy" blocked "$P"
  python3 scripts/compare_traces.py \
    "runs/pretrained-check/c-prefix-$strategy.jsonl" \
    runs/pretrained-check/reference-prefix.jsonl --atol 1e-4 --rtol 2e-4 || break
done
```

Inspect each exit status; a failed run stops being acceptable evidence even if a later shell command succeeds. This prefix sweep also evaluates the final selected ID as an input and obtains one extra prediction that generation did not need. The short experiment has room for that input. At the model's position limit, omit that extra prefix or choose a shorter experiment; do not claim that generation's $P+G-1$ capacity implies room to consume all $P+G$ transcript IDs.

## Completion criteria and remaining limits

A completed local genuine-model check records the actual tokenizer versions and flags, explicit prompt IDs, source revision, C and reference settings, numerical tolerances, first divergence if any, all-logit error and greedy margins, plus a short C-selected transcript. The decoded text illustrates the selected IDs; it is not the correctness test.

At this chapter's tested state, source mapping, C observation, fixed-prefix comparison, cache splits, EOS state, high IDs and the explicit eager reference are exercised on two independent synthetic fixtures. Genuine pretrained agreement and genuine tokenization remain **unexecuted** in the authoring environment. No M4 execution, pretrained performance or process-memory measurement is claimed here. The commands and failure checks are present so those specific gaps can be tested rather than inferred.

The next chapter, **End-to-end demonstration and measured tradeoffs**, assembles the trained small model, cached generation, tiled training and imported-model path into one reproducible technical demonstration, with separately reported correctness, runtime and memory observations.
