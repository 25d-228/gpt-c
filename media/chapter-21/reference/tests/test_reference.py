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
