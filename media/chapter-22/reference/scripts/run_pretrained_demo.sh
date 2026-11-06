#!/bin/sh
# Existing pinned assets only; model execution and token selection stay in C.
set -eu
if [ "$#" -ne 4 ]; then
    echo 'usage: sh scripts/run_pretrained_demo.sh SOURCE CONVERTED TOKENIZER NEW_OUTPUT' >&2
    exit 1
fi
root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
source=$(CDPATH= cd "$1" && pwd)
converted=$(CDPATH= cd "$2" && pwd)
tokenizer=$(CDPATH= cd "$3" && pwd)
case "$4" in /*) out=$4 ;; *) out=$PWD/$4 ;; esac
mkdir "$out"
cd "$root"
python3 - "$converted" "$tokenizer" > "$out/assets.txt" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from export_smollm2 import MODEL, REVISION
weights = json.loads((Path(sys.argv[1])/'source.json').read_text())
tokens = json.loads((Path(sys.argv[2])/'source.json').read_text())
if (weights.get('model_profile') != MODEL or weights.get('expected_revision') != REVISION
        or tokens.get('model') != MODEL or tokens.get('revision') != REVISION):
    raise SystemExit('asset provenance must name the selected model/revision, not a fixture')
print(json.dumps(dict(weights=weights, tokenizer=tokens), indent=2))
print('Provenance records are declarations, not authentication of the file contents.')
PY
make -B CC="${CC:-xcrun clang}" build/release/compare > "$out/build.log" 2>&1
{
    date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
    uname -a
    python3 -c 'import sys, torch, safetensors, transformers, tokenizers; print(sys.version); print("torch",torch.__version__,"safetensors",safetensors.__version__,"transformers",transformers.__version__,"tokenizers",tokenizers.__version__)'
} > "$out/environment.txt"
printf 'Once upon a time,' > "$out/prompt.utf8"
python3 scripts/tokenize.py encode "$tokenizer" "$out/prompt.utf8" "$out/prompt.ids" \
    > "$out/tokenization.json"
python3 scripts/tokenize.py decode "$tokenizer" "$out/prompt.ids" "$out/roundtrip.utf8"
cmp "$out/prompt.utf8" "$out/roundtrip.utf8"
P=$(python3 -c 'import sys; sys.path.insert(0,"scripts"); from probe_io import read_ids; p=len(read_ids(sys.argv[1])[1]); assert 0<p<=16, "trace prompt must contain 1..16 IDs"; print(p)' "$out/prompt.ids")
compare=./build/release/compare
python3 scripts/reference_smollm2.py "$source" "$out/prompt.ids" "$out/reference.jsonl"
for mode in full tiled; do
    "$compare" pretrained trace "$converted/model.cfg" "$converted/weights.dlw" \
        "$out/prompt.ids" "$out/c-$mode.jsonl" "$mode" blocked
    python3 scripts/compare_traces.py "$out/c-$mode.jsonl" "$out/reference.jsonl" \
        --atol 1e-4 --rtol 2e-4 > "$out/trace-$mode.txt"
done
"$compare" pretrained generate "$converted/model.cfg" "$converted/weights.dlw" \
    "$out/prompt.ids" "$out/transcript.ids" cache blocked 4 0
python3 scripts/tokenize.py decode "$tokenizer" "$out/transcript.ids" "$out/transcript.utf8"
python3 scripts/reference_smollm2.py "$source" "$out/transcript.ids" \
    "$out/reference-prefix.jsonl" --mode prefix --start "$P"
for mode in full tiled cache; do
    "$compare" pretrained prefix "$converted/model.cfg" "$converted/weights.dlw" \
        "$out/transcript.ids" "$out/prefix-$mode.jsonl" "$mode" blocked "$P"
    python3 scripts/compare_traces.py "$out/prefix-$mode.jsonl" "$out/reference-prefix.jsonl" \
        --atol 1e-4 --rtol 2e-4 > "$out/prefix-$mode.txt"
    # Deterministic synthetic IDs for timing, not a claim about generated text.
    "$compare" model check "$converted/model.cfg" "$converted/weights.dlw" \
        "$mode" blocked 4 2 1 16 32 > "$out/measure-$mode-check.txt"
    "$compare" model measure "$converted/model.cfg" "$converted/weights.dlw" \
        "$mode" blocked 4 2 1 16 32 > "$out/measure-$mode.txt"
done
printf '%s\n' 'All requested real-asset commands exited successfully; inspect errors, margins and output.' \
    > "$out/completed.txt"
