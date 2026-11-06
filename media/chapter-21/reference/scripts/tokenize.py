#!/usr/bin/env python3
"""Course CLI when executed; preserve Python's stdlib tokenize when imported."""
import importlib.util
from pathlib import Path
import sys
import sysconfig

if __name__ == "__main__":
    from tokenizer_boundary import main
    raise SystemExit(main())
else:
    # Other Python libraries import stdlib tokenize. This neighboring filename
    # must not accidentally replace its generate_tokens/open/TokenInfo API.
    path = Path(sysconfig.get_path("stdlib")) / "tokenize.py"
    spec = importlib.util.spec_from_file_location("_dl_stdlib_tokenize", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if not name.startswith("__") or name == "__all__":
            globals()[name] = value
