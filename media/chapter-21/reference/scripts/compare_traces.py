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
