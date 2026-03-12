#!/usr/bin/env python3
"""
Pairs vulnerable and secure functions from the SARD JSONL.
Groups by source file, then pairs each vulnerable function with each
secure counterpart from the same file.

Output schema per record:
  vulnerable  - the bad function body
  secure      - the corresponding good function body
  language    - C / C++ / Java
  cwe_id      - e.g. CWE-78
  cwe_name    - e.g. OS Command Injection
  source_file - original filename
  test_suite  - test suite name

Usage:
    python scripts/make_pairs.py \
        --input  artifacts/sard_functions.jsonl \
        --output artifacts/sard_pairs.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="artifacts/sard_functions.jsonl")
    p.add_argument("--output", default="artifacts/sard_pairs.jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    # Group records by (source_file, test_suite) so files from different
    # suites with the same name don't get mixed.
    groups = defaultdict(lambda: {"vulnerable": [], "secure": [], "meta": None})

    print("Loading records ...")
    with open(args.input, encoding="utf-8") as f:
        for line in tqdm(f):
            rec = json.loads(line)
            key = (rec["source_file"], rec["test_suite"])
            g = groups[key]
            if rec["label"] == 1:
                g["vulnerable"].append(rec["func"])
            else:
                g["secure"].append(rec["func"])
            # Store metadata from any record in this group (all share the same)
            if g["meta"] is None:
                g["meta"] = {k: rec[k] for k in ("language", "cwe_id", "cwe_name",
                                                   "source_file", "test_suite")}

    print(f"Loaded {len(groups)} source files.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for g in tqdm(groups.values(), desc="Building pairs"):
            vulns  = g["vulnerable"]
            safes  = g["secure"]
            meta   = g["meta"]

            if not vulns or not safes:
                skipped += 1
                continue

            # Pair each vulnerable with each secure from the same file.
            # Juliet typically has 1 bad and 2 good (goodG2B + goodB2G),
            # so most files produce 2 pairs.
            for v in vulns:
                for s in safes:
                    record = {
                        "vulnerable": v,
                        "secure":     s,
                        **meta,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pairs += 1

    print(f"\nDone.")
    print(f"  Pairs written : {total_pairs:,}")
    print(f"  Files skipped : {skipped:,}  (missing one side)")
    print(f"  Output        : {output_path.resolve()}")


if __name__ == "__main__":
    main()
