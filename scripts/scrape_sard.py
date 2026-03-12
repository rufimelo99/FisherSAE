#!/usr/bin/env python3
"""
Scraper for NIST SARD (Software Assurance Reference Dataset) test suites.
Downloads Juliet C/C++ 1.3 and Juliet Java 1.3 test suites, extracts
functions labeled as vulnerable (bad) or secure (good), and writes to JSONL.

Usage:
    python scripts/scrape_sard.py [--output artifacts/sard_functions.jsonl]
                                   [--suites cpp java]
                                   [--cache-dir .sard_cache]
"""

import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Test suite definitions
# ---------------------------------------------------------------------------
SUITES = {
    "cpp": {
        "name": "Juliet C/C++ 1.3",
        "url": "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip",
        "languages": ["C", "C++"],
        "extensions": [".c", ".cpp", ".h"],
    },
    "java": {
        "name": "Juliet Java 1.3",
        "url": "https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-java-v1-3.zip",
        "languages": ["Java"],
        "extensions": [".java"],
    },
}

# CWE regex: match CWE number and name from Juliet-style file paths like
# "CWE114_Process_Control__char_connect_socket_01.c"
# The CWE name ends at the double-underscore separator.
CWE_PATTERN = re.compile(r"CWE(\d+)_(.+?)__")

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> Path:
    """Download url to dest with a progress bar. Returns dest path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [cache] {dest.name} already exists, skipping download.")
        return dest

    print(f"  Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


# ---------------------------------------------------------------------------
# CWE extraction from path
# ---------------------------------------------------------------------------

def extract_cwe(filepath: str):
    """Extract (cwe_id, cwe_name) from a Juliet-style file path."""
    fname = Path(filepath).name
    m = CWE_PATTERN.search(fname)
    if m:
        cwe_num = m.group(1)
        cwe_name = m.group(2).replace("_", " ")
        return f"CWE-{cwe_num}", cwe_name
    return None, None


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".java":
        return "Java"
    if ext in (".cpp", ".cc", ".cxx"):
        return "C++"
    return "C"


# ---------------------------------------------------------------------------
# Function extraction: C / C++
# ---------------------------------------------------------------------------

# Matches the start of a C/C++ function definition (not a declaration).
# We look for an identifier followed by a parameter list and then `{`.
# This is intentionally loose to handle Juliet's consistent style.
C_FUNC_START = re.compile(
    r"^(?:(?:static|inline|void|int|char|short|long|unsigned|signed|double|float|bool|"
    r"size_t|int64_t|int32_t|wchar_t|WSADATA|SOCKET|FILE|SomeDataStruct)\s+)*"
    r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"\s*\([^;{]*\)\s*\{",
    re.MULTILINE,
)

# Names that map to vulnerable / safe
VULN_NAMES = re.compile(r"\b(bad|badSink|badSource|CWE\w+bad)\b", re.IGNORECASE)
SAFE_NAMES  = re.compile(r"\b(good|goodG2B|goodB2G|goodG2BSink|goodB2GSink|goodG2BSource|goodB2GSource)\b", re.IGNORECASE)


def _brace_match(text: str, start: int) -> int:
    """Return the index of the closing `}` that matches the `{` at `start`."""
    depth = 0
    i = start
    in_string = False
    string_char = None
    in_line_comment = False
    in_block_comment = False

    while i < len(text):
        ch = text[i]

        # Line comment
        if not in_string and not in_block_comment and ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            in_line_comment = True
            i += 2
            continue
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        # Block comment
        if not in_string and not in_line_comment and ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        if in_block_comment:
            if ch == "*" and i + 1 < len(text) and text[i + 1] == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        # String literals
        if not in_string and ch in ('"', "'"):
            in_string = True
            string_char = ch
            i += 1
            continue
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == string_char:
                in_string = False
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def extract_c_functions(source: str):
    """
    Yield (function_name, function_body, label) tuples from C/C++ source.
    label: 'vulnerable' | 'safe' | None (unknown, skipped)
    """
    for m in C_FUNC_START.finditer(source):
        name = m.group("name")
        brace_pos = source.index("{", m.start())
        end = _brace_match(source, brace_pos)
        if end == -1:
            continue
        body = source[m.start():end + 1]

        if VULN_NAMES.search(name):
            label = "vulnerable"
        elif SAFE_NAMES.search(name):
            label = "safe"
        else:
            # Skip helper / main / support functions
            continue

        yield name, body, label


# ---------------------------------------------------------------------------
# Function extraction: Java
# ---------------------------------------------------------------------------

JAVA_METHOD_START = re.compile(
    r"(?:public|private|protected|static|void|final|synchronized|\s)+"
    r"(?:void|int|String|boolean|char|byte|short|long|double|float|Object|"
    r"[A-Z][a-zA-Z0-9_<>\[\]]*)\s+"
    r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;{]*\)\s*(?:throws\s+\S+\s*)?\{",
    re.MULTILINE,
)


def extract_java_methods(source: str):
    """
    Yield (method_name, method_body, label) from Java source.
    """
    for m in JAVA_METHOD_START.finditer(source):
        name = m.group("name")
        brace_pos = source.index("{", m.start())
        end = _brace_match(source, brace_pos)
        if end == -1:
            continue
        body = source[m.start():end + 1]

        if VULN_NAMES.search(name):
            label = "vulnerable"
        elif SAFE_NAMES.search(name):
            label = "safe"
        else:
            continue

        yield name, body, label


# ---------------------------------------------------------------------------
# Process a single source file
# ---------------------------------------------------------------------------

def process_file(zip_path_str: str, source: str, suite_name: str, language: str):
    """
    Parse source code and return list of record dicts ready for JSONL output.
    """
    cwe_id, cwe_name = extract_cwe(zip_path_str)
    records = []

    extractor = extract_java_methods if language == "Java" else extract_c_functions

    for func_name, func_body, label in extractor(source):
        records.append({
            "func":          func_body,
            "label":         1 if label == "vulnerable" else 0,
            "vulnerability": label,
            "language":      language,
            "cwe_id":        cwe_id,
            "cwe_name":      cwe_name,
            "function_name": func_name,
            "source_file":   Path(zip_path_str).name,
            "test_suite":    suite_name,
        })

    return records


# ---------------------------------------------------------------------------
# Process a downloaded ZIP
# ---------------------------------------------------------------------------

def process_zip(zip_path: Path, suite_meta: dict, writer):
    """Open the ZIP, iterate source files, extract functions, write to writer."""
    suite_name = suite_meta["name"]
    extensions = tuple(suite_meta["extensions"])
    total_records = 0

    print(f"  Processing {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(extensions)]
        print(f"    Found {len(names)} source files.")

        for zname in tqdm(names, desc="  Extracting functions"):
            # Skip header-only files (mostly support/helper code)
            if zname.lower().endswith(".h") and "_support" not in zname.lower():
                # Only include headers if they contain bad/good functions
                pass

            try:
                with zf.open(zname) as f:
                    source = f.read().decode("utf-8", errors="replace")
            except Exception:
                continue

            language = detect_language(zname)
            records = process_file(zname, source, suite_name, language)

            for rec in records:
                writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1

    print(f"    Wrote {total_records} function records from {zip_path.name}.")
    return total_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Scrape NIST SARD Juliet test suites to JSONL.")
    p.add_argument("--output", default="artifacts/sard_functions.jsonl",
                   help="Output JSONL file path.")
    p.add_argument("--suites", nargs="+", choices=list(SUITES.keys()), default=list(SUITES.keys()),
                   help="Which test suites to include (default: all).")
    p.add_argument("--cache-dir", default=".sard_cache",
                   help="Directory to cache downloaded ZIPs.")
    return p.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for suite_key in args.suites:
            suite_meta = SUITES[suite_key]
            print(f"\n=== Suite: {suite_meta['name']} ===")

            zip_filename = suite_meta["url"].split("/")[-1]
            zip_path = cache_dir / zip_filename

            # Download
            download_file(suite_meta["url"], zip_path)

            # Process
            n = process_zip(zip_path, suite_meta, out_f)
            total += n

    print(f"\nDone. Total records written: {total}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
