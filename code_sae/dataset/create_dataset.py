"""
Script to read SecureCode V2 dataset from local JSONL file in artifacts
and create a processed dataset with vulnerable/secure code samples.

SecureCode V2 entry format:
{
    "id": <unique identifier>,
    "metadata": {
        "lang": <programming language>,
        "category": <security category>,
        "subcategory": <subcategory>,
        "technique": <attack technique>,
        "owasp_llm_2025": <OWASP LLM identifier>,
        "cwe": <CWE identifier>,
        "severity": <HIGH|MEDIUM|LOW>,
        "complexity": <high|medium|low>,
        "created": <date>,
        "validated": <boolean>,
        "grounding_tier": <T1|T2|T3>,
        "grounding_source": <source description>,
        "grounding_source_type": <cve|research|etc>
    },
    "context": {
        "description": <vulnerability description>,
        "impact": <impact description>,
        "real_world_example": {
            "incident": <incident description>,
            "date": <date>,
            "cost_impact_usd": <cost>,
            "organization_type": <org type>
        }
    },
    "conversations": [
        {"role": "human", "content": <question>},
        {"role": "assistant", "content": <response with code examples>}
    ],
    "validation": {...},
    "security_assertions": [...],
    "quality_score": <number>,
    "references": [...]
}

Code is extracted from assistant responses by looking for:
- **Vulnerable Implementation:** followed by code blocks
- **Secure Implementation:** followed by code blocks
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from markdown text.
    Returns list of (language, code) tuples.
    """
    # Match ```language\ncode\n``` patterns
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang or "", code.strip()) for lang, code in matches]


def extract_vulnerable_and_secure_code(content: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract vulnerable and secure code from assistant response content.
    Returns (vulnerable_codes, secure_codes) where each is a list of dicts
    with 'code' and 'language' keys.
    """
    vulnerable_codes = []
    secure_codes = []

    # Split content by implementation markers
    # Look for "Vulnerable Implementation" and "Secure Implementation" sections

    # Patterns to identify vulnerable code sections
    vuln_patterns = [
        r"\*\*Vulnerable Implementation[:\*]*\*\*",
        r"# Vulnerable",
        r"## Vulnerable",
        r"VULNERABLE:",
        r"\*\*Vulnerable Code[:\*]*\*\*",
    ]

    # Patterns to identify secure code sections
    secure_patterns = [
        r"\*\*Secure Implementation[:\*]*\*\*",
        r"# Secure",
        r"## Secure",
        r"SECURE:",
        r"\*\*Secure Code[:\*]*\*\*",
        r"\*\*Production-Grade Defense[:\*]*\*\*",
    ]

    # Find all code blocks with their positions
    code_block_pattern = r"```(\w+)?\n(.*?)```"
    code_matches = list(re.finditer(code_block_pattern, content, re.DOTALL))

    # Find positions of vulnerable/secure markers
    vuln_positions = []
    for pattern in vuln_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            vuln_positions.append(match.start())

    secure_positions = []
    for pattern in secure_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            secure_positions.append(match.start())

    # Classify each code block based on preceding markers
    for code_match in code_matches:
        code_pos = code_match.start()
        lang = code_match.group(1) or ""
        code = code_match.group(2).strip()

        # Skip non-code blocks (text, yaml configs, etc.)
        if lang.lower() in ["text", "yaml", "yml", "json", "bash", "sh", "shell"]:
            continue

        # Skip very short code blocks (likely examples, not implementations)
        if len(code) < 100:
            continue

        # Find the closest preceding marker
        closest_vuln = max([p for p in vuln_positions if p < code_pos], default=-1)
        closest_secure = max([p for p in secure_positions if p < code_pos], default=-1)

        code_entry = {"code": code, "language": lang}

        if closest_vuln > closest_secure:
            vulnerable_codes.append(code_entry)
        elif closest_secure > closest_vuln:
            secure_codes.append(code_entry)
        # If no markers found or equal, try to infer from code comments
        elif "VULNERABLE" in code.upper() or "# INJECTION" in code.upper():
            vulnerable_codes.append(code_entry)
        elif "SECURE" in code.upper() or "SANITIZ" in code.upper():
            secure_codes.append(code_entry)

    return vulnerable_codes, secure_codes


def process_securecode_entry(row: Dict) -> List[Dict]:
    """
    Process a single SecureCode V2 entry and extract paired vulnerable/secure samples.
    Returns list of rows with both vulnerable_code and secure_code columns.
    """
    entry_id = row.get("id", "unknown")
    metadata = row.get("metadata", {})
    context = row.get("context", {})
    conversations = row.get("conversations", [])

    # Common fields for all extracted code
    common_fields = {
        "entry_id": entry_id,
        "lang": metadata.get("lang", ""),
        "category": metadata.get("category", ""),
        "subcategory": metadata.get("subcategory", ""),
        "technique": metadata.get("technique", ""),
        "owasp_llm_2025": metadata.get("owasp_llm_2025", ""),
        "cwe": metadata.get("cwe", ""),
        "severity": metadata.get("severity", ""),
        "complexity": metadata.get("complexity", ""),
        "grounding_tier": metadata.get("grounding_tier", ""),
        "grounding_source": metadata.get("grounding_source", ""),
        "grounding_source_type": metadata.get("grounding_source_type", ""),
        "description": context.get("description", ""),
        "impact": context.get("impact", ""),
        "quality_score": row.get("quality_score"),
        "security_assertions": row.get("security_assertions", []),
    }

    # Collect all vulnerable and secure code from all assistant responses
    all_vuln_codes = []
    all_secure_codes = []

    for conv in conversations:
        if conv.get("role") == "assistant":
            content = conv.get("content", "")
            vuln_codes, secure_codes = extract_vulnerable_and_secure_code(content)
            all_vuln_codes.extend(vuln_codes)
            all_secure_codes.extend(secure_codes)

    # Only keep first vulnerable and first secure code (others are often tests/examples)
    vuln_code = all_vuln_codes[0]["code"] if all_vuln_codes else ""
    secure_code = all_secure_codes[0]["code"] if all_secure_codes else ""

    # Only create row if BOTH codes exist
    if vuln_code and secure_code:
        return [
            {
                **common_fields,
                "vulnerable_code": vuln_code,
                "secure_code": secure_code,
            }
        ]

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Parse SecureCode V2 dataset from artifacts folder"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSONL file (default: artifacts/securecode_v2_train.jsonl)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo to push combined dataset (e.g., 'username/securecode')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the pushed dataset private",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        default=True,
        help="Save JSONL file locally (default: True)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="securecode_v2",
        help="Prefix for output files (default: securecode_v2)",
    )
    args = parser.parse_args()

    # Load from local JSONL file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "securecode_v2_train.jsonl"
        )

    print(f"Loading dataset from {input_path}...")

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        print("Please provide the path to your SecureCode V2 JSONL file using --input")
        return

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Dataset loaded with {len(data)} entries")
    if data:
        print(f"Top-level keys: {list(data[0].keys())}")
        if "metadata" in data[0]:
            print(f"Metadata keys: {list(data[0]['metadata'].keys())}")

    # Process each entry (only keeps entries with both vulnerable and secure code)
    all_rows = []
    skipped = 0

    for row in data:
        result = process_securecode_entry(row)
        if result:
            all_rows.extend(result)
        else:
            skipped += 1

    print(f"\nProcessing complete:")
    print(f"  - Total entries: {len(data)}")
    print(f"  - Entries with both vulnerable & secure: {len(all_rows)}")
    print(f"  - Skipped (missing one or both): {skipped}")

    # Create HuggingFace Dataset
    if all_rows:
        dataset = Dataset.from_list(all_rows)
        print(f"\nDataset columns: {dataset.column_names}")
    else:
        dataset = None
        print("\nNo code samples extracted")

    # Save to JSONL locally
    if args.save_local and all_rows:
        output_path = Path(__file__).parent / f"{args.output_prefix}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nCombined dataset saved to: {output_path}")

    # Push to HuggingFace Hub
    if args.push_to_hub and dataset:
        print(f"\nPushing dataset to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(
            args.push_to_hub,
            private=args.private,
        )
        print(
            f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}"
        )

    # Print samples
    if all_rows:
        print("\n--- Sample entry ---")
        row = all_rows[0]
        print(f"Entry ID: {row['entry_id']}")
        print(f"Language: {row['lang']}")
        print(f"CWE: {row['cwe']}")
        print(f"Category: {row['category']}")
        print(f"Severity: {row['severity']}")
        print(f"Vulnerable code length: {len(row['vulnerable_code'])} chars")
        print(f"Secure code length: {len(row['secure_code'])} chars")
        if row["vulnerable_code"]:
            print(f"Vulnerable code preview:\n{row['vulnerable_code'][:300]}...")
        if row["secure_code"]:
            print(f"Secure code preview:\n{row['secure_code'][:300]}...")

    # Print statistics by category
    if all_rows:
        print("\n--- Samples by category ---")
        categories = {}
        for row in all_rows:
            cat = row.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
