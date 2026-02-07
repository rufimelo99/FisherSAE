import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import einops
import litellm
import pandas as pd
import torch
from datasets import load_dataset
from pydantic import BaseModel
from tqdm import tqdm, trange

from code_sae.logger import logger


class CWENLExplanation(BaseModel):
    vuln_id: str
    cwe: str
    vulnerability_explanation: str
    fix_explanation: str
    summary: str

    def append_to_jsonl(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if path.exists() else "w"
        with path.open(mode) as f:
            f.write(json.dumps(self.model_dump()) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Natural Language Explanations for CWEs"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name for grabbing explanations"
    )
    parser.add_argument(
        "--hf_path",
        type=str,
        default="rufimelo/DeltaSecommits",
        help="Hugging Face repo ID (e.g., 'username/sae-qwen-layer0')",
    )
    parser.add_argument(
        "--secure_column",
        type=str,
        default="after_version",
        help="Secure Code Column name in HF Repo",
    )
    parser.add_argument(
        "--vulnerable_column",
        type=str,
        default="prior_version",
        help="Vulnerable Code Column name in HF Repo",
    )

    parser.add_argument(
        "--cwe_column", type=str, default="cwe", help="CWE Column name in HF Repo"
    )

    args = parser.parse_args()

    hf_path = args.hf_path

    before_func_col = args.vulnerable_column
    after_func_col = args.secure_column
    model_arg = args.model
    cwe_col = args.cwe_column

    MSR_df = load_dataset(hf_path, split="train").to_pandas()

    for i in trange(len(MSR_df)):
        # for i in trange(200):  # Limit to 200 samples for faster testing
        secure_code = str(MSR_df.iloc[i][before_func_col])
        vulnerable_code = str(MSR_df.iloc[i][after_func_col])
        cwe = str(MSR_df.iloc[i][cwe_col])

        AZUREAI_OPENAI_BASE_URL = os.environ.get("AZUREAI_OPENAI_BASE_URL")
        AZUREAI_OPENAI_API_KEY = os.environ.get("AZUREAI_OPENAI_API_KEY")
        model = os.environ.get(model_arg, "azure/gpt-4.1-mini")

        if not AZUREAI_OPENAI_BASE_URL:
            raise RuntimeError("Missing AZUREAI_OPENAI_BASE_URL env var")
        if not AZUREAI_OPENAI_API_KEY:
            raise RuntimeError("Missing AZUREAI_OPENAI_API_KEY env var")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a software security expert specializing in analyzing code vulnerabilities and generating natural language explanations. "
                    "You will receive two different code snippets: one that is vulnerable and one that has been fixed. Your task is to analyze the differences between the two snippets and generate a clear, concise explanation of the vulnerability, how it can be exploited. Then, you shall responde with a natural language explanation of the vulnerability and the same for the fix, on why it is secure."
                    "Respond ONLY with valid JSON matching this schema:\n"
                    "{\n"
                    '  "vulnerability_explanation": string,\n'
                    '  "fix_explanation": string,\n'
                    '  "summary": string,\n'
                    "}\n"
                    "Do not include explanations, markdown, or extra text."
                ),
            },
            {
                "role": "user",
                "content": f"Here is the vulnerable code snippet:\n```\n{vulnerable_code}\n```\n\nAnd here is the fixed code snippet:\n```\n{secure_code}\n```\n\nThe vulnerability is categorized as: {cwe}.\n\nPlease provide your analysis and explanations in the required JSON format.",
            },
        ]

        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=AZUREAI_OPENAI_API_KEY,
            api_base=AZUREAI_OPENAI_BASE_URL,
            response_format={"type": "json_object"},
        )
        print("Full model response:", response, flush=True)

        raw_content = response.choices[0].message.content
        print("Raw model response:", raw_content, flush=True)

        try:
            explanation_data = json.loads(raw_content)
            cwe_explanation = CWENLExplanation(
                vuln_id=f"{hf_path}_{i}",
                cwe=cwe,
                vulnerability_explanation=explanation_data.get(
                    "vulnerability_explanation", ""
                ),
                fix_explanation=explanation_data.get("fix_explanation", ""),
                summary=explanation_data.get("summary", ""),
            )
            cwe_explanation.append_to_jsonl("cwe_nl_explanations.jsonl")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for vuln_id {hf_path}_{i}: {e}")
            logger.error(f"Model response that caused the error: {raw_content}")
            continue


if __name__ == "__main__":
    main()
