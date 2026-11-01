#!/usr/bin/env python3
"""Export token ids from GGUF tokenizer metadata or an HF tokenizer.

This is a deliberately small reference tool for the tokenizer gate. It can use
the independent GGUF parser and GPT-2 byte-BPE implementation from
ref_export.py, or use transformers when --hf-tokenizer is required for models
such as LLaMA.
"""

import argparse
import json
import os
import sys

from ref_export import parse_gguf, sha256_file, tokenize_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to a GGUF file")
    ap.add_argument("--prompt", default="Hello", help="prompt text")
    ap.add_argument("--prompt-file", help="read prompt from this file")
    ap.add_argument("--hf-tokenizer",
                    help="transformers tokenizer id/path for prompt tokenisation")
    ap.add_argument("--out", help="write JSON to this path; stdout if omitted")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()
    else:
        prompt = args.prompt

    meta, tensors, data_start, _data = parse_gguf(args.model)
    token_ids, tokenizer_source = tokenize_prompt(meta, prompt, args.hf_tokenizer)
    payload = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model),
        "prompt": prompt,
        "token_ids": token_ids,
        "tokenizer_source": tokenizer_source,
        "gguf": {
            "data_start": data_start,
            "tensor_count": len(tensors),
            "metadata_keys": len(meta),
        },
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
