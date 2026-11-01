#!/usr/bin/env python3
"""Tokenizer boundary gate for G7.

This script compares the tokenizer path used by the reference exporter against
the Hugging Face tokenizer for boundary prompts. It is intentionally separate
from ref_export.py so tokenizer drift can be checked without running a forward
pass.
"""

import argparse
import json
import os
import sys
from functools import lru_cache

from ref_export import parse_gguf, tokenize_prompt


CASES = [
    ("ascii", "Hello"),
    ("leading_space", " hello"),
    ("double_space", "  hello"),
    ("newline_tab", "\nHello\tworld"),
    ("cjk_mixed", "你好 Hello"),
    ("emoji", "🙂"),
    ("zwj_emoji", "👩\u200d💻"),
    ("empty", ""),
]

TINY_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SMOL_TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"

TINY_EXPECTED = {
    "ascii": [15043],
    "leading_space": [29871, 22172],
    "double_space": [259, 22172],
    "newline_tab": [29871, 13, 10994, 12, 11526],
    "cjk_mixed": [29871, 30919, 31076, 15043],
    "emoji": [29871, 243, 162, 156, 133],
    "zwj_emoji": [29871, 243, 162, 148, 172, 30722, 243, 162, 149, 190],
    "empty": [],
}

SMOL_EXPECTED = {
    "ascii": [19556],
    "leading_space": [33662],
    "double_space": [216, 33662],
    "newline_tab": [198, 19556, 197, 6693],
    "cjk_mixed": [18645, 250, 48392, 138, 38699],
    "emoji": [10813, 38887],
    "zwj_emoji": [10813, 235, 119, 321, 231, 10813, 236, 136],
    "empty": [],
}


@lru_cache(maxsize=None)
def hf_tokenizer(tokenizer_name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def hf_ids(tokenizer_name, prompt):
    return hf_tokenizer(tokenizer_name).encode(prompt, add_special_tokens=False)


def assert_equal(label, got, want, context):
    if got != want:
        raise AssertionError(
            "%s %s: got %s, want %s" % (context, label, got, want))


def check_tiny(path):
    meta, _tensors, _data_start, _data = parse_gguf(path)
    if meta.get("tokenizer.ggml.model") != "llama":
        raise AssertionError("TinyLlama tokenizer model is %r, want 'llama'"
                             % meta.get("tokenizer.ggml.model"))

    # The built-in GGUF byte-BPE path must not be used for LLaMA tokenizers.
    try:
        tokenize_prompt(meta, "Hello")
    except ValueError:
        pass
    else:
        raise AssertionError("TinyLlama tokenized without --hf-tokenizer")

    rows = []
    for label, prompt in CASES:
        ids, source = tokenize_prompt(meta, prompt, TINY_TOKENIZER)
        assert_equal(label, ids, TINY_EXPECTED[label], "tiny hf table")
        assert_equal(label, ids, hf_ids(TINY_TOKENIZER, prompt), "tiny hf live")
        rows.append({"case": label, "prompt": prompt, "ids": ids, "source": source})
    return rows


def check_smol(path):
    meta, _tensors, _data_start, _data = parse_gguf(path)
    if meta.get("tokenizer.ggml.model") != "gpt2":
        raise AssertionError("SmolLM2 tokenizer model is %r, want 'gpt2'"
                             % meta.get("tokenizer.ggml.model"))

    rows = []
    for label, prompt in CASES:
        ids, source = tokenize_prompt(meta, prompt)
        assert_equal(label, ids, SMOL_EXPECTED[label], "smol gguf table")
        assert_equal(label, ids, hf_ids(SMOL_TOKENIZER, prompt), "smol hf live")
        rows.append({"case": label, "prompt": prompt, "ids": ids, "source": source})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny-model", required=True, help="TinyLlama Q4_0 GGUF")
    ap.add_argument("--smol-model", required=True, help="SmolLM2 Q8_0 GGUF")
    ap.add_argument("--out", help="write JSON summary to this path")
    args = ap.parse_args()

    payload = {
        "tiny_model": os.path.abspath(args.tiny_model),
        "smol_model": os.path.abspath(args.smol_model),
        "cases": [label for label, _prompt in CASES],
        "tiny": check_tiny(args.tiny_model),
        "smol": check_smol(args.smol_model),
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
