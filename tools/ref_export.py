#!/usr/bin/env python3
"""Export reference tensors for the llama.go correctness gates (R1).

This is the reference implementation. It shares no code with the Go engine:
it parses GGUF itself, dequantises itself, and runs the forward pass through
transformers. Any agreement between this script and the Go implementation is
therefore evidence, not a tautology.

Weight source
-------------
Weights are read from the already-downloaded GGUF file, not from a
transformers checkpoint. Downloading full-precision weights would need about
4.4 GB for TinyLlama and the machine does not have that free. The side effect
is useful: the dequantisation here is an independent implementation of the
same blocks the Go quant package decodes, so it doubles as the R0 reference
for Q4_0 and Q6_K.

Outputs (into --out dir)
-----------------------
  logits.npy        [seq, vocab] float32, full sequence
  hidden_<n>.npy    [seq, embd]   float32, output of layer n (0-based)
  tokens.json       token ids, prompt string, model metadata
  manifest.json     sha256 of every output plus the input file, so a later
                    comparison can prove it used the same bytes

Usage
-----
  python tools/ref_export.py --model /tmp/tiny-Q4_0.gguf \
      --prompt "Hello" --out /tmp/ref/hello
"""

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np


# --------------------------------------------------------------------------
# GGUF parsing. Deliberately a separate implementation from the Go reader.
# --------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"

# Metadata value types (gguf spec v3).
(VAL_UINT8, VAL_INT8, VAL_UINT16, VAL_INT16, VAL_UINT32, VAL_INT32,
 VAL_FLOAT32, VAL_BOOL, VAL_STRING, VAL_ARRAY, VAL_UINT64, VAL_INT64,
 VAL_FLOAT64) = range(13)

SCALAR_FORMATS = {
    VAL_UINT8: ("<B", 1), VAL_INT8: ("<b", 1),
    VAL_UINT16: ("<H", 2), VAL_INT16: ("<h", 2),
    VAL_UINT32: ("<I", 4), VAL_INT32: ("<i", 4),
    VAL_FLOAT32: ("<f", 4), VAL_BOOL: ("<B", 1),
    VAL_UINT64: ("<Q", 8), VAL_INT64: ("<q", 8),
    VAL_FLOAT64: ("<d", 8),
}

# Tensor element types.
(T_F32, T_F16, T_Q4_0, T_Q4_1, T_Q5_0, T_Q5_1, T_Q8_0,
 T_TQ1_0, T_TQ2_0, T_IQ4_NL, T_IQ4_XS,
 T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_Q8_K) = (
    0, 1, 2, 3, 6, 7, 8, 34, 35, 20, 23, 10, 11, 12, 13, 14, 15)

TYPE_NAMES = {
    T_F32: "f32", T_F16: "f16", T_Q4_0: "q4_0", T_Q4_1: "q4_1",
    T_Q5_0: "q5_0", T_Q5_1: "q5_1", T_Q8_0: "q8_0",
    T_Q2_K: "q2_k", T_Q3_K: "q3_k", T_Q4_K: "q4_k", T_Q5_K: "q5_k",
    T_Q6_K: "q6_k", T_Q8_K: "q8_k",
}

# GGML packs two 4-bit values per byte but orders them inside a single 32-byte
# unit, so the shift depends on the position modulo 32 (the size of the
# shuffle mask in llama.cpp). Using i % 16 here would be a plausible-looking
# bug: it type-checks, produces the right count, and scrambles the weights.
QK4_0 = 32
QK8_0 = 32
QK_K = 256
SIZE_Q4_0 = 2 + QK4_0 // 2      # 18
SIZE_Q8_0 = 2 + QK8_0           # 34
SIZE_Q6_K = 128 + 64 + 16 + 2   # 210


class Reader:
    """Minimal little-endian binary reader with explicit bounds checks."""

    def __init__(self, data, path):
        self.data = data
        self.pos = 0
        self.path = path

    def need(self, n):
        if n < 0:
            raise ValueError("negative read length %d" % n)
        if self.pos + n > len(self.data):
            raise ValueError(
                "%s: truncated at offset %d, wanted %d bytes, have %d"
                % (self.path, self.pos, n, len(self.data) - self.pos))

    def take(self, n):
        self.need(n)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        if n > 1 << 28:
            raise ValueError("%s: implausible string length %d at %d"
                             % (self.path, n, self.pos))
        return self.take(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in SCALAR_FORMATS:
            fmt, size = SCALAR_FORMATS[vtype]
            return struct.unpack(fmt, self.take(size))[0]
        if vtype == VAL_STRING:
            return self.string()
        if vtype == VAL_ARRAY:
            etype = self.u32()
            count = self.u64()
            if count > 1 << 28:
                raise ValueError("%s: implausible array length %d at %d"
                                 % (self.path, count, self.pos))
            return [self.value(etype) for _ in range(count)]
        raise ValueError("%s: unknown metadata value type %d" % (self.path, vtype))


def parse_gguf(path):
    """Return (metadata dict, tensor info list, data offset, file bytes)."""
    with open(path, "rb") as fh:
        data = fh.read()

    r = Reader(data, path)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("%s: not a GGUF file" % path)

    version = r.u32()
    if version != 3:
        raise ValueError("%s: expected GGUF v3, got %d" % (path, version))

    n_tensors = r.u64()
    n_kv = r.u64()
    if n_tensors > 1 << 24 or n_kv > 1 << 24:
        raise ValueError("%s: implausible counts %d / %d" % (path, n_tensors, n_kv))

    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        meta[key] = r.value(vtype)

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        if n_dims > 8:
            raise ValueError("%s: tensor %r has %d dims" % (path, name, n_dims))
        dims = [r.u64() for _ in range(n_dims)]
        ttype = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype,
                        "offset": offset})

    alignment = int(meta.get("general.alignment", 32))
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("%s: alignment %d is not a power of two" % (path, alignment))

    # The tensor data section starts at the next multiple of `alignment` after
    # the info blocks.
    data_start = r.pos
    pad = (-data_start) % alignment
    data_start += pad
    if data_start > len(data):
        raise ValueError("%s: data section starts past end of file "
                         "(%d > %d)" % (path, data_start, len(data)))

    return meta, tensors, data_start, data


# --------------------------------------------------------------------------
# Dequantisation. Independent of the Go implementation on purpose.
# --------------------------------------------------------------------------

def f16_to_f32(bits):
    """IEEE-754 binary16 array -> float32 array. No flush-to-zero."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


def dequant_q4_0(raw, n):
    """y[i] = (nibble - 8) * d, nibble order as ggml (shift depends on i%32)."""
    nb = n // QK4_0
    if len(raw) != nb * SIZE_Q4_0:
        raise ValueError("q4_0: payload %d bytes, want %d" % (len(raw), nb * SIZE_Q4_0))
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, SIZE_Q4_0)
    d = f16_to_f32(blocks[:, 0:2].copy().view(np.uint16).ravel())
    qs = blocks[:, 2:]                      # [nb, 16]

    # ggml stores 32 nibbles in 16 bytes: byte j holds element j (low) and
    # element j+16 (high). Reconstruct in that order rather than interleaving.
    lo = (qs & 0x0F).astype(np.float32)
    hi = ((qs >> 4) & 0x0F).astype(np.float32)
    out = np.empty((nb, QK4_0), dtype=np.float32)
    out[:, :16] = lo
    out[:, 16:] = hi
    out = (out - 8.0) * d[:, None]
    return out.reshape(-1)


def dequant_q8_0(raw, n):
    """y[i] = int8(qs[i]) * d."""
    nb = n // QK8_0
    if len(raw) != nb * SIZE_Q8_0:
        raise ValueError("q8_0: payload %d bytes, want %d" % (len(raw), nb * SIZE_Q8_0))
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, SIZE_Q8_0)
    d = f16_to_f32(blocks[:, 0:2].copy().view(np.uint16).ravel())
    qs = blocks[:, 2:].copy().view(np.int8).astype(np.float32)   # [nb, 32]
    return (qs * d[:, None]).reshape(-1)


def dequant_q6_k(raw, n):
    """Q6_K: 256 weights per 210-byte super-block.

    Layout per super-block:
        ql[128]      lower 4 bits of every weight
        qh[64]       upper 2 bits
        scales[16]   int8, one per 16-weight group
        d            fp16 super-block scale

    Decoding follows ggml's dequantize_row_q6_K. The 256 weights are produced
    in two halves of 128; within a half, index l in [0, 32) writes to
    l, l+32, l+64 and l+96. Those four are NOT adjacent, and getting that
    wrong leaves three quarters of the block unwritten - which np.empty will
    happily fill with garbage that still has the right array length.
    """
    nb = n // QK_K
    if len(raw) != nb * SIZE_Q6_K:
        raise ValueError("q6_k: payload %d bytes, want %d" % (len(raw), nb * SIZE_Q6_K))

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(nb, SIZE_Q6_K)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)  # [nb,16]
    d = f16_to_f32(blocks[:, 208:210].copy().view(np.uint16).ravel())

    # zeros, not empty: an unwritten slot must be a loud wrong answer, never
    # leftover heap bytes.
    out = np.zeros((nb, QK_K), dtype=np.float32)

    for half in range(2):
        ql_h = ql[:, half * 64:(half + 1) * 64]     # [nb,64]
        qh_h = qh[:, half * 32:(half + 1) * 32]     # [nb,32]
        off = half * 128
        sc_h = sc[:, half * 8:(half + 1) * 8]       # [nb,8]

        for l in range(32):
            is_ = l // 16
            q1 = (ql_h[:, l] & 0x0F).astype(np.int32) \
                | ((qh_h[:, l] >> 0) & 0x03).astype(np.int32) << 4
            q2 = (ql_h[:, l + 32] & 0x0F).astype(np.int32) \
                | ((qh_h[:, l] >> 2) & 0x03).astype(np.int32) << 4
            q3 = ((ql_h[:, l] >> 4) & 0x0F).astype(np.int32) \
                | ((qh_h[:, l] >> 4) & 0x03).astype(np.int32) << 4
            q4 = ((ql_h[:, l + 32] >> 4) & 0x0F).astype(np.int32) \
                | ((qh_h[:, l] >> 6) & 0x03).astype(np.int32) << 4

            out[:, off + l + 0] = d * sc_h[:, is_ + 0] * (q1 - 32)
            out[:, off + l + 32] = d * sc_h[:, is_ + 2] * (q2 - 32)
            out[:, off + l + 64] = d * sc_h[:, is_ + 4] * (q3 - 32)
            out[:, off + l + 96] = d * sc_h[:, is_ + 6] * (q4 - 32)

    return out.reshape(-1)


DEQUANT = {
    T_F32: lambda raw, n: np.frombuffer(raw, dtype="<f4").astype(np.float32),
    T_F16: lambda raw, n: f16_to_f32(np.frombuffer(raw, dtype="<u2")),
    T_Q4_0: dequant_q4_0,
    T_Q8_0: dequant_q8_0,
    T_Q6_K: dequant_q6_k,
}


def load_tensor(tensors, data, data_start, name):
    """Return (float32 array in ggml order, dims, type name)."""
    for t in tensors:
        if t["name"] == name:
            break
    else:
        raise KeyError("no tensor named %r" % name)

    n = 1
    for d in t["dims"]:
        n *= d

    ttype = t["type"]
    if ttype not in DEQUANT:
        raise NotImplementedError(
            "tensor %r has type %s, which this script cannot dequantise"
            % (name, TYPE_NAMES.get(ttype, ttype)))

    fn = DEQUANT[ttype]
    if ttype == T_F32:
        need = 4 * n
    elif ttype == T_F16:
        need = 2 * n
    elif ttype == T_Q4_0:
        need = (n // QK4_0) * SIZE_Q4_0
    elif ttype == T_Q8_0:
        need = (n // QK8_0) * SIZE_Q8_0
    elif ttype == T_Q6_K:
        need = (n // QK_K) * SIZE_Q6_K
    else:
        raise AssertionError(ttype)

    start = data_start + t["offset"]
    if start + need > len(data):
        raise ValueError("tensor %r payload [%d,+%d) exceeds file (%d)"
                         % (name, start, need, len(data)))

    return fn(data[start:start + need], n), t["dims"], TYPE_NAMES.get(ttype, str(ttype))


def tensor_map(tensors):
    out = {}
    for t in tensors:
        if t["name"] in out:
            raise ValueError("duplicate tensor name %r" % t["name"])
        out[t["name"]] = t
    return out


def reshape_gguf(values, dims, hf_shape, name):
    """Reshape GGUF flat values to the matching transformers parameter shape.

    GGUF stores dimensions as ne[0], ne[1], ... where ne[0] is the innermost,
    contiguous dimension. A 2-D linear weight with dims [in, out] is therefore
    already laid out row-major as [out, in], exactly like torch nn.Linear.
    There is no transpose here.
    """
    if len(dims) == 1:
        got = (int(dims[0]),)
    elif len(dims) == 2:
        got = (int(dims[1]), int(dims[0]))
    else:
        raise ValueError("%s: unsupported GGUF rank %d" % (name, len(dims)))

    want = tuple(int(x) for x in hf_shape)
    if got != want:
        raise ValueError("%s: GGUF dims %s reshape to %s, HF wants %s"
                         % (name, dims, got, want))
    return values.reshape(got)


def hf_name_map(n_layer, has_output):
    yield "token_embd.weight", "model.embed_tokens.weight"
    for i in range(n_layer):
        p = "blk.%d." % i
        q = "model.layers.%d." % i
        yield p + "attn_norm.weight", q + "input_layernorm.weight"
        yield p + "attn_q.weight", q + "self_attn.q_proj.weight"
        yield p + "attn_k.weight", q + "self_attn.k_proj.weight"
        yield p + "attn_v.weight", q + "self_attn.v_proj.weight"
        yield p + "attn_output.weight", q + "self_attn.o_proj.weight"
        yield p + "ffn_norm.weight", q + "post_attention_layernorm.weight"
        yield p + "ffn_gate.weight", q + "mlp.gate_proj.weight"
        yield p + "ffn_up.weight", q + "mlp.up_proj.weight"
        yield p + "ffn_down.weight", q + "mlp.down_proj.weight"
    yield "output_norm.weight", "model.norm.weight"
    if has_output:
        yield "output.weight", "lm_head.weight"


def parse_layer_selection(spec, n_layer):
    if spec == "none":
        return []
    if spec == "all":
        return list(range(n_layer))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if i < 0 or i >= n_layer:
            raise ValueError("layer index %d outside [0,%d)" % (i, n_layer))
        out.append(i)
    return out


def parse_token_ids(text):
    if text is None:
        return None
    out = []
    for part in text.replace(",", " ").split():
        out.append(int(part))
    if not out:
        raise ValueError("--token-ids was provided but no ids were parsed")
    return out


def gpt2_byte_encoder():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tokenize_bytes_bpe(meta, text):
    """Small byte-BPE tokenizer for GGUF tokenizer metadata.

    This is intentionally independent of transformers. It relies on the regex
    package for GPT-2 compatible pre-tokenisation, then uses
    tokenizer.ggml.tokens plus tokenizer.ggml.merges. If the model metadata
    omits add_bos_token, no BOS is invented.
    """
    vocab = {tok: i for i, tok in enumerate(meta.get("tokenizer.ggml.tokens", []))}
    merges = meta.get("tokenizer.ggml.merges", [])
    ranks = {}
    for rank, merge in enumerate(merges):
        if not isinstance(merge, str) or " " not in merge:
            continue
        left, right = merge.split(" ", 1)
        ranks[(left, right)] = rank

    unk_id = int(meta.get("tokenizer.ggml.unknown_token_id", 0))
    add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", False))
    bos_id = int(meta.get("tokenizer.ggml.bos_token_id", 1))

    import regex

    pat = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    encoder = gpt2_byte_encoder()
    ids = []
    for token in regex.findall(pat, text):
        pieces = [encoder[b] for b in token.encode("utf-8")]
        while len(pieces) > 1:
            best = None
            best_rank = None
            for i in range(len(pieces) - 1):
                rank = ranks.get((pieces[i], pieces[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best = i
                    best_rank = rank
            if best is None:
                break
            merged = pieces[best] + pieces[best + 1]
            pieces = pieces[:best] + [merged] + pieces[best + 2:]
        ids.extend(vocab.get(piece, unk_id) for piece in pieces)
    if add_bos:
        ids.insert(0, bos_id)
    return ids


def tokenize_prompt(meta, prompt, hf_tokenizer=None):
    if hf_tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_tokenizer, use_fast=True)
        return tok.encode(prompt, add_special_tokens=False), "hf:%s" % hf_tokenizer

    model = meta.get("tokenizer.ggml.model")
    if model == "gpt2":
        return tokenize_bytes_bpe(meta, prompt), "gguf:gpt2-bpe"

    raise ValueError(
        "tokenizer.ggml.model=%r is not supported by the built-in tokenizer; "
        "pass --hf-tokenizer or --token-ids" % model)


def build_config(meta, tensors_by_name, model_layers):
    from transformers import LlamaConfig

    arch = meta.get("general.architecture", "llama")
    n_layer_file = int(meta.get("%s.block_count" % arch))
    n_layer = n_layer_file if model_layers is None else int(model_layers)
    if n_layer <= 0 or n_layer > n_layer_file:
        raise ValueError("--model-layers must be in [1,%d], got %d"
                         % (n_layer_file, n_layer))

    embd_dims = tensors_by_name["token_embd.weight"]["dims"]
    vocab = int(embd_dims[1])
    has_output = "output.weight" in tensors_by_name

    return LlamaConfig(
        vocab_size=vocab,
        hidden_size=int(meta.get("%s.embedding_length" % arch)),
        intermediate_size=int(meta.get("%s.feed_forward_length" % arch)),
        num_hidden_layers=n_layer,
        num_attention_heads=int(meta.get("%s.attention.head_count" % arch)),
        num_key_value_heads=int(meta.get("%s.attention.head_count_kv" % arch)),
        max_position_embeddings=int(meta.get("%s.context_length" % arch)),
        rms_norm_eps=float(meta.get("%s.attention.layer_norm_rms_epsilon" % arch)),
        rope_theta=float(meta.get("%s.rope.freq_base" % arch)),
        tie_word_embeddings=not has_output,
        bos_token_id=int(meta.get("tokenizer.ggml.bos_token_id", 1)),
        eos_token_id=int(meta.get("tokenizer.ggml.eos_token_id", 2)),
        pad_token_id=int(meta.get("tokenizer.ggml.eos_token_id", 2)),
        attention_bias=False,
        mlp_bias=False,
    )


def stream_weights(model, tensors, data, data_start):
    import torch

    tensors_by_name = tensor_map(tensors)
    params = dict(model.named_parameters())
    written = set()
    has_output = "output.weight" in tensors_by_name

    for gguf_name, hf_name in hf_name_map(model.config.num_hidden_layers, has_output):
        if hf_name not in params:
            raise KeyError("HF model has no parameter %r for GGUF %r"
                           % (hf_name, gguf_name))
        values, dims, type_name = load_tensor(tensors, data, data_start, gguf_name)
        param = params[hf_name]
        arr = reshape_gguf(values, dims, tuple(param.shape), gguf_name)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        with torch.no_grad():
            param.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
        written.add(hf_name)
        print("loaded %-34s -> %-48s %s %s"
              % (gguf_name, hf_name, tuple(param.shape), type_name), flush=True)

    missing = sorted(set(params) - written)
    if missing:
        raise ValueError("unmapped HF parameters: %s" % ", ".join(missing))
    return written


def write_numpy(path, arr):
    np.save(path, np.asarray(arr, dtype=np.float32))
    return sha256_file(path)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a GGUF file")
    ap.add_argument("--prompt", default="Hello", help="prompt text")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--layers", default="all",
                    help="'all', 'none', or comma-separated layer indices to dump")
    ap.add_argument("--model-layers", type=int,
                    help="instantiate only the first N layers for smoke tests")
    ap.add_argument("--token-ids",
                    help="explicit token ids; overrides prompt tokenisation")
    ap.add_argument("--hf-tokenizer",
                    help="transformers tokenizer id/path for prompt tokenisation")
    ap.add_argument("--prompt-file", help="read the prompt from this file instead")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()
    else:
        prompt = args.prompt

    os.makedirs(args.out, exist_ok=True)

    meta, tensors, data_start, data = parse_gguf(args.model)
    print("parsed %s: %d tensors, %d metadata keys, data at %d"
          % (os.path.basename(args.model), len(tensors), len(meta), data_start),
          flush=True)
    tensors_by_name = tensor_map(tensors)

    arch = meta.get("general.architecture", "llama")
    n_layer = meta.get("%s.block_count" % arch)
    n_embd = meta.get("%s.embedding_length" % arch)
    n_head = meta.get("%s.attention.head_count" % arch)
    n_head_kv = meta.get("%s.attention.head_count_kv" % arch, n_head)
    n_ff = meta.get("%s.feed_forward_length" % arch)
    eps = meta.get("%s.attention.layer_norm_rms_epsilon" % arch)
    n_ctx = meta.get("%s.context_length" % arch)

    rope_theta = meta.get("%s.rope.freq_base" % arch)
    print("arch=%s layers=%s embd=%s heads=%s/%s ffn=%s rms_eps=%s rope_theta=%s"
          % (arch, n_layer, n_embd, n_head, n_head_kv, n_ff, eps, rope_theta),
          flush=True)

    # token_embd.weight has dims [n_embd, vocab] in GGML order: Dims[0] is the
    # innermost (fastest-varying) dimension, so the vocabulary is Dims[1].
    # Taking Dims[0] silently yields n_embd, which looks plausible (2048) and
    # would make every logits shape assertion fail for the wrong reason.
    embd_dims = tensors_by_name["token_embd.weight"]["dims"]
    if len(embd_dims) != 2:
        raise ValueError("token_embd.weight has %d dims, expected 2" % len(embd_dims))
    vtype_name = TYPE_NAMES.get(
        next(t["type"] for t in tensors if t["name"] == "token_embd.weight"), "?")
    vocab = embd_dims[1]
    print("token_embd: dims=%s type=%s -> vocab=%d"
          % (embd_dims, vtype_name, vocab), flush=True)

    # Cross-check against the tokenizer metadata, which is an independent
    # statement of the vocabulary size. A mismatch means the tensor table and
    # the metadata disagree and any logits comparison would be meaningless.
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens is not None and len(tokens) != vocab:
        raise ValueError(
            "vocab mismatch: token_embd dim1 = %d but tokenizer.ggml.tokens "
            "has %d entries" % (vocab, len(tokens)))
    print("vocab cross-check against tokenizer.ggml.tokens: OK", flush=True)

    token_ids = parse_token_ids(args.token_ids)
    tokenizer_source = "explicit"
    if token_ids is None:
        token_ids, tokenizer_source = tokenize_prompt(meta, prompt, args.hf_tokenizer)
    if not token_ids:
        raise ValueError("prompt tokenised to an empty sequence")
    print("tokens (%s): %s" % (tokenizer_source, token_ids), flush=True)

    print("importing torch/transformers...", flush=True)
    import torch
    from transformers import LlamaForCausalLM

    cfg = build_config(meta, tensors_by_name, args.model_layers)
    print("building LlamaForCausalLM with %d/%d layers..."
          % (cfg.num_hidden_layers, n_layer), flush=True)
    model = LlamaForCausalLM(cfg).eval()

    print("streaming GGUF weights into HF parameters...", flush=True)
    written = stream_weights(model, tensors, data, data_start)

    dump_layers = parse_layer_selection(args.layers, cfg.num_hidden_layers)
    hidden = {}
    hooks = []
    for i in dump_layers:
        def save_hidden(_module, _inputs, output, layer=i):
            hs = output[0] if isinstance(output, tuple) else output
            hidden[layer] = hs.detach().to(torch.float32).cpu().numpy()[0]
        hooks.append(model.model.layers[i].register_forward_hook(save_hidden))

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print("running forward seq=%d..." % len(token_ids), flush=True)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    hashes = {}
    logits_path = os.path.join(args.out, "logits.npy")
    hashes["logits.npy"] = write_numpy(logits_path, out.logits[0].detach().cpu().numpy())
    print("wrote %s" % logits_path, flush=True)

    for i in dump_layers:
        if i not in hidden:
            raise RuntimeError("hook for layer %d did not fire" % i)
        name = "hidden_%d.npy" % i
        hashes[name] = write_numpy(os.path.join(args.out, name), hidden[i])
        print("wrote %s" % os.path.join(args.out, name), flush=True)

    tokens_path = os.path.join(args.out, "tokens.json")
    with open(tokens_path, "w", encoding="utf-8") as fh:
        json.dump({"prompt": prompt, "token_ids": token_ids}, fh,
                  indent=2, ensure_ascii=False)
    hashes["tokens.json"] = sha256_file(tokens_path)

    manifest = {
        "model_path": os.path.abspath(args.model),
        "model_sha256": sha256_file(args.model),
        "prompt": prompt,
        "token_ids": token_ids,
        "tokenizer_source": tokenizer_source,
        "script": os.path.abspath(__file__),
        "gguf": {
            "version": 3,
            "data_start": data_start,
            "tensor_count": len(tensors),
            "metadata_keys": len(meta),
        },
        "hparams": {
            "arch": arch, "n_layer": n_layer, "n_embd": n_embd,
            "n_head": n_head, "n_head_kv": n_head_kv, "n_ff": n_ff,
            "rms_eps": eps, "n_ctx": n_ctx, "vocab": vocab,
            "rope_theta": rope_theta,
            "tie_word_embeddings": "output.weight" not in tensors_by_name,
            "model_layers": cfg.num_hidden_layers,
        },
        "mapped_parameters": sorted(written),
        "outputs": hashes,
        "tensor_types": {
            TYPE_NAMES.get(t["type"], str(t["type"])): 0 for t in tensors
        },
    }
    for t in tensors:
        name = TYPE_NAMES.get(t["type"], str(t["type"]))
        manifest["tensor_types"][name] = manifest["tensor_types"].get(name, 0) + 1

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("wrote %s" % os.path.join(args.out, "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
