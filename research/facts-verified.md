# 已核实事实清单（Coordinator 直接取证，2026-10-27）

> 说明：本次调研子代理在声明"Now writing the brief"后结束，未产出文件（已核实：`/home/ubuntu/.openclaw/workspace/team-researcher/` 下无 `brief-mvp-research.md`）。
> 为避免空转重试，Coordinator 直接从一手来源取证，下列每条均可复核。标注 **[已核实]** = 本次实际读取远端源；**[推断]** = 未直接取证。

## 1. 宿主环境 [已核实]

| 项 | 值 | 证据 |
|---|---|---|
| 架构 | aarch64 (arm64) | `uname -m` |
| CPU | 4 核 | `nproc` |
| 内存 | 3 GiB total | `free -g` |
| Go 工具链 | **未安装** | `go version` → not found |
| llama.cpp 源码 | 本地无 checkout | `ls ~/llama.cpp` 为空 |
| 网络 | 可访问 huggingface.co / raw.githubusercontent.com | curl 200 |

**影响**：M0 必须先装 Go 与构建参考实现；3 GiB 内存是 MVP 模型的硬约束（见 §3）。

## 2. MVP 模型候选（config.json 原始值）[已核实]

### TinyLlama-1.1B-Chat-v1.0
来源：`https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json`

```
architectures: LlamaForCausalLM   model_type: llama
hidden_size: 2048   intermediate_size: 5632
num_hidden_layers: 22
num_attention_heads: 32   num_key_value_heads: 4      → GQA 8:1
vocab_size: 32000
max_position_embeddings: 2048
rms_norm_eps: 1e-05   rope_theta: 10000.0   rope_scaling: null
tie_word_embeddings: false
attention_bias: false   hidden_act: silu
bos_token_id: 1   eos_token_id: 2
torch_dtype: bfloat16
```

`tokenizer_config.json` [已核实]：
```
tokenizer_class: LlamaTokenizer, legacy: false
chat_template: "{% for message in messages %}{% if message['role']=='user' %}{{ '<|user|>\n' + message['content'] + eos_token }}...{% if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}{% endfor %}"
特殊 token: 0=<unk> 1=<s> 2=</s>   无 BOS 前缀出现在模板中
```

### SmolLM2-135M-Instruct（建议作为 CI 小模型）
来源：`https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json`

```
architectures: LlamaForCausalLM   model_type: llama
hidden_size: 576   intermediate_size: 1536
num_hidden_layers: 30
num_attention_heads: 9   num_key_value_heads: 3      → GQA 3:1
vocab_size: 49152
max_position_embeddings: 8192
rms_norm_eps: 1e-05   rope_theta: 100000
tie_word_embeddings: true        ← 无独立 output.weight，必须验证 tied embedding
```

**关键洞察**：两者同为 `LlamaForCausalLM`。因此"一个 Llama 前向实现"覆盖两个模型，但 SmolLM2 额外压测 tied-embedding 与 rope_theta=100000 路径，是廉价的正交回归。TinyLlama 权重 2.2 GB (bfloat16 safetensors) [已核实文件大小]，Q4_0 约 0.6 GB，可放进 3 GiB 宿主。

## 3. GGUF 格式 [已核实]

来源：`https://raw.githubusercontent.com/ggerganov/ggml/master/docs/gguf.md`（828 行）

```
struct gguf_header_t {
    uint32_t magic;              // 'G','G','U','F'
    uint32_t version;            // 3 = 当前规范（含 big-endian 支持）
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};
struct gguf_tensor_info_t {
    gguf_string_t name;          // 最长 64 字节
    uint32_t n_dimensions;       // 目前 ≤ 4
    uint64_t dimensions[n_dimensions];
    ggml_type type;
    uint64_t offset;             // 相对 tensor_data，必须 ALIGNMENT 对齐
};
// header + tensor_infos 后补齐到 ALIGNMENT；ALIGNMENT 取自 general.alignment
```

每条 tensor 数据偏移必须为 `ALIGNMENT` 的倍数，tensor 之间补齐 0x00。

## 4. 量化块布局与反量化公式 [已核实]

来源：`ggml/src/ggml-common.h`、`ggml/src/ggml-quants.c`

```c
#define QK4_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;   // sizeof = 2 + 16 = 18 B
#define QK8_0 32
typedef struct { ggml_half d; int8_t  qs[QK8_0];  } block_q8_0;   // sizeof = 2 + 32 = 34 B
```

反量化（逐字核对 ggml-quants.c）：
```c
// q4_0
const float d = GGML_FP16_TO_FP32(x[i].d);
for (j = 0; j < qk/2; ++j) {
    const int x0 = (x[i].qs[j] & 0x0F) - 8;
    const int x1 = (x[i].qs[j] >>   4) - 8;
    y[i*qk + j]        = x0*d;
    y[i*qk + j + qk/2] = x1*d;      // 低半字节 → 前半，高半字节 → 后半
}
// q8_0
const float d = GGML_FP16_TO_FP32(x[i].d);
for (j = 0; j < qk; ++j) y[i*qk + j] = x[i].qs[j]*d;
```

**易错点**：q4_0 的 qs 第 j 字节同时承载下标 j 与 j+16 两个权重，不是相邻两个。写错会得到"看起来合理但数值全错"的结果。

## 5. llama.cpp 张量命名（Llama 架构）[已核实]

来源：`src/llama-arch.cpp`（tensor name 表）

| 名称 | 说明 |
|---|---|
| `token_embd.weight` | [vocab, hidden] |
| `blk.%d.attn_norm.weight` | RMSNorm |
| `blk.%d.attn_q.weight` | [n_head*head_dim, hidden] |
| `blk.%d.attn_k.weight` | [n_head_kv*head_dim, hidden] |
| `blk.%d.attn_v.weight` | [n_head_kv*head_dim, hidden] |
| `blk.%d.attn_output.weight` | [hidden, n_head*head_dim] |
| `blk.%d.ffn_norm.weight` | RMSNorm |
| `blk.%d.ffn_gate.weight` | w1 |
| `blk.%d.ffn_up.weight` | w3 |
| `blk.%d.ffn_down.weight` | w2 |
| `output_norm.weight` | final RMSNorm |
| `output.weight` | 可能缺失（tied） |

## 6. 待补证项（[推断]，需 M0 期间核实）

1. `general.alignment` 的默认值（规范描述为"由该字段指定"，llama.cpp 写默认 32；**未直接读取写入端代码**）。
2. `llama-cli` 是否提供 logits dump 类参数（疑为 `--logits-file` / `--logits-all`，需读 `common/arg.cpp` 确认；若无，退化为 HF transformers f32 参考）。
3. GGUF 中 `tokenizer.ggml.merges` / `tokenizer.ggml.tokens` / `tokenizer.ggml.scores` 的确切键名与 BPE "legacy" 合并规则细节。
4. TinyLlama 官方 Q4_0/Q4_K_M GGUF 是否有可信发布者（TheBloke 归档可用性）。
