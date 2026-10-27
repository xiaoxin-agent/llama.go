# go-llm：CPU-only Go 推理引擎 —— 技术设计与实施计划（草案 v0.1 · 待批准）

> 状态：**计划草案，未开始实现**。按约束"用户批准前不开展大规模实现"，本阶段仅产出文档。
> 一手取证清单见 `research/facts-verified.md`；本文件中标 **[已核实]** 的数值来自该清单，标 **[待证]** 的将在 M0 关闭。

---

## 0. 一句话目标

用纯 Go（无 cgo）实现一个单文件 GGUF 加载 + Llama 架构 CPU 前向 + 采样 + 文本生成的推理引擎，
**先证明数值与可信参考实现一致，再谈性能**；全部完成声明必须附带可复现的测试输出。

---

## 1. 范围与非目标

### 范围内（v1）
- GGUF v3 解析（little-endian，含 `general.alignment`）
- 反量化：`F32`、`F16`、`Q8_0`、`Q4_0`（MVP 必需）；`Q4_K`/`Q6_K` 列为 v1.1 可选
- Llama 架构前向：RMSNorm、RoPE、GQA、SwiGLU、pre-norm 残差
- KV cache（单序列、连续分配）
- 采样：greedy / temperature / top-k / top-p / repetition penalty
- BPE tokenizer（SentencePiece 风格，byte fallback）+ chat template 最小实现（仅 TinyLlama 模板）
- CLI：`go-llm run -m model.gguf -p "..."`，以及 `go-llm logits` 导出调试子命令
- 数值验证工具：`go-llm compare`（对比参考 logits）

### 明确不做（v1）
- 训练 / 微调 / LoRA
- CUDA / Metal / Vulkan / SIMD 手写汇编（Go 编译器自动向量化为上限）
- 批处理（batch > 1）、并发请求、连续 batching
- 除 Llama 外的任何架构（Qwen2 只允许作为**未来**扩展点，不在 v1）
- 模型量化工具（只消费官方 GGUF）
- 完整的 Jinja2 chat template 引擎（只支持硬编码的 TinyLlama 模板 + 模板直通）

### 冻结的技术选型
| 决策 | 选择 | 理由 |
|---|---|---|
| 语言 | Go ≥ 1.22 | 泛型、`slices`/`maps`、稳定 toolchain |
| cgo | **禁止** | 纯 Go 可交叉编译、单二进制、避免 Arm 上 toolchain 折腾 |
| 浮点 | `float32` 全链路 | 与 HF f32 参考可比；避免 f16 累加误差 |
| GGUF | 仅 v3，little-endian | 训练/转换端默认输出；big-endian 只做显式报错 |
| 张量布局 | 行主序、GGML 的"最右维连续"，加载时统一转成 `[out][in]` 行主序 f32 | 简化 matmul 实现；MMap 层单独处理 |
| 内存 | 支持 `mmap`（`golang.org/x/exp/mmap` 或 syscall）与全量读入两种模式 | 3 GiB 宿主上 TinyLlama Q4_0 约 0.6 GB，两者都可行 |
| 测试框架 | 标准 `testing` + `testdata/` golden 文件 | 零依赖 |

---

## 2. MVP 模型选择

### 主模型：TinyLlama-1.1B-Chat-v1.0

依据（全部 **[已核实]** 自 HF config.json）：

| 属性 | 值 | 对实现的意义 |
|---|---|---|
| arch | `LlamaForCausalLM` | 目标架构就是最经典的 Llama |
| hidden / inter | 2048 / 5632 | 单步计算量适中，CPU 可交互 |
| layers | 22 | 调试时逐层 dump 可行 |
| heads / kv heads | 32 / 4（GQA 8:1） | **必须**实现 GQA，否则形状都不对 |
| vocab | 32000 | embedding 表 32000×2048 f32 = 262 MB，需量化加载 |
| rope_theta | 10000 | 标准 RoPE |
| tie_embeddings | **false** | 有独立 `output.weight`，可先不做 tied 逻辑 |
| rms_norm_eps | 1e-5 | 硬编码为默认，从 GGUF metadata 读 |
| 权重文件 | 2.2 GB bf16 → **Q4_0 GGUF ≈ 0.6 GB** | 适配 3 GiB 宿主 |

**为什么不是别的**：
- Qwen2.5-0.5B：需要额外 qkv bias 路径，扩大 MVP 面
- Llama-3.2-1B：rope_scaling/theta=500000 + 128K vocab，引入非必要复杂度
- SmolLM2-135M：层数多（30）但每层极窄，更适合当**回归测试**而非主演示模型

### CI 小模型：SmolLM2-135M-Instruct（Q8_0，约 145 MB）

**[已核实]** 它同样是 `LlamaForCausalLM`，但有两个正交差异，恰好压测我们的代码路径：
1. `tie_word_embeddings: true` → 无 `output.weight`，必须复用 `token_embd.weight`
2. `rope_theta: 100000` → 验证 theta 不是硬编码
3. `num_attention_heads: 9`（非 2 的幂）、kv=3 → 验证 GQA 的整数除法与非 2 幂 head_dim 处理

> 若 135M 的 GGUF 在 HF 上不可得，备选：用 `convert_hf_to_gguf.py` 自行转换（M0 验证可行性）。

---

## 3. 仓库布局

```
go-llm/
├── go.mod                          module github.com/xiaoxin-agent/llama.go
├── README.md
├── Makefile                        build / test / test-race / bench / verify
├── cmd/
│   ├── go-llm/main.go              CLI 入口（run / logits / compare / info）
│   └── gguf-inspect/main.go        独立小工具：打印 GGUF metadata 与 tensor 表
├── gguf/
│   ├── reader.go                   Reader：header、KV、tensor info
│   ├── types.go                    GGUFValueType 枚举、Value union
│   ├── metadata.go                 已知 key 的强类型访问器（llama.* 命名空间）
│   ├── tensor.go                   TensorInfo、DataOffset 计算、分块读取
│   └── *_test.go
├── quant/
│   ├── dequant.go                  F32/F16/Q8_0/Q4_0 → []float32
│   ├── blocks.go                   块结构体（与 ggml 字节布局一一对应）
│   └── *_test.go                   golden 用例（见 §6）
├── tensor/
│   ├── tensor.go                   稠密 f32 张量（shape + strides + data）
│   ├── matmul.go                   GEMM / GEMV（行主序）
│   ├── ops.go                      rmsnorm / softmax / silu / add / mul
│   └── *_test.go
├── model/
│   ├── config.go                   HParams（从 GGUF metadata 构造）
│   ├── loader.go                   按命名规则装配权重、校验形状
│   ├── llama.go                    Llama 前向：Forward(tokens) → logits
│   ├── kvcache.go                  KV cache
│   └── *_test.go
├── tokenizer/
│   ├── bpe.go                      SentencePiece 风格 BPE + byte fallback
│   ├── vocab.go                    从 GGUF 读 tokens/scores/merges
│   ├── chat.go                     TinyLlama chat template（硬编码 + 直通）
│   └── *_test.go
├── sample/
│   ├── sampler.go                  greedy / temp / top-k / top-p / rep-penalty
│   └── *_test.go
├── internal/testutil/
│   ├── golden.go                   golden 文件读写（.bin 小端 float32）
│   └── fixtures.go                 跳过条件（模型缺失时 t.Skip，非失败）
├── testdata/
│   ├── goldens/                    ≤64KB 的小型 golden 二进制
│   ├── refs/                       参考实现导出的 logits 与 token id
│   └── README.md                   每个 fixture 的来源与生成命令
└── docs/
    ├── design.md                   本文件
    ├── interfaces.md               §4 的正式版
    ├── verification.md             §6 的正式版
    └── milestones.md               §7 的正式版
```

**约定**：`internal/` 之外不导出未在 `docs/interfaces.md` 声明的 API。每次 PR 必须 `go vet` + `gofmt -l` 干净。

---

## 4. 模块接口（Go 签名草案）

### 4.1 gguf

```go
package gguf

type File struct { /* file handle, alignment, metadata, tensors */ }

func Open(path string) (*File, error)          // 解析 header + KV + tensor info；不加载权重
func (f *File) Alignment() uint64
func (f *File) TensorCount() int
func (f *File) Tensor(name string) (TensorInfo, bool)

// Metadata 访问器：缺失 key 返回 (zero, false)，不 panic。
func (f *File) String(key string) (string, bool)
func (f *File) Uint(key string) (uint64, bool)
func (f *File) Float(key string) (float64, bool)
func (f *File) UintArray(key string) ([]uint64, bool)
func (f *File) Strings(key string) ([]string, bool)   // tokens / merges

type TensorInfo struct {
    Name  string
    Dims  []uint64      // GGML 序：Dims[0] 为最内层
    Type  Type
    Offset uint64       // 相对 data section
}

// ReadRaw 返回该张量原始字节（含量化块），不反量化。
func (f *File) ReadRaw(t TensorInfo) ([]byte, error)

// Close 释放 mmap / 文件句柄。
func (f *File) Close() error
```

**契约**：`Open` 必须对 v1/v2 GGUF 返回 `ErrUnsupportedVersion`；对 big-endian 返回 `ErrBigEndian`。所有读取错误必须带 key/张量名上下文。

### 4.2 quant

```go
package quant

const (
    QK4_0 = 32
    QK8_0 = 32
    SizeOfQ4_0 = 18   // bytes per block
    SizeOfQ8_0 = 34
)

// Dequantize 将 raw 按 typ 反量化为行主序 f32，n 为元素总数。
// 返回切片长度必须等于 n；n 不必是 32 的倍数（尾部由调用方保证对齐）。
func Dequantize(typ Type, raw []byte, n int) ([]float32, error)

// DequantizeInto 复用调用方缓冲区，热路径使用。
func DequantizeInto(typ Type, raw []byte, dst []float32) error

// DequantRowQ4_0 单行反量化，暴露给单元测试做逐元素比对。
func DequantRowQ4_0(raw []byte, dst []float32) error
```

### 4.3 tensor

```go
package tensor

type Tensor struct {
    Shape []int       // 行主序，Shape[0] 为最外层
    Data  []float32
}

// 全部运算接受显式 dst，避免热路径分配。
func MatVec(dst, m, x []float32, rows, cols int)            // dst = m·x，m 为 rows×cols 行主序
func MatMul(dst, a, b []float32, m, k, n int)               // dst[m×n] = a[m×k]·b[k×n]
func RMSNorm(dst, x, w []float32, eps float32)
func SoftmaxInPlace(x []float32)
func SiLU(x float32) float32
func AddInPlace(dst, src []float32)
```

**契约**：所有函数在长度不匹配时 `panic`（内部 API，不做错误返回）；公开边界在调用前校验一次。

### 4.4 model

```go
package model

type HParams struct {
    NLayer, NHidden, NFF, NVocab, NHead, NHeadKV int
    HeadDim        int
    RMSNormEps     float32
    RopeTheta      float32
    TieWordEmb     bool
}

type Model struct { /* weights, hparams, tokenizer refs */ }

func Load(f *gguf.File) (*Model, error)         // 按 llama.* metadata 装配 + 形状校验
func (m *Model) HParams() HParams

// NewSession 创建带有独立 KV cache 的会话。
func (m *Model) NewSession(ctxLen int) (*Session, error)

type Session struct { /* pos, kv cache, logits buffer */ }

// Eval 对 tokens 逐个前向；返回最后位置的 logits（长度 NVocab）。
// 支持增量调用：pos 自动推进，因此 prefill 与 decode 共用同一路径。
func (s *Session) Eval(tokens []int) ([]float32, error)

// EvalAll 返回每个 token 位置的 logits 切片视图，仅调试/验证使用。
func (s *Session) EvalAll(tokens []int) ([][]float32, error)

func (s *Session) Reset()
```

**契约**：`Eval` 在 `pos+len(tokens) > ctxLen` 时返回 `ErrContextOverflow`。`Load` 必须校验每个张量的形状与 `HParams` 推导值一致，任何不一致立即报错并列出期望/实际。

### 4.5 tokenizer / sample

```go
package tokenizer
type Tokenizer struct{ /* ... */ }
func New(vocab []string, scores []float32, merges []string) (*Tokenizer, error)
func (t *Tokenizer) Encode(text string, addBOS bool) []int
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string
func TinyLlamaChat(messages []Message, genPrompt bool) string

package sample
type Params struct { Temp float32; TopK int; TopP float32; RepeatPenalty float32; Seed uint64 }
func Greedy(logits []float32) int
func Sample(logits []float32, p Params, recent []int) int
```

---

## 5. 关键算法细节（易错点清单）

1. **RoPE**：Llama 使用 **相邻对** 旋转（`ggml_rope` 的 normal 模式），不是 GPT-NeoX 的"前后半"配对。`freq_i = 1/theta^(2i/head_dim)`，位置从 `pos` 开始。验证方法：单头、单层、已知输入的手算 golden。
2. **GQA**：`attn_k`/`attn_v` 的 out 维是 `NHeadKV*HeadDim`；head `h` 使用 kv head `h / (NHead/NHeadKV)`。整数除法必须在 `NHead % NHeadKV == 0` 时成立（TinyLlama 8:1 满足，SmolLM2 9:3=3:1 满足）。
3. **pre-norm 残差顺序**：`x = x + attn(rmsnorm(x))`，再 `x = x + ffn(rmsnorm(x))`；最后 `rmsnorm(x)` 后投影。
4. **SwiGLU**：`silu(gate) * up`，即 `down(silu(W1·x) ⊙ W3·x)`。
5. **softmax**：先减 max 再 exp；因果掩码在 prefill 时对未来位置填 `-Inf`。
6. **q4_0 nibble 顺序**：见 `research/facts-verified.md` §4——低半字节对应下标 `j`，高半字节对应 `j+16`。
7. **BPE legacy 规则**：SentencePiece 的"首个 token 前的隐式空格"与 byte fallback（`<0xNN>`）行为是 tokenizer 正确性的最大陷阱。验证方法：对固定文本与 HF `tokenizer.encode` 逐 id 比对。

---

## 6. 正确性验证策略（采用"阶梯式对齐"）

原则：**每一步都与一个可独立复核的参照物对齐，绝不跨步**。

### 6.1 参照实现的优先级

| 层级 | 参照物 | 产出 | 用途 | 依赖 |
|---|---|---|---|---|
| R0 | 手算 / 独立小脚本（Python + numpy，公式直接抄 ggml-quants.c） | golden 二进制 | 量化反量化、RoPE、softmax | 无 |
| R1 | HF `transformers` + `torch` CPU f32 | 每层隐藏状态 dump、最终 logits、token id | 全链路逐层对齐 | PyTorch CPU 版 |
| R2 | llama.cpp `llama-cli` / `llama-perplexity` | 生成 token id 序列、plausible logits | 端到端生成一致性、量化数值健康度 | 需本地构建 llama.cpp |
| R3 | 自洽性（无外部参照） | 早停/对齐 | 回归保护 | 无 |

> **[待证]** R2 的 logits 导出能力：需在 M0 读 `common/arg.cpp` 确认是否存在 `--logits-file`/`--logits-all` 类参数；若不存在，R2 降级为"仅比对生成 token 序列"，logits 数值以 R1 为准。

### 6.2 容差标准（在 f32 下）

| 检查项 | 指标 | 阈值 | 说明 |
|---|---|---|---|
| 反量化单块 | max abs diff | ≤ 1e-6（Q8_0）/ ≤ 1e-6（Q4_0，量化值本身有跳变，比对的是 `q*d`） | R0 |
| RMSNorm | max rel diff | ≤ 1e-5 | R1 |
| 单层 attn 输出 | cosine sim | ≥ 0.99999 | R1 |
| 22 层后 logits | max abs diff | ≤ 1e-3 | R1，f32 累积误差自然上限 |
| logits argmax | 完全一致 | 100% | R1，对 20 条固定 prompt |
| 贪心生成 32 token | id 序列完全一致 | 100% | R2 |
| 采样生成 | 固定 seed 下 id 序列一致 | 100% | 只测自身确定性，不比 R2 |

**任何阈值都必须由实际测得的误差分布校准一次后，写入 `docs/verification.md` 并锁定；不得为了通过而放宽。**

### 6.3 分阶段门禁（每个阶段不通过则不进入下一阶段）

```
G1 反量化正确      ← R0，覆盖率 100% 的 (type × 边界长度)
G2 单算子正确      ← R0 + R1，matmul/rmsnorm/rope/softmax/silu 各自独立测试
G3 单层前向正确    ← R1，固定输入下 attn 子块与 ffn 子块分别对齐
G4 全模型 prefill  ← R1，logits max abs diff ≤ 1e-3
G5 增量 decode     ← R1，逐 token 与 prefill 版结果一致（自洽）
G6 端到端生成      ← R2，贪心 32 token id 完全一致
G7 tokenizer       ← R1（HF tokenizer）逐 id 对齐，覆盖中英混排/emoji/空白边界
G8 性能基线        ← 记录 tokens/s，不做优化承诺
```

---

## 7. 里程碑计划与交付证据

> 每个里程碑的"完成"定义 = 列出的**全部**证据存在且可复现。缺一项即不算完成。

> **执行进度不记录在本文档内。**
> 里程碑状态、各门禁证据与 G1–G8 进度见 [`docs/status.md`](status.md)。
> 本文档只描述计划本身，不随执行进展修改。

### M0 · 环境与参照基线（预计 0.5 天）

**交付**
- `go.mod`、`Makefile`、CI 脚本骨架
- Go ≥1.22 安装脚本（Arm64）
- Python 侧参考环境：`transformers` + `torch` CPU + `numpy`，固定版本于 `requirements-ref.txt`
- 下载 TinyLlama Q4_0 GGUF 与 SmolLM2-135M Q8_0 GGUF；记录 sha256
- R1 参考导出脚本 `tools/ref_export.py`：输入 prompt，输出 `logits.npy` + `hidden_<layer>.npy` + `tokens.json`

**证据**
1. `go version`、`python -c "import torch; print(torch.__version__)"` 的实际输出
2. `sha256sum` 两个 GGUF 文件
3. `tools/ref_export.py --prompt "Hello"` 成功产出 4 类文件，且 `logits.npy` 形状为 `[seq, 32000]`
4. 关闭 §1 中的 4 个 **[待证]** 项，结果追加进 `research/facts-verified.md`

**风险**：3 GiB 内存下同时跑 torch 参考与 Go 可能 OOM → 参考导出在独立进程中顺序执行，且用 `torch.inference_mode()` + `float32`。

---

### M1 · GGUF 解析器（预计 1.5 天）

**交付**
- `gguf/` 全部实现 + `cmd/gguf-inspect`
- 支持 v3 完整 KV 类型表、tensor info、对齐、mmap

**证据**
1. `go test ./gguf/ -v` 全绿，测试覆盖：magic 错误、版本不支持、截断文件、超大 dims、未知 KV 类型（必须报错而非 panic）
2. `go-llm inspect -m tinyllama.gguf | head -40` 输出与参考 `python -m gguf.scripts.gguf_dump` 的 metadata 段**逐行等价**（数量、key、类型、值），差异表为空
3. `sha256` 校验：对某一张量 `ReadRaw` 后与独立 Python 脚本切片得到的字节完全一致
4. 内存峰值：`inspect` 全程 RSS < 100 MB（证明没有误加载权重）

**门禁**：证据 2 是硬门禁——metadata 读错则后续全错。

---

### M2 · 反量化与张量算子（预计 2 天）

**交付**
- `quant/`（F32/F16/Q8_0/Q4_0）+ `tensor/`（matvec/matmul/rmsnorm/softmax/silu/add）
- R0 golden 生成脚本 `tools/gen_r0_goldens.py`（numpy 直接按 ggml 公式实现，与 Go 实现**无代码共享**）

**证据**
1. `go test ./quant/ ./tensor/ -race -cover`，覆盖率 ≥ 90%
2. golden 比对日志：对 Q4_0/Q8_0 各 ≥ 20 组（长度 32、64、32000、含尾部）max abs diff ≤ 1e-6，输出中打印实测最大值
3. `go test -bench . ./tensor/` 输出，记录 GEMV 在 2048×2048 上的 GFLOPS（仅记录，不设目标）
4. 交叉验证：Go 反量化结果与 llama.cpp `ggml` 小工具（或 R1 中 `dequantize` 结果）比对通过

**常见失败模式（需在测试中显式覆盖）**：q4_0 nibble 错位、q4_0 未减 8、f16 标度解析错、tensor 维度顺序颠倒（GGML 的 dims[0] 是最内层）。

---

### M3 · Tokenizer 与 chat template（预计 1.5 天）

**交付**
- `tokenizer/bpe.go`（含 byte fallback 与 legacy prefix-space 规则）
- `tokenizer/chat.go`（TinyLlama 模板）

**证据**
1. 对 30 条固定文本（中英混排、emoji、连续空格、行首空格、控制字符、超长行）逐条比对 HF `AutoTokenizer.encode`：id 数组**完全相同**，输出为逐条 PASS 表格
2. Round-trip 测试：`Decode(Encode(s)) == s` 对上述 30 条全部成立
3. Chat 模板渲染输出与 HF `apply_chat_template(add_generation_prompt=True)` 字符串**完全相同**（diff 为空）

**门禁**：tokenizer 不一致会污染所有后续 logits 对比，必须硬门禁。

---

### M4 · 单层前向（预计 2.5 天）

**交付**
- `model/loader.go` 权重装配 + `model/llama.go` 单层：RMSNorm → QKV proj → RoPE → GQA attn → out proj → 残差 → RMSNorm → SwiGLU → 残差
- 调试钩子：`--dump-layer N` 导出隐藏状态

**证据**
1. `go-llm logits -m tinyllama.gguf --tokens 1,15043 --dump-layer 0` 导出 layer0 输出
2. 与 R1 `hidden_0.npy` 比对：cosine sim ≥ 0.9999，max abs diff 打印实测值
3. 单独算子测试：固定 8-token 输入下，attention 输出（RoPE 后）与 R1 中间量对齐
4. 形状校验测试：故意篡改 HParams（如 n_head_kv=5）时 `Load` 报错且错误信息含期望/实际

---

### M5 · 全模型与 KV cache（预计 2 天）

**交付**
- 22 层堆叠、final norm、output projection（tied 支持）
- `kvcache.go` + 增量 decode 路径

**证据**
1. `go test ./model/ -v` 中的 G4：对 5 条 prompt，全序列 logits max abs diff ≤ 1e-3（逐 prompt 打印实测值）
2. **G5 自洽性**：同一序列一次性 prefill 得到的最后位置 logits，与逐 token 增量 Eval 得到的 logits，max abs diff ≤ 1e-5
3. G6 前置：贪心生成 32 token 的 id 序列写入 golden
4. `EvalAll` 返回的每位置 logits 与 R1 逐位置比对通过
5. SmolLM2-135M（tied embedding + theta=100000 + 9 heads）同样通过 G4，证明泛化

---

### M6 · 采样与 CLI（预计 1 天）

**交付**
- `sample/` 全部采样器 + `cmd/go-llm` 交互式与单次生成

**证据**
1. 采样器单元测试：top-k/top-p 边界（k>vocab、p=0、p=1）、repetition penalty 符号
2. 固定 seed 下生成可复现：连续两次运行输出**字节级相同**
3. **G6 硬门禁**：`llama-cli` 与 `go-llm run` 在相同 prompt、相同 `--temp 0` 下生成 32 token，id 序列完全一致；附两侧原始输出
4. 端到端 demo：生成一段可读中文/英文回答，附完整命令与输出

---

### M7 · 文档、基准、发布（预计 1 天）

**交付**
- `docs/verification.md` 定稿（含所有实测误差数值）
- 基准报告：TinyLlama Q4_0 在 4 核 arm64 上的 prompt-eval 与 token-eval t/s
- `v0.1.0` tag + 复现脚本 `make verify`（一键跑全部门禁）

**证据**
1. `make verify` 从干净 checkout 一次跑通，输出全部 PASS 表格
2. 基准原始输出（含 CPU 型号、Go 版本、RSS 峰值）
3. 误差表：每一门禁的阈值 vs 实测值
4. README 中的快速开始命令经实际执行验证

---

## 8. 任务分配

| 里程碑 | 主责 | 复核 | 说明 |
|---|---|---|---|
| M0 环境与参照基线 | Coordinator | Reviewer | Coordinator 亲自搭，避免早期歧义 |
| M1 GGUF 解析 | **Writer** | Reviewer + Researcher | Researcher 提供 GGUF 规范细节与边界用例清单 |
| M2 反量化与算子 | **Writer** | Reviewer（R0 golden 独立生成） | Reviewer **不得**复用 Writer 的实现生成 golden |
| M3 Tokenizer | **Writer** | Researcher（提供 BPE 规则说明）+ Reviewer | 本项风险最高，需双人核对 |
| M4 单层前向 | **Writer** | Reviewer（对照 R1 hidden dump） | 逐层禁门 |
| M5 全模型 + KV | **Writer** | Reviewer（G4/G5 门禁） | Coordinator 负责接口冲突裁决 |
| M6 采样 + CLI | **Writer** | Reviewer（G6 对 llama.cpp） | |
| M7 文档与基准 | **Writer**（文档）| Reviewer（复跑 `make verify`） | Coordinator 整合发布说明 |

**并行策略**：M1 与 M2 可并行（无共享状态）；M3 与 M2 可并行；M4 依赖 M1+M2；M5 依赖 M4；M6 依赖 M5+M3。

**Git 协调**：每个里程碑一个分支 `feat/mN-<slug>` + worktree；合并前必须通过对应门禁测试；Coordinator 不做代码覆盖，只做冲突裁决与接口变更审批（变更须先改 `docs/interfaces.md`）。

---

## 9. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 3 GiB 宿主上同时跑 torch 参考会 OOM | M0 阻塞 | 参考导出用独立进程 + `float32` + 小 seq；必要时只导出子集层 |
| llama.cpp 无 logits dump 参数 [待证] | R2 降级 | 以 R1 为 logits 权威，R2 仅比 token 序列 |
| Tokenizer legacy BPE 细节偏差 | M3 返工 | M3 前置硬门禁；Researcher 先产出规则说明 |
| Go 无 SIMD，性能远低于 llama.cpp | 性能不达预期 | MVP 明确**不设性能目标**，只记录基线；优化留 v0.2 |
| 调研子代理未交付（已发生） | 信息缺口 | 已由 Coordinator 直接取证填补（见 `research/facts-verified.md`），M0 关闭剩余 4 项 |
| Q4_0 对 1.1B 模型质量下降导致生成异常 | 演示效果差 | 同时准备 Q8_0 版本做对照；数值门禁用 Q8_0/Q4_0 双跑 |

---

## 10. 需要用户决策的事项

1. **模块路径**：`go.mod` 的 module 名（如 `github.com/<你的用户名>/go-llm`）。
2. **MVP 模型确认**：TinyLlama-1.1B-Chat-v1.0 (Q4_0) 作为主模型是否可接受？CI 用 SmolLM2-135M。
3. **是否接受 v1 不做性能优化**（只记录基线，不设吞吐目标）。
4. **是否允许在 M0 下载约 0.75 GB 模型文件**（TinyLlama Q4_0 + SmolLM2 Q8_0）到本机。
5. **是否允许安装 Go 工具链与 Python 参考环境**（torch CPU 约 1-2 GB 磁盘）。

批准后即从 M0 开始，每完成一个里程碑提交一次证据包。
