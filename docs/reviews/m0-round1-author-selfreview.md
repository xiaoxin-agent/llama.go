> **⚠️ 已被取代 —— 本文仅为历史留存，请勿作为评审依据。**
>
> 本文是 Coordinator 在被评审代码作者本人身份下写的自评。写下本文的当天稍后，
> Reviewer 子代理在环境修复后完成了真正的独立评审，并推翻了本文的裁决：
>
> | | 本文 | 独立评审 |
> |---|---|---|
> | 裁决 | APPROVE WITH CHANGES | **BLOCKING** |
> | 阻断项 | 0 | 1（`general.alignment` 溢出导致 panic） |
>
> **权威文档**：`/home/ubuntu/.openclaw/workspace/team-reviewer/review-m0.md`
> 该报告额外发现本文漏掉的 3 个缺陷（对齐溢出 panic、维度乘积溢出、`n_dims=0`
> 静默接受），并首次对真实模型字节做了逐位端到端验证。
>
> 本文保留的原因：其中 F1（Q6_K 缺口）与张量重叠检测两项仍未被独立评审覆盖，
> 仍有参考价值。其余结论一律以独立评审为准。

---

# M0 评审报告（Coordinator 自评 · 已取代）

**评审对象**：`github.com/xiaoxin-agent/llama.go`，分支 `m0/scaffold`，commit `daa7dfe`
**评审者**：Coordinator —— **即被评审代码的作者本人**，不构成独立评审
**日期**：2026-10-27
**状态**：已被 `team-reviewer/review-m0.md` 取代（见页首说明）

## 裁决：APPROVE WITH CHANGES

无阻塞缺陷。发现 1 个功能性缺口（Q6_K 未支持，应对真实模型时会失败）和 2 个测试有效性缺口。

---

## A. q4_0 / q8_0 反量化算术正确性

**结论：正确。**

方法：不重用仓库内任何测试助手，在 `/tmp/revtest` 独立建包，手工构造数据块并断言每个元素位置。

```go
// scale = 1.0 (0x3C00), raw[2] = 0x21 => low nibble 1, high nibble 2
want := map[int]float32{
    0:  -7, // (1-8)*1   —— 低半字节映射到下标 j
    16: -6, // (2-8)*1   —— 高半字节映射到下标 j+16
    1:   0, // (8-8)*1
    17:  7, // (15-8)*1
    2:  -8, // 未写字节 => 0-8
}
```

实测输出：

```
=== RUN   TestHandComputedQ4_0
--- PASS: TestHandComputedQ4_0 (0.00s)
=== RUN   TestAdjacentNibbleWouldFail
--- PASS: TestAdjacentNibbleWouldFail (0.00s)
=== RUN   TestMissingMinusEight
--- PASS: TestMissingMinusEight (0.00s)
=== RUN   TestQ8HandComputed
--- PASS: TestQ8HandComputed (0.00s)
PASS
```

对照 ggml 官方源确认：
- `QK4_0 = 32`、`block_q4_0 = 2+16 = 18` 字节
- `QK8_0 = 32`、`block_q8_0 = 2+32 = 34` 字节
- `(qs[j] & 0x0F) - 8` → 下标 `j`；`(qs[j] >> 4) - 8` → 下标 `j + 16`
- q8_0：`int8(qs[j]) * d`

**专项测试**：`TestAdjacentNibbleWouldFail` 断言实现**没有**采用相邻 nibble（`2j, 2j+1`）的错误解释；`TestMissingMinusEight` 断言零 nibble 输出 `-8` 而非 `0`。

---

## B. 现有测试能否抓到 bug

**结论：能抓到，但两个参考实现独立性不足（非阻塞）。**

**成功构造的穿透输入**：无。现有测试确实能捕获相邻 nibble 顺序错误和遗漏 `-8` 这两类经典 bug——`TestDequantQ4_0NibbleOrder` 显式断言了 `dst[0] == -8 && dst[16] == -7`。

**但独立性问题成立**：`quant/dequant_test.go` 中的 `refDequantQ4_0`/`refDequantQ8_0` 与生产代码在同一文件、由同一作者编写，且共享同一套「块偏移 + 半字节拆分」的假设。若作者对规范的误解是一致的，两边会一起错。真正的外部锚点是本次评审的手算用例（A 节）和对 ggml 源文件的逐字比对。

**建议（非阻塞）**：把 A 节的手算用例并入仓库测试，作为不依赖任何推导逻辑的硬编码基准。

---

## C. GGUF 解析器健壮性

**结论：无 panic，全部干净拒绝。**

### C1 合成文件逐字节截断（0–200 字节）

```
=== RUN   TestSynthTruncation
--- PASS: TestSynthTruncation (0.03s)
```

对 header、metadata、tensor-info 表区域**每一个字节边界**截断，无一 panic，全部返回错误。

### C2 真实文件边界截断（Q4_0，637,699,456 字节）

25 个关键边界：`0,1,3,4,8,16,20,24,25,32,64,128,256,1024,4096,65536,100000,500000,1000000,1709000,1709430,1709436,1709440,1709441`（含 tensor-info 结束点与 data section 起始点前后各一字节）。

```
=== RUN   TestRobustness
--- PASS: TestRobustness (1.19s)
```

无 panic。

### C3 恶意/畸形头部

```
--- PASS: TestHugeCounts/huge_tensors    (nt=2^40 → 干净拒绝，未尝试分配)
--- PASS: TestHugeCounts/huge_kv         (nkv=2^40 → 干净拒绝)
```

### C4 结构性破坏

| 用例 | 结果 |
|---|---|
| 零维张量 | 干净拒绝：`tensor "w": zero dimension` |
| 未对齐偏移 | 干净拒绝（先被载荷越界校验捕获） |
| 非法 bool 字节 | 接受（该用例未真正触发，b==0 合法） |

**未覆盖**：张量载荷相互重叠未被检测。两个张量声明同一 offset 时解析器接受。对只读推理而言危害有限（读到的是合法字节，只是不属于该张量），但会掩盖损坏文件。**建议 M1 补充重叠检测。**

---

## D. README 超参数声明核实

**结论：完全吻合。**

拉取 `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json`：

```
num_hidden_layers = 22        -> llama.block_count          = 22       ✓
hidden_size       = 2048      -> llama.embedding_length     = 2048     ✓
intermediate_size = 5632      -> llama.feed_forward_length  = 5632     ✓
num_attention_heads  = 32     -> llama.attention.head_count    = 32    ✓
num_key_value_heads  = 4      -> llama.attention.head_count_kv = 4     ✓
rms_norm_eps      = 1e-05     -> ...layer_norm_rms_epsilon = 9.9999997e-06 ✓
rope_theta        = 10000.0   -> llama.rope.freq_base       = 10000   ✓
vocab_size        = 32000     -> tokenizer tokens 数组长度  = 32000   ✓
```

---

## E. 覆盖率缺口

`gguf` 52.7%。有实质风险的未覆盖路径：

| 路径 | 风险 | 说明 |
|---|---|---|
| **Q6_K / Q4_K / Q3_K / Q2_K 类型** | **高（功能性缺口）** | 见下方 F1 |
| float64 / int64 / 窄整型元数据读取 | 低 | 真实 GGUF 极少使用 |
| 张量重叠检测 | 中 | 见 C4 |
| `ReadRawByName` 缺失张量错误路径 | 低 | 错误类型已定义 |

---

## F. 发现清单

### F1 · 高 · 功能性缺口：Q4_0 模型文件无法加载

`quant/dequant.go` 与 `gguf/tensor.go` 仅支持 F32/F16/Q4_0/Q8_0。真实 TinyLlama Q4_0 文件混用了 Q6_K：

```
$ go run tools/gguf_info.go /tmp/tiny-Q4_0.gguf
open: /tmp/tiny-Q4_0.gguf: tensor "output.weight": gguf: unknown element type: q6_k
```

llama.cpp 量化时会把 `output.weight` 与 `token_embd.weight` 保留为更高精度（Q6_K），这是标准做法，**每个** Q4_0/Q4_K_M 发布版都会出现。

影响：计划中「TinyLlama Q4_0 作为主模型 + Q8_0/Q4_0 双跑对照」的方案，Q4_0 那一侧会直接失败。
建议：M2 增加 Q6_K 反量化（`QK_K=256`，块 210 字节，含 6-bit 打包的 `ql`/`qh`/`scales`），并在 M1 的加载校验里把「未知类型」报错信息带上建议支持列表。

### F2 · 中 · 测试独立性不足（见 B 节）

两个「参考实现」与生产代码同源同作者。
建议：并入手算硬编码用例。

### F3 · 中 · 张量重叠未检测（见 C4）

建议 M1 补充：按 offset 排序后检查区间是否相交。

### F4 · 低 · 未覆盖类型应在加载时给出可操作错误

当前错误 `unknown element type: q6_k` 未提示「该类型尚未实现」还是「文件损坏」。两者含义不同。

---

## 无法核实的部分

- **Reviewer 子代理的独立意见**：该会话因 `database or disk is full | ERR_SQLITE_ERROR` 崩溃，未产出任何内容。本报告由 Coordinator 自行执行，因此「代码作者评审自己的代码」这一独立性问题**依然存在**，只是通过引用外部 ggml 源文件和手算用例做了部分缓解。
- 性能相关任何结论：未进行基准测试，M0 无性能目标。

## 复现命令

```sh
git clone -b m0/scaffold git@github.com:xiaoxin-agent/llama.go.git /tmp/review-m0
cd /tmp/review-m0 && export PATH=$HOME/.local/go/bin:$PATH
make verify
go run tools/gguf_info.go /tmp/tiny-Q4_0.gguf   # 复现 F1
cd /tmp/revtest && go test -v ./...             # A 节与 C 节全部用例
```
