# 独立评审报告 M0：GGUF 解析器（gguf / quant 包）

## BLOCKING

阻断 1 项（F1：对齐值溢出导致 panic），非阻断 8 项（F2-F9）。

评审对象：/tmp/review-m0 @ daa7dfe（分支 m0/scaffold）
评审环境：Go 1.23.4，/tmp/tiny-Q4_0.gguf（637,699,456 B）

检查方法：所有构建物、测试模块与篡改用例位于 /tmp/reviewer-m0/（独立 module，`replace` 指向 /tmp/review-m0）；未修改被评审仓库任何文件。

---

## A. 反量化算术（q4_0 / q8_0）

### A.1 与 ggml 上游源文件比对

抓取命令：
```
curl -sS -o ggml-quants.c https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
curl -sS -o ggml-common.h https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h
```

上游事实（ggml-common.h）：
- 第 194 行 `#define QK4_0 32`；第 196–198 行 `block_q4_0 { ggml_half d; uint8_t qs[QK4_0/2]; }`，第 199 行 static_assert 大小 = 2 + 16 = 18。
- 第 251 行 `#define QK8_0 32`；第 252–255 行 `block_q8_0 { ggml_half d; int8_t qs[QK8_0]; }`，第 256 行 static_assert 大小 = 2 + 32 = 34。
- `ggml_half` 即 fp16，结构体第一个成员，因此**前 2 字节是 fp16 scale**，无填充（两个类型都是 2 字节对齐的自然排布）。

上游事实（ggml-quants.c）：
- `dequantize_row_q4_0`（第 459–476 行）：`x0 = (qs[j]&0x0F)-8; x1 = (qs[j]>>4)-8; y[j]=x0*d; y[j+qk/2]=x1*d;`，j ∈ [0,16)。
- `dequantize_row_q8_0`（第 553–567 行）：`y[j] = qs[j]*d`（qs 为 int8），j ∈ [0,32)。

仓库实现对照：
- quant/dequant.go:16-22 `QK4_0=32, QK8_0=32, SizeOfQ4_0=18, SizeOfQ8_0=34` —— 一致。
- quant/dequant.go:120-135 `dequantQ4_0`：`d` 取自 `blk[0:2]`（第 130 行），低半字节 -> `base+j`，高半字节 -> `base+j+16`（第 133–134 行）—— 与上游一致。
- quant/dequant.go:139-153 `dequantQ8_0`：`d` 取自 `blk[0:2]`，`int8(blk[2+j])*d` —— 与上游一致。

**结论：未发现与上游的任何差异（无阻断项）。**

### A.2 自建手造 block 测试（不重用仓库测试助手）

独立模块 /tmp/reviewer-m0（`replace github.com/xiaoxin-agent/llama.go => /tmp/review-m0`），
测试文件 /tmp/reviewer-m0/src/nibble_test.go。要点：每个元素位置赋唯一 nibble，逐位置断言。

命令与输出：
```
$ export PATH=$HOME/.local/go/bin:$PATH; cd /tmp/reviewer-m0 && go test -count=1 -v ./...
=== RUN   TestQ4HandBlock
dst[0]=-7 dst[16]=1
--- PASS: TestQ4HandBlock (0.00s)
--- PASS: TestQ4ScaleIsFirstTwoBytes (0.00s)
--- PASS: TestQ8HandBlock (0.00s)
--- PASS: TestQ4MultiBlockStride (0.00s)
```
覆盖了：位置映射（byte j -> 元素 j 与 j+16）、scale 为前两字节且为 fp16（0x3E00=1.5、0xC000=-2.0）、
多 block stride 为 18、q8_0 的 int8 符号与 stride 34。

### A.3 真实模型数据端到端比对（最有力的一项）

由于仓库的 `gguf.Open` 在真实文件上会失败（见 D 节），**无法**通过仓库 reader 取到 q4_0 张量数据。
因此我另写了一个不依赖 Go 代码的 Python GGUF 解析器（/tmp/reviewer-m0/py/gguf.py），
定位 `token_embd.weight` 的 offset（53760000）+ data_start（1709440），取前 3 个 block（54 字节），
分别用：(1) 仓库的 `quant.DequantizeInto`（直接喂字节，绕过 reader）；(2) Python 按规范手写的反量化。

```
token_embd.weight dims [2048, 32000] type q4_0 offset 53760000
go    : [5.72e-06, 3.81e-06, 3.81e-06, 9.54e-06, 3.81e-06, -5.72e-06]
python: [5.72e-06, 3.81e-06, 3.81e-06, 9.54e-06, 3.81e-06, -5.72e-06]
max abs diff go vs python = 0.0
max diff go vs adjacent-nibble variant = 2.6702880859375e-05
```
真实数据上 **bit-exact 一致**；且“相邻 nibble”错误变体在真实数据上差异 2.7e-5，说明该测试有区分力。

**A 结论：通过。** 未发现算术、块大小、nibble 顺序、scale 位置的缺陷。

---

## B. 现有测试是否真能抓 bug

### B.1 "独立参考实现" 的独立性判断

`quant/dequant_test.go:15-63` 的 `refDequantQ4_0` / `refDequantQ8_0` 声称"independent"，
但它们 (a) 与生产代码在同一文件、由同一作者写，(b) 嵌入的是**同一个算法理解**。
独立性的真正来源是"重新实现中的错误是否与生产代码的错误相互独立"——这里显然不独立。

不过需要注意：`TestFloat16AllBitPatterns`（dequant_test.go:131-152）用穷举 65536 个 fp16 位型对比
`refHalfToFloat`，这是**测试向量完整**的证明（穷举空间），对 fp16 转换这块的独立性要求较低，
这部分可信度较高。

### B.2 我构造的"貌似合理但错误"的实现能否通过现有测试

逐个评估（结论基于逐行阅读测试里的输入构造方式）：

1. **相邻 nibble 顺序**（byte j -> 元素 2j, 2j+1）：**会被抓到**。
   理由：`TestDequantQ4_0AgainstReference` 的参考实现也会用同一顺序，所以它本身抓不到；
   但 `TestDequantQ4_0NibbleOrder`（dequant_test.go:198-218）显式断言 dst[0]=-8, dst[16]=-7，
   换顺序后 dst[1], dst[17] 会被写成 -7，断言失败。**因此这一经典 bug 有保护。**
2. **漏掉 -8 偏移**（直接 `q*d`）：会被 `TestDequantQ4_0NibbleOrder` 抓到（期望 -8、-7）。
   也会被参考实现抓到，因为参考实现里写了 -8。**有保护。**
3. **错误的 block stride**（例如把 18 当作 16 或 20；nb 用 len/20）：
   - `TestDequantQ4_0AgainstReference` 用 nb=1..4，且 `got` 的期望长度是 `nb*32`，错误 stride 会产生
     长度/内容不匹配，`maxAbsDiff` 中 `b[i]` 索引一致（两 slice 长度相同，因为生产代码按 n 分配），
     但缩放常量读错位置会导致数值差 -> 会被抓到。
   - `TestDequantErrors` 有 "q4_0 short payload"(17 字节) 和 "args mismatch"(36 字节 vs 32 元素)，
     会把 stride 混用的错误暴露。
   **基本有保护**，但对"仅第 0 个 block 正确、后续 block 用错 stride"的情形，由于 `got` 与 `want`
   长度一致且参考实现同样可能同错——若作者同时写错生产与参考实现中的 stride，则**测试通过**。
   我在独立测试 `TestQ4MultiBlockStride` 中明确用不同 scale 检测到了边界，仓库自身无等价测试。
4. **nb 的 off-by-one**（`nb = n/QK4_0 + 1` 或 `len(raw)/18 + 1`）：会因为越界 panic 或长度不符而失败；
   `TestDequantErrors` 的 "args mismatch" 也覆盖。**有保护。**
5. **高/低 nibble 互换但位置正确**（lo->j+16, hi->j）：会被 nibble 顺序测试抓到。

**我是否找到了能漏过现有测试的错误实现？**
- 在现有测试覆盖的维度（第 0 个 block 的 nibble、符号、偏移、基本 stride）上：**没有找到**。
  这比预期好，主要因为 dequant_test.go:198 的显式顺序断言。
- 但确实存在**未被覆盖的缝隙**（不构成"通过现有测试的错误实现"的直接反例，但值得指出）：
  (i) 参考与生产代码同源，若二者共享同一误解（如 block 起始位置计算），当前测试无法发现；
     我用真实模型数据（A.3）作为外部锚点来补这个缺口。
  (ii) 没有任何测试拿**真实 gguf 文件**走完整路径（reader->ReadRaw->dequant）。
      `gguf.Open` 在真实文件上直接失败（见 D），说明集成路径从未被真实数据验证过。
  (iii) scale 为负数或非 1.0/0.5 的 block 在 nibble 顺序测试里没有，但 A.2 的 1.5/-2.0 覆盖了。

**B 结论：未找到能通过现有测试的"貌似合理但错误"实现（就 nibble/偏移/stride/off-by-one 四类而言）。**
但"独立参考实现"的独立性声明在方法论上是弱的；真正的独立证据来自 A.3 的真实数据比对，
这一点应在文档里修正措辞（把 dequant_test.go:15-18 的 "no shared code" 改成
"same-author, same-file reference; independence only in the sense of not calling production helpers"）。

---

## C. 解析器健壮性

方法：所有用例都基于 `/tmp/tiny-Q4_0.gguf` 的**副本**或自行构造的字节流，从不写原文件。
构造器：/tmp/reviewer-m0/py/build.py（独立 GGUF 写出器）；
测试驱动：/tmp/reviewer-m0/harness/h（recover 捕获 panic，打印 `OK/ERR/PANIC` 并尝试读取每个张量）。

### C.1 阻断项：`general.alignment` 巨大值导致 panic

用例：`general.alignment = 2^64-1`（以及 2^63）。命令：
```
$ ./harness/h cases/align_max.gguf
cases/align_max.gguf	PANIC: runtime error: slice bounds out of range [-1:]
github.com/xiaoxin-agent/llama.go/gguf.(*File).parse(...)
	/tmp/review-m0/gguf/reader.go:274 +0x1018
$ ./harness/h cases/align_2p63.gguf
cases/align_2p63.gguf	PANIC: runtime error: slice bounds out of range [-9223372036854775808:]
	/tmp/review-m0/gguf/reader.go:274
```

**根因**：gguf/reader.go:233-235
```go
pos := consumed
pad := (f.alignment - uint64(pos)%f.alignment) % uint64(f.alignment)
f.dataStart = pos + int64(pad)
```
`pos + int64(pad)` 在 int64 上溢出为负数；第 236 行 `if f.dataStart > size` 只检查上界，
负数通过检查；`dataLen := size - f.dataStart`（第 241 行）变成巨大的正数（溢出），
每个张量"都放得下"校验通过；最终 `whole[f.dataStart:]`（第 274 行）用负索引切片 -> panic。

**违反要求**：AGENTS 的健壮性目标（对损坏输入返回错误而非 panic）；GGUF 解析器不应因文件内容崩溃。
**修复**：把对齐计算放在 uint64 域并在转 int64 前做上界检查，例如
```go
if f.alignment > uint64(size) { return fmt.Errorf(...) }
rem := uint64(pos) % f.alignment
pad := (f.alignment - rem) % f.alignment
if uint64(pos) > uint64(size) || pad > uint64(size)-uint64(pos) { return ErrTruncated }
f.dataStart = int64(uint64(pos) + pad)
```
并额外拒绝非 2 的幂的 alignment（见 C.4 第 2 条）。

### C.2 非阻断：维度乘积溢出被静默接受（数据被静默丢弃）

用例：dims=[2^32, 2^32, 2^32]，f32。命令：
```
$ go run ./harness2
tensor: w dims=[4294967296 4294967296 4294967296] type=f32 offset=0 NBytes: 0 <nil>
ReadRaw err: <nil> len: 0
ElementCount: 0
Dequantize err: <nil> outlen: 0
```
`gguf/tensor.go:34-41`（NBytes）与 `gguf/tensor.go:104-110`（ElementCount）用 uint64 无溢出检查地连乘，
2^32*2^32*2^32 = 2^96 ≡ 0 (mod 2^64)。结果：一个声明 2^96 元素的张量被当作**0 字节**接受，
`ReadRaw` 返回空 slice 且 `err == nil`，下游拿到全零/空数据不会报错。q4_0 版本同样被接受（`dim_overflow_q4.gguf` OK）。
**违反要求**：损坏数据应被拒绝；这是"静默接受"而不是"报错"。
**修复**：连乘时用 `bits.Mul64` 或先判断 `d != 0 && n > math.MaxUint64/d`，溢出即返回错误；并在解析阶段对每个 dim 设合理上限。

### C.3 非阻断：零维张量（n_dims = 0）被秒收

```
$ ./harness3
cases/ndims0.gguf z dims=[] nbytes=4,<nil> rawlen=4,<nil>
```
`nDims == 0` 未被拒绝（reader.go:199-206 只检查 `> 4`），`TensorInfo{NBytes}` 对空 Dims 返回 1 个元素（4 字节）。
GGUF 规范要求 n_dims ≥ 1；应返回错误。
（注：`dims[d] == 0` 已被拒绝——`zero_dim.gguf` 返回 `tensor "z": zero dimension`，这条是对的。）

### C.4 观察项（非阻断）

1. **未对齐 offset 被拒是正确行为**，但仅当 alignment 是 2 的幂时才有意义。`align_3.gguf`（alignment=3）被接受：
   `OK: tensors=1 align=3 meta=1 readable=1`。GGUF 规范要求 alignment 为 2 的幂；非 2 的幂的 alignment
   会让 `Offset%alignment != 0` 的校验变得毫无保护（例如 alignment=3 时 offset 3 合法，落在任意字节处）。
   建议在读取 alignment 后拒绝非 2 的幂与 0（0 目前会静默回退到 32，见 `align_0.gguf` OK，这至少是安全的）。
2. **两个张量共享同一 offset（重叠）被静默接受**：
   ```
   cases/overlap_same.gguf	OK: tensors=2 align=32 readable=2
   ```
   两条张量都声明 offset 0 且 NBytes=16，读出来是同一段数据。解析器不做区间重叠检查。
   对只读加载器这不致命，但会掩盖损坏文件；建议增加"按 offset 排序后检查区间不重叠"的校验。
3. **截断**：19 个截断点全部返回错误，无 panic。含 tensor-info/data 边界：
   ```
   trunc_1709436 (info_end)      -> ERR: data section starts at 1709440, file is 1709436 bytes
   trunc_1709437/1709439         -> ERR: data section starts at 1709440 ...
   trunc_1709441 / 1709458 / 637699455 -> ERR: tensor "output.weight": unknown element type: q6_k（先撞类型错误）
   ```
   在真实文件上，因为 q6_k 不被支持，Open 在**数据校验之前**就失败，所以"真实文件的数据段越界"路径实际未被走到；
   我用构造用例 `off_exact_end`(OK) / `off_one_over`(ERR) / `off_beyond`(ERR) 验证了该路径本身是正确的。
4. **巨大声明计数**：`2^63`、`2^64-1` 均返回 `implausible counts` 错误；`5,000,000` 张量计数在真实文件上
   解析到一半时报 `string length ... too large`（错误信息误导，但确实是错误，不 panic）。
5. **重复 key / 重复张量名**：均返回错误（`duplicate metadata key`、`duplicate tensor`）。
6. **字符串/数组长度**：`str_huge`(2^40) 被 1<<28 上限拒绝；`arr_huge`(2^40) 被拒；
   `arr_268m`（恰好 1<<28，合法上限内）在第一个元素处遇到 EOF 迅速返回错误（实测 0.00s / 1.9MB RSS），
   因为元素是逐个读取的，不存在预分配攻击面。**唯一**的分配风险是 `buf := make([]byte, n)`（reader.go:281）
   在 n 恰好 = 2^28（256 MiB）且文件确实包含 256 MiB 字符串时——这是合法输入，可接受，但上限在
   64 位机器上偏宽松；建议改为与文件剩余字节数比较。
7. `bool` 字节 = 2 -> 错误（`invalid bool byte 2`）；未知值类型 -> 错误；嵌套数组 -> 错误；5 维 -> 错误；
   65 字节张量名 -> 错误，64 字节 -> 通过（llama.cpp 的 GGML_MAX_NAME=64 含 NUL，这里允许 64 字节非 NUL 字符串，
   比上游宽松 1 字节，非阻断）。

### C 结论
**存在 1 个阻断缺陷（对齐溢出 panic，reader.go:234/274）**；2 个静默接受缺陷（维度乘积溢出、零维张量），
1 个非 2 幂 alignment 未校验，1 个重叠区间未校验。其余损坏输入均正确报错。

---

## D. README 的 "与 HuggingFace config.json 完全一致" 声明

### D.1 用 README 自己给的方法跑，结果是失败

```
$ cd /tmp/review-m0 && go run tools/gguf_info.go /tmp/tiny-Q4_0.gguf
open: /tmp/tiny-Q4_0.gguf: tensor "output.weight": gguf: unknown element type: q6_k
exit status 1
```
**一行输出都没有**——失败发生在 `gguf.Open` 内部（reader.go:256-261 的逐张量 NBytes 校验），
而不是在打印阶段。也就是说：**这个解析器无法打开任何一个混合量化（Q4_0 + Q6_K 输出层）的真实模型**，
而 README 抽样的那个 Q8_0 文件恰好是纯 Q8_0，才没暴露问题。

### D.2 这对 "loader completeness" 意味着什么（与超参数问题分开看）

1. **解析层被反量化支持度绑死**：`Open` 在解析阶段就要求**每一个**张量的类型都能算出字节数
   （`TypeSize` 只支持 f32/f16/q4_0/q8_0，gguf/tensor.go:19-42）。
   一个纯元数据/张量表解析器本不需要理解 q6_k——它只需要信任文件里声明的数据段长度，或者延迟到 ReadRaw 再判断。
   现在的设计导致：**任何含 Q5/Q6/K-quant 层的模型全部打不开**，包括最常见的 Q4_K_M、Q5_K_M、Q6_K 文件。
   这直接使 M0 的"M1 gate: metadata matches gguf_dump exactly" 在当前实现下不可达。
2. **README "Verified against a real model" 的示例不可复现**：README 引用的文件名
   `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` 在仓库中不存在，评测环境也没有该文件（只有 Q4_0 混合版）。
   我**无法**独立重现该次运行，因此那 7 行输出只能作为作者自述，不能算已核实证据。
3. 这是否是"已知限制"？代码里 `TypeSize` 的注释说 "Unknown types return ErrUnknownType together with the type so callers can report it"，
   但 `parse()` 把它当致命错误，与注释意图不一致。
   建议：`Open` 允许未知类型（NBytes 推迟），或在 parse 时用"文件长度 - 对齐后偏移"兜底，仅在 ReadRaw 时报错。

### D.3 超参数字段逐项比对（绕开 loader，用独立 Python 解析器）

由于 `go run tools/gguf_info.go` 无法产出任何输出，我用 /tmp/reviewer-m0/py/gguf.py（独立实现）
解析同一文件的元数据，并与
`https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/config.json`
（已保存至 /tmp/reviewer-m0/hf-config.json）逐项比较：

```
llama.block_count                       = 22          | num_hidden_layers         = 22         | MATCH
llama.embedding_length                  = 2048        | hidden_size               = 2048       | MATCH
llama.feed_forward_length               = 5632        | intermediate_size         = 5632       | MATCH
llama.attention.head_count              = 32          | num_attention_heads       = 32         | MATCH
llama.attention.head_count_kv           = 4           | num_key_value_heads       = 4          | MATCH
llama.attention.layer_norm_rms_epsilon  = 9.999999747378752e-06 | rms_norm_eps  = 1e-05      | MATCH (float32 舍入, |diff|=2.5e-13)
llama.rope.freq_base                    = 10000.0     | rope_theta                = 10000.0    | MATCH
llama.context_length                    = 2048        | max_position_embeddings   = 2048       | MATCH
tokenizer.ggml.bos_token_id             = 1           | bos_token_id              = 1          | MATCH
tokenizer.ggml.eos_token_id             = 2           | eos_token_id              = 2          | MATCH
llama.rope.dimension_count              = 64          | hidden_size/num_heads     = 64         | MATCH (派生量)
tokenizer.ggml.tokens count             = 32000       | vocab_size                = 32000      | MATCH
```
**超参数本身完全正确。** 文件中的 fp32 值 `9.999999747378752e-06` 是 `1e-05` 转 float32 的必然结果，
README 引用的也是这个值，属于正确的舍入，不是缺陷。

**D 结论**：README 的超参数声明为真（就本 Q4_0 文件而言，我独立核实）；
但 README 声称的"用 tools/gguf_info.go 解析并输出"的**方法**在同类文件上不成立，
且 loader 对 q6_k 的拒绝是实质性的完整度缺陷（非阻断 M0 的 dequant 范围，但阻断 M1 gate 与任何真实模型加载）。
README 中 "Status: M0 complete" 与 "Verified against a real model" 这两句需要修正。

---

## E. 覆盖率与未测试路径的风险

命令与输出（未修改仓库）：
```
$ cd /tmp/review-m0 && go test ./... -cover -count=1
ok  github.com/xiaoxin-agent/llama.go/gguf    coverage: 52.7% of statements
ok  github.com/xiaoxin-agent/llama.go/quant   coverage: 93.9% of statements
```
README 中写的 52.7% / 93.9% 与实测一致（这一项属实）。
另外 `go vet ./gguf/... ./quant/...` 通过，`-race` 下测试通过。

### 真正有正确性风险的低覆盖路径（按风险排序）

1. **gguf/reader.go:309-453 `readValueBody` —— 28.6%**（最高风险）。
   13 种值类型只测到了 uint32 / string / array-of-string 三种。未覆盖：
   - `ValUint8/16/64`、`ValInt8/16/32/64`、`ValFloat32/64`、`ValBool` 的**标量**分支；
   - **数组元素**为 uint8/16/32/64、int8/16/32/64、float32/64、bool 的分支（reader.go:437-450）。
   风险：这些分支各自有独立的手写 `binary.Read` + `*consumed += n` 记账。`consumed` 一旦算错，
   `dataStart` 就错，随后所有张量 offset 都偏——**而现有测试只用 uint32/string，抓不到任何尺寸记账错误**。
   我构造了包含全部 13 类标量 + 10 类数组元素类型的文件（/tmp/reviewer-m0/cases/alltypes.gguf）验证它们
   在当前实现下能正确解析（间接说明记账没错），但**仓库自身没有这个测试**。建议补上表驱动测试。
   实际优先级：中——因为 llama.cpp 模型目前只用 uint32/float32/string/bool/array。
2. **gguf/reader.go:33-49 `Value.String` / reader.go:73-95 `any()` —— 0%**。
   `String()` 对数组打印 `len(v.Strs)+len(v.Floats)+...`，对**空数组**和各类型混合是安全的，
   但 `any()` 遇到 `ValArray` 返回 nil，`String()` 对 array 不走 `any()`，暂无 panic。
   主要风险：`tools/gguf_info.go` 依赖它输出，从未被测试；由于该工具在真实文件上直接打不开（见 D.1），
   **工具本身从未被真实数据跑通过**。
3. **gguf/reader.go:234 对齐计算 —— 被覆盖（76% 的 parse 里含此行），但无边界测试**。
   这就是 C.1 panic 所在；语句覆盖率高不代表正确性。**必须新增**：alignment=2^63/2^64-1/非 2 的幂 的测试。
4. **gguf/tensor.go:43-52 `ElementCount` —— 0%**；`NBytes`（reader.go:27）虽然 100% 覆盖，但只在单元素乘法
   下被测（fixture 只有 1 维 [4]）。**维度乘积溢出（C.2）完全无测试**。
5. **gguf/metadata.go:105 `ReadRawByName` —— 0%**、`MetadataKeys`/`TensorNames`/`Lookup`/`Path` —— 0%。
   逻辑简单，风险低；但 `ReadRawByName` 是主要公开 API，至少应有一个命中/未命中用例。
6. **gguf/reader.go:455-476 `Close` —— 84.6%**：未覆盖 `Munmap` 失败和 `f.f.Close()` 失败分支，
   以及 Close 后再 Close 的行为（`f.mapLen=0` 后二次 Close 返回 nil，安全）。
7. **quant 包 93.9%**：唯一未覆盖的是 `Kind.String()` 的 default 分支（dequant.go:31）和
   `RowStride` 的 F16/F32 负 cols 分支。**无正确性风险**。

### 覆盖率"数字低但不重要"的部分
- `types.go` 的 `Writer.write` 46.4%：Writer 仅用于测试 fixture，未覆盖的分支（int8/float32/等）风险低，
  但注意它**没有 WriteString 长度的失败路径测试**，且 `write` 在部分写成功后仍继续接受数据的注释与行为
  （types.go:150-156）未验证。
- `types.go:106 ValueType.String` / `tensor.go:12 ErrNoTensor.Error` 0%：只是格式化，无风险。


---

## 结论与发现清单

**BLOCKING** — 存在 1 个由损坏输入触发的 panic，可直接用于拒绝服务；修复点明确且很小。

### 阻断项（1）

**F1 [阻断] `general.alignment` 溢出导致解析器 panic · gguf/reader.go:234（panic 点为 reader.go:274）**
- 违反：解析器对损坏输入必须返回 error 而非 panic。
- 复现：`./harness/h cases/align_max.gguf` →
  `PANIC: runtime error: slice bounds out of range [-1:]`；`cases/align_2p63.gguf` → `[-9223372036854775808:]`。
- 修复：见 C.1 中的 uint64 域对齐计算 + alignment ≤ 文件大小 + 2 的幂校验。

### 非阻断项（按影响排序）

**F2 [应修] 维度乘积溢出被静默接受，声明 2^96 元素的张量变成 0 字节 · gguf/tensor.go:34-41, 104-110**
- 复现：`cases/dim_overflow.gguf` → `NBytes: 0 <nil>`, `ReadRaw err: <nil> len: 0`, `Dequantize err: <nil>`。
- 修复：连乘前 `if d != 0 && n > math.MaxUint64/d { return 0, err }`。

**F3 [应修] 零维张量（n_dims=0）被接受 · gguf/reader.go:199-206**
- 复现：`cases/ndims0.gguf` → 解析成功，Dims=[]，NBytes=4。
- 修复：`if nDims < 1 || nDims > 4`。

**F4 [应修] 解析器无法打开任何含 q5/q6/K-quant 的真实模型 · gguf/reader.go:253-261 + gguf/tensor.go:19-42**
- 复现：`go run tools/gguf_info.go /tmp/tiny-Q4_0.gguf` → `open: ... unknown element type: q6_k`，零输出。
- 影响：M1 gate（元数据与 gguf_dump 一致）不可达；README "Verified against a real model" 的方法不可复现。
- 修复：parse 阶段不要求类型可解析，或用 `dataLen - offset` 兜底，仅在 ReadRaw 报 ErrUnknownType；或至少实现 Q6_K 的字节尺寸。

**F5 [建议] 非 2 的幂 alignment 未校验 · gguf/reader.go:178-181**
- 复现：`cases/align_3.gguf` → `OK: tensors=1 align=3 meta=1 readable=1`；此时 offset 对齐检查失去意义。

**F6 [建议] 张量区间重叠未被检测 · gguf/reader.go:250-262**
- 复现：`cases/overlap_same.gguf` → `OK: tensors=2 ... readable=2`（两个 16 字节张量共享 offset 0）。

**F7 [建议] 元数据/张量表 75% 以上的值类型分支无测试 · gguf/reader.go:309-453（覆盖率 28.6%）**
- 现有 fixture 只覆盖 uint32/string/array<string>；`consumed` 记账错误不会被任何测试发现。

**F8 [建议] 测试"独立参考实现"的独立性声明不实 · quant/dequant_test.go:15-18**
- `refDequantQ4_0` 与生产代码同文件同作者，算法理解同源；建议把注释改成如实的表述，
  并补一个真实 gguf 文件的外部锚点测试（见 A.3）。

**F9 [建议] README 修正 · README.md:24-33**
- 移除或标注"Verified against a real model"；`Status: M0 complete` 应保留但注明 q6_k loader 缺失。

### 已核实为正确的部分
- q4_0 / q8_0 反量化算术、块大小、nibble 顺序、scale 位置：与 ggml 上游逐行一致，并在真实数据上与独立 Python 实现 **bit-exact**（差值 0.0）。
- 截断（19 个切点，含 tensor-info/data 边界）、重复 key、重复张量名、超大计数、超大字符串/数组长度、
  越界 offset、未对齐 offset（对齐为 2 的幂时）、未知类型、非块对齐元素数：均返回错误，无 panic。
- README 引用的 7 个超参数 + 2 个 token id 与 HF config.json 逐项一致。

---

## 无法核实

1. **README 中 Q8_0 文件的运行结果**：`tinyllama-1.1b-chat-v1.0.Q8_0.gguf` 在仓库和本机均不存在，
   无法重现 "201 tensors / 23 metadata keys / 1,170,781,568 bytes" 那一组数字。本次 Q4_0 文件的输出是
   201 tensors / 23 metadata keys，数值上一致但文件不同、无法确认作者当次运行。
2. **`go run tools/gguf_info.go` 在真实模型上的完整输出**：因 q6_k 拒绝，工具在任何真实量化模型上都跑不出结果，
   包括评测用的这唯一一个模型文件。所有元数据结论来自我独立写的 Python 解析器。
3. **Q5/Q6/K-quant 反量化**：不在 M0 范围，未评审。
4. **大文件 mmap 在 32 位平台上的行为**：`int(size)`（reader.go:268、270）在 32 位平台会溢出，但目标平台为 64 位 ARM64，
   未实测 32 位构建。
5. **并发安全**：`File` 的 `ReadRaw` 返回 mmap 别名，多个 goroutine 同时读是安全的，但 `Close` 与读并发会
   触发 SIGSEGV（不是 Go panic，recover 抓不到）。未构造该用例，属未验证风险。
6. **`-race` 下的全部测试**：已在 gguf/quant 上跑过并全部通过；未对仓库外的 harness 跑 race。
7. 未修改被评审仓库中的任何文件；所有新增测试/构造器位于 /tmp/reviewer-m0/。
