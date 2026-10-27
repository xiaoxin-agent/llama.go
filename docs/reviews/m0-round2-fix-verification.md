# 独立评审报告 M0（第二轮）：F1 修复验证

**结论行：FIXED-BUT-NEW-ISSUES**

- 原阻断缺陷 F1（`general.alignment` 溢出 panic）：**已修复并已验证**（原用例返回 error，无 panic）。
- 修复的第二层（uint64 域算术）**独立有效**，不是只靠第一层校验兜着。
- 回归测试**确实有区分力**：原样搬到旧代码 daa7dfe 上失败 5 处；但它**只覆盖第一层**，第二层零覆盖。
- 新问题 1 项（N1，应修）：uint32 编码的 alignment 被截断后报出误导性错误文本。
- 新问题 1 项（N2，建议）：回归测试的固定装置从不进入真正会 panic 的数据段路径。
- 原 F2–F9 中，F5 被新校验**顺带修复**；F9 因覆盖率数字未更新而**部分变差**；其余不变。

评审对象：/tmp/review-m0 @ f4a96d4（分支 m0/scaffold），基线 daa7dfe。
`git diff daa7dfe..f4a96d4` 实测仅改动 gguf/reader.go (+23/-7)、gguf/reader_test.go (+56)，与任务描述一致。
环境：Go 1.23.4（linux/arm64）。构造物全部位于 /tmp/reviewer-m0-v2/（独立 module + 自写 Python GGUF 写出器）；
旧版本测试通过 `git worktree add /tmp/reviewer-m0-v2/wt-old daa7dfe` 完成，结束后已 `worktree remove`，
/tmp/review-m0 最终 `git status` 为空（我中途为定位问题临时插桩过 reader.go，每次均已从备份恢复并用
`md5sum`/`git status` 确认与 HEAD blob 一致；所有临时测试文件已删除）。

---

## 1. 原 panic 是否消失

用自写生成器重建等价用例（`general.alignment` 以 ValUint64 写入，文件带真实数据段与 1 个张量）：

```
$ ./h/h c3/a9223372036854775808.gguf c3/a18446744073709551615.gguf
c3/a9223372036854775808.gguf  ERR: ... general.alignment 9223372036854775808 is implausibly large (max 1048576)
c3/a18446744073709551615.gguf ERR: ... general.alignment 18446744073709551615 is implausibly large (max 1048576)
```

第一轮原始输出为 `PANIC: runtime error: slice bounds out of range [-9223372036854775808:]` 与 `[-1:]`
（panic 点 reader.go:274）。现在**返回 error，无 panic**。2^62、2^40 同样被拒。

**结论：F1 已关闭。**

## 2. 第二层是否独立有效（关键）

如果只有第一层在起作用，"未来算术改动仍安全"的承诺就是空的，所以我绕过校验直接测算术：
把 gguf 包复制到独立 module，在包内测试中**直接赋值 `f.alignment`**（不经过校验）后复算 reader.go:248-251：

```
align=9223372036854775808  pad=9223372036854775792  start=9223372036854775808  wrapOrOOB=true
align=18446744073709551615 pad=18446744073709551599 start=18446744073709551615 wrapOrOOB=true
align=4611686018427387904  start=4611686018427387904   wrapOrOOB=true
align=1099511627776        start=1099511627776         wrapOrOOB=true
align=128                  start=128                   wrapOrOOB=false（合法值正确放行）
```

**结论：第二层独立成立。** 另已核对全文件中 `f.dataStart` 仅在 reader.go:254 被赋值一次，
`whole[f.dataStart:]` 的 `dataStart ∈ [0, size]` 不变量由该检查唯一保证。

## 3. 尝试攻破修复（均未成功）

| 向量 | 用例 | 结果 |
|---|---|---|
| 恰好 1 MiB（真实文件，带 1 MiB 填充后的数据段） | `c3/real_1mib.gguf` | **OK，读取数据正确** `[1 2 3 4]` |
| 1 MiB−1 / 1 MiB+1 / 2 MiB | `a1048575`/`a1048577`/`a2097152` | 均为 ERR（非 2 幂 / 超上限 / 超上限） |
| 高位置 1 且为 2 的幂（2^63、2^62） | `a9223372036854775808`、`a4611686018427387904` | ERR: implausibly large |
| 2 的幂但远大于文件（2^40、512 KiB、1 MiB） | 见上 | 全部 ERR（truncated，不 panic） |
| 0 / 3 / 48 / 负数 | `a0`/`a3`/`a48`/`i32_neg` | 全部 ERR |
| 合法 16/32/64/128 | `a16`/`a32`/`a64`/`a128` | 全部 OK |
| 由声明的 size/alignment 组合绕过 uint64 界 | 第 2 节 6 组极端值 | 未绕过 |
| 非 uint64 编码 | `u32_2p40`、`u32_2p31`、`i32_neg`、`str` | 见 N1；`str` 类型键被忽略并回退 32（合理） |

**误拒检查**：1 MiB 内所有 2 的幂取值（2..1048576）均被接受，实测 16/32/64/128/512Ki/1MiB 端到端可读，
**未发现对合法文件的误拒**。风险仅在于"1 MiB 上限"是自选阈值而非规范值（见"无法核实"）。

## 4. 新问题

### N1 [应修] uint32 编码的 alignment 被截断后报出误导性错误

`f.Uint()`（gguf/metadata.go:17-29）对 ValUint8/16/32/64 **统一转成 uint64**，`parse` 拿不到原始宽度：

```
c3/u32_2p40.gguf (uint32 值 = 2^40 → 截断为 0)
  -> ERR: general.alignment is zero          ← 文件里写的不是 0，错误文本误导
c3/u32_2p31.gguf (uint32 值 = 2^31)
  -> ERR: general.alignment 2147483648 is implausibly large (max 1048576)   ← 正确但未指出编码问题
```

**严重性**：不导致 panic，安全上没有问题（2^40 的 u32 值本就不可能被正确表示）。
**真正的问题是可诊断性**：一份"声明了巨大对齐"的损坏文件会被报成"没写对齐"。
另外，修复把 v==0 的处理从"静默回退 32 并继续"改成**硬失败**——对真正写 0 的文件这是**更正确**的行为
（规范要求 2 的幂），但它是一个**未在提交信息中声明的行为变化**（第一轮 C.4 记录的 `align_0 → OK align=32`
现在不再成立）。建议：v==0 且 `Kind != ValUint64` 时报 `alignment does not fit its declared type`。

### N2 [建议] 回归测试从未执行到会 panic 的代码路径

`TestAlignmentValidation`（reader_test.go:257-308）构造的文件：头部 + **kv=0** + **tensor=1** + 1 条对齐 KV +
64 字节零。实测布局 `consumed=61`，对齐到 64，文件长 125，剩余 61 字节。
注意 `tensor count = 1` 但没有张量表——然而代码先走 `dataLen == 0` 分支，
在 `len(f.order) > 0` 判断处**直接 return nil**（f.order 为空），
所以 reader.go:248-251 的 uint64 算术虽被执行，但**失败分支从未触发**，且 267 行之后的张量校验/切片全部未执行。

只跑该测试的覆盖率（`go test ./gguf -run TestAlignmentValidation -coverprofile`）证实：

```
reader.go:248.2,251.41   4 1    <- 算术块执行了
reader.go:251.41,253.3   1 0    <- 越界/回绕分支：0 次
reader.go:267.2,267.31   1 1    <- dataLen==0 判断
reader.go:267.31,270.17  3 0    <- 之后所有数据段路径：0 次
gguf coverage: 31.4% of statements
```

**该测试证明的是第一层校验，不是第二层算术。** 这与 §2 的结论一致：第二层目前**没有任何测试覆盖**。
建议补一个带真实张量与数据段的用例（如 `c3/real_1mib.gguf` 的形状），并断言错误子串而非仅 `Contains("alignment")`。

### 旧代码上的表现（证明测试有区分力）

把该测试原样放入 daa7dfe 的 worktree：

```
--- FAIL: TestAlignmentValidation
  alignment 9223372036854775808 panicked: slice bounds out of range [-9223372036854775808:]
  alignment 18446744073709551615 panicked: slice bounds out of range [-1:]
  alignment 1099511627776: error does not mention alignment: ... data section starts at 1099511627776, file is 125 bytes
  alignment 48: accepted, want rejection
  alignment 0: accepted, want rejection
```

**5 处失败 → 测试确实有效，不是新旧都通过的空测试。**
附带发现：`1<<40` 在旧代码上是**被拒绝的**（只是错误文本不含 "alignment"），
所以该条断言实际上只测了错误文本措辞；删除第一层后若仅靠第二层，这条会以"措辞"理由报错，属于脆弱的断言。

## 5. 对原 F2–F9 的影响

| 编号 | 现状 | 证据 |
|---|---|---|
| F2 维度乘积溢出 | **未变**（仍然静默接受） | `c2/dim_overflow.gguf`: dims=[2^32,2^32,2^32] → `nbytes=0 nbErr=<nil> rawlen=0 rawErr=<nil> elems=0` |
| F3 零维张量 | **未变**（仍接受） | `c2/ndims0.gguf`: dims=[] → `nbytes=4 rawlen=4`，无错误 |
| F4 q6_k 无法打开 | **未变**（预期，未作为新问题重复上报） | 真实模型文件仍失败 |
| F5 非 2 幂 alignment | **已被顺带修复** | `a3`/`a48` → `not a power of two` |
| F6 张量区间重叠 | **未变**（仍接受） | `c2/overlap.gguf`: 两个张量都 offset 0、都读出 16 字节 |
| F7 值类型分支无测试 | **未变** | 新测试只写 ValUint64 一种 |
| F8 "独立参考实现"措辞 | **未变** | 未修改 |
| F9 README | **部分变差（新增）** | README.md:17 仍写 gguf 覆盖率 `52.7%`，实测已 `54.6%` |

## 6. 全量测试

```
$ export PATH=$HOME/.local/go/bin:$PATH
$ cd /tmp/review-m0 && make verify
go vet ./...
go test ./... -race -count=1
ok  github.com/xiaoxin-agent/llama.go/gguf   1.031s
ok  github.com/xiaoxin-agent/llama.go/quant  1.032s
go test ./... -count=1 -cover
ok  github.com/xiaoxin-agent/llama.go/gguf   0.008s  coverage: 54.6% of statements
ok  github.com/xiaoxin-agent/llama.go/quant  0.005s  coverage: 93.9% of statements
```

关键前提：**这是干净工作树的结果**。第一轮遗留的 `harness5/` 等目录已不存在，
我本次新增的临时测试文件全部删除，`git status` 为空。覆盖率 52.7% → 54.6% 的说法**属实**。
（顺带说明：若工作树里残留任何含 `func Test` 的目录，`make verify` 会变更/失败，说明当前门禁没有隔离检查。）

---

## 无法核实

1. **1 MiB 上限是否有真实文件逾界**：未找到写入 >1 MiB 的 2 的幂 alignment 的 GGUF 样本。
   若存在，本解析器会拒绝而 llama.cpp 可能接受——属潜在兼容性差异，非已证实回归。
2. **上游 llama.cpp 对 "uint32 编码的 2^40 alignment" 的具体行为**：未构建上游 C++ 复现，N1 涉及的上游对照仅基于代码阅读。
3. **第二层算术在真实 panic 场景下的端到端证明**：只能通过包内直接赋值字段验证（§2）；由于第一层总是先拦截，
   无法用外部文件走通"第二层独自拯救"的路径。这既是 N2 的原因，也是本条的限制。
