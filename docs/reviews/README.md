# M0 评审记录

本目录保存 M0 里程碑的评审报告。报告与代码版本绑定：审的是哪个 commit，就在这里写明。

## 索引

| 文件 | 轮次 | 评审者 | 被审 commit | 裁决 |
|---|---|---|---|---|
| `m0-round1-author-selfreview.md` | 第 1 轮 | Coordinator（**即代码作者**） | `daa7dfe` | ~~APPROVE WITH CHANGES~~ **已取代** |
| `m0-round1-independent.md` | 第 1 轮 | team-reviewer（独立） | `daa7dfe` | **BLOCKING** |
| `m0-round2-fix-verification.md` | 第 2 轮 | team-reviewer（独立） | `f4a96d4` | **FIXED-BUT-NEW-ISSUES** |

## 为什么要保留三份

第 1 轮存在两份报告，结论相反。作者自评判"无阻断缺陷"，独立评审判 **BLOCKING**，找出了作者漏掉的
`general.alignment` 整数溢出 panic。两份都保留，因为它们本身是证据：**代码作者不能当自己代码的评审者**。
自评里仍有两项（Q6_K 缺口、张量重叠未检测）未被独立评审覆盖，仍有参考价值。

## 时间线

1. `daa7dfe` — M0 初始提交：仓库脚手架、GGUF v3 解析器、Q4_0/Q8_0/F16 反量化。
2. 第 1 轮评审。独立评审发现 F1（对齐值溢出导致 panic）为阻断缺陷。
3. `f4a96d4` — 修复 F1：解析时校验 `general.alignment`（非零、2 的幂、≤ 1 MiB），
   并在 `uint64` 域内计算数据段起点后降窄，杜绝负索引。
4. 第 2 轮评审，只审该修复。结论：panic 确已消失，回归测试在旧代码上确实失败（有证明力），
   但**新回归测试的夹具不含 tensor 表，从未执行到它本应保护的算术**；
   另发现 `Uint()` 对 `ValUint32` 不做宽度判断，截断后给出误导性错误信息。

## 未关闭项

第 2 轮结束时仍未修复：

- 回归测试夹具缺 tensor 表 → `gguf/reader.go:251-253` 越界分支执行 0 次（第二层防御无测试覆盖）
- `Uint()` 窄类型截断（`gguf/metadata.go:17-29`）
- 维度乘积溢出：声明 2^96 元素 → 0 字节张量且 `err=nil`
- `n_dims=0` 静默接受
- 张量载荷重叠未检测
- `readValueBody` 覆盖率 28.6%，13 种值类型里 10 种的宽度记账未测
- Q6_K 等 K-quant 类型未支持 → 真实 Q4_0 模型无法打开
