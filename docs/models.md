# 模型与校验值

> M0 交付物之一：下载的 GGUF 文件及其 sha256。
> 校验值用于确认后续所有比对都在同一份字节上进行；换了文件就必须重算。

## 主模型：TinyLlama-1.1B-Chat-v1.0（Q4_0）

```
文件      tiny-Q4_0.gguf
大小      637,699,456 字节
sha256    da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556
来源      https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

已用 `cmd/gguf-inspect` 核验可打开：

```
version    = 3
alignment  = 32
data_start = 1709440
tensors    = 201
metadata   = 23 keys
```

注意：该文件并非纯 Q4_0。`output.weight` 与 `token_embd.weight` 为 **q6_k**，这是
llama.cpp 的量化惯例（把最敏感的两张量单独提高精度）。这使 Q6_K 反量化成为读取
真实权重的前置条件，而不是可选项。

## CI 小模型：SmolLM2-135M-Instruct（Q8_0）

```
文件      smollm2-135m-q8.gguf
大小      144,811,072 字节
sha256    40f7094960b6ede829145d102ca79451b364b27d9d8694d4406e002024cff357
来源      https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q8_0.gguf
```

已用 `cmd/gguf-inspect` 核验可打开：

```
version    = 3
alignment  = 32
data_start = 1785664
tensors    = 272
metadata   = 33 keys
```

用途：CI 中的回归模型。层数多（30）但每层极窄，跑一遍快；架构特征与主模型不同
（tied embedding、9 heads / 3 kv heads、rope theta 不同），可验证实现的泛化性而非
只拟合一个模型。

## 复核命令

```bash
sha256sum <file>          # 与上表比对
go run ./cmd/gguf-inspect -meta-only <file>   # 结构可读性
```

## 历史文件

`tinyllama-q8.gguf`（1,170,781,568 字节）曾在 `/tmp`，现已不存在。若需要请重新下载
并补记 sha256，不要凭旧记录假设其内容。
