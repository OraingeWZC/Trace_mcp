# TADBench 整合方案

> 主题：把 `/Users/wzc/Project/bk/TADBench/` 仓库及其论文成果作为 trace 相关工作的**师门内部 baseline 库 + 公开数据集源**接入到本课题（2024YFB4505903）。
> 状态：分析与方案，未改动任何代码。
> 配套阅读：`项目全景与对比分析.md`（你的工作在课题地图上的位置）、`范式对比与改造思路.md`（评测对齐路径）。

---

## 0. 重要前置说明

### 0.1 论文身份与归属

- **正确论文**：`/Users/wzc/Project/bk/TADBench/TSC3622122.pdf`
  *"A Comprehensive Benchmark and Empirical Study of Trace Anomaly Detection"*
  Yongqian Sun (孙永谦, 南开)、Minyi Shao、Xiaohui Nie、Kaiwen Yang、Xingda Li、Bowen Hao、Shenglin Zhang (张圣林, 南开)、Changhua Pei (裴昶华, CNIC/CAS)、Dongbiao He、Yanbiao Li、**Dan Pei (裴丹, 清华)** —— IEEE TSC（Transactions on Services Computing）。
- **关键身份**：**裴丹是合作者**——他既是这篇 TADBench 论文的作者之一，又是你们课题（2024YFB4505903）的负责人。也就是说 **TADBench 不是外部第三方 baseline，而是你们师门生态内部的成果**。
- **作废文件**：`bk/TADBench/2212.09518v1.pdf` 是 FedTADBench（不相关），可以删除或标记为误下载。

### 0.2 论文关键数字

| 维度 | 数值 |
|---|---|
| 数据集总规模 | 3.6 GB / 约 104 万条 trace / 约 21 万条标注 |
| 数据集 | TrainTicket / GAIA / AIOps2020 / AIOps2022 / AIOps2023 |
| 算法 | 7 个，分 3 类（VAE-based / GNN-based / LSTM-based） |
| 开源地址 | https://github.com/nkalgo/TADBench.git |

---

## 1. 与linyu TrUST 工作的核心关联（最关键的发现）

### 1.1 TADBench 论文末尾的"未来方向"= TrUST 的实现

论文 §VI 的 *"Building upon the strengths of prior methods"* 段落写道：

> ...future anomaly detection methods could integrate **the entropy gap reduction strategies pioneered in TraceVAE**. Specifically, this involves implementing **Bernoulli & Categorical Scaling** for structural anomaly identification, **Node Count Normalization** for dimensional consistency, and **Gaussian Std-Limit** thresholding for latency anomalies. These techniques are to be synergistically combined with the **hierarchical graph encoding architecture of GTrace**, which separates global structure modeling from node-level feature processing through its innovative dispatching layer. Furthermore, the entire system can leverage **GTrace's optimized caching strategy** that utilizes dynamic programming and LRU-cached trees to enable batched processing of merged subgraphs.

**关键时间线**：TADBench 论文（IEEE TSC, 2025）发表时linyu TrUST 还没开工。TrUST 是**后续在该课题（2024YFB4505903）中独立开展**的工作，方向上恰好回应了 TADBench 的呼吁——这是**独立研究路径的顺承关系**，不是事先按论文呼吁实现。

**这段话和linyu TrUST 的创新点对照如下**：

| 中期报告 3.3 节（TrUST 创新点） | TADBench 论文未来方向 |
|---|---|
| Bernoulli-Categorical Scaling（伯努利-分类缩放） | ✅ Bernoulli & Categorical Scaling |
| Gaussian Standard Deviation Constraint（高斯标准差限制） | ✅ Gaussian Std-Limit |
| Node Count Normalization（节点计数归一化） | ✅ Node Count Normalization |
| 结构子网络（GNN 编码 + GNN 解码） + 时序子网络（MLP 编码 + 树结构 LSTM 解码） | ✅ 结合 TraceVAE entropy 优化 + GTrace 层次化图编码 |
| 不确定性感知自适应联合优化 | ⊕ 进一步突破 |

**结论**：linyu TrUST 的工作方向与 TADBench 论文给出的 *future direction* 高度吻合，可以视为该方向上的**后续独立探索**。论文谱系上：TraceVAE (WWW'23) → GTrace (FSE'23) → **TrUST (本课题)**——表达时建议用"在……基础上进一步发展"或"沿……提出的方向探索"，避免"实现"这类隐含因果的措辞。

### 1.2 引用建议

- linyu论文如果要发表，引这篇 TADBench 时可以直接说："本工作回应了 [TADBench, TSC'25] 中提出的研究方向，将 TraceVAE 的熵减策略与 GTrace 的层次化图编码协同集成。"
- 结题报告 3.3 节可以引这篇 TADBench 作为权威综述背书"已有方法对比"，让 TrUST 的定位更清晰。

---

## 2. TADBench 论文核心结论速读

### 2.1 算法分类（论文 §II-C）

| 类别 | 算法 | 特点 |
|---|---|---|
| **VAE-based** | TraceAnomaly / CRISP / TraceVAE / GTrace | 重构正常 trace 模式，重构误差判异常 |
| **GNN-based** | PUTraceAD / TraceCRL | 图神经网络捕获结构与依赖关系 |
| **LSTM-based** | Multimodal LSTM | 序列建模时延+结构 |

### 2.2 数据集真实异常比（论文 Table II）

| Dataset | Total | Structure | Latency |
|---|---|---|---|
| TrainTicket | 32.9% | 26.7% | 28.5% |
| GAIA | **49.9%** | 21.8% | 29.9% |
| AIOps2020 | **21.8%** | 3.5% | 21.5% |
| AIOps2022 | 36.3% | 4.0% | 35.9% |
| AIOps2023 | **53.6%** | 39.0% | 24.0% |

注意：GAIA 和 AIOps2023 异常比异常高（接近 50%），这是因为标注流程把每个故障注入时段的 trace 都打了标，而非真实生产分布。

### 2.3 总体性能（论文 Table III，F1%）

| Algorithm | TrainTicket | GAIA | AIOps2020 | AIOps2022 | AIOps2023 |
|---|---|---|---|---|---|
| Multimodal LSTM | 68.5 | 89.8 | 54.5 | 59.2 | 64.4 |
| TraceAnomaly | 62.6 | 46.3 | 56.8 | 58.5 | 55.4 |
| CRISP | 62.5 | 44.5 | 57.0 | 58.1 | 66.4 |
| PUTraceAD | 95.4 | 68.6 | 48.3 | 68.1 | **74.7** |
| TraceCRL | 55.6 | 75.0 | 53.0 | 53.2 | 44.2 |
| TraceVAE | 97.3 | **90.9** | 57.1 | **78.9** | 74.0 |
| GTrace | **99.4** | 70.9 | **71.8** | 76.5 | 70.3 |

**RQ1 关键结论**：没有任何算法在所有数据集上一致最优。

### 2.4 算法选择决策树（论文 Fig. 18，结题选模型直接照抄）

```
         anomaly ratio < 10%  ──────┐
                                    │
                                    └── span count ≤ 5  → GTrace
                                        span count 5~30 → trace depth ≤ 3 → TraceVAE
                                                          trace depth > 3 → anomaly ratio ≤ 1% → GTrace
                                                                            anomaly ratio 1~3% → TraceVAE
                                                                            anomaly ratio > 3% → GTrace
                                        span count > 30 → GTrace

         anomaly ratio ≥ 10% → PUTraceAD
```

**给你们的提示**：你们故障注入数据集异常比通常在 35%~50%（参考赛西测试 YS02-013：12000 样本中 1200 异常 ≈ 10%；YS02-009：6229 样本中大部分是异常 ≈ 70%+），按此决策树，**PUTraceAD 是 TADBench 里最该重点对照的 baseline**，不是 TraceVAE 或 GTrace。

### 2.5 时间效率（论文 Table V，TrainTicket 51K trace）

| Algorithm | Training Time | Detecting Speed |
|---|---|---|
| Multimodal LSTM | 419s | 4500 traces/s |
| TraceAnomaly | 3422s | 107 traces/s |
| CRISP | 3827s | 107 traces/s |
| TraceCRL | 13176s | 700 traces/s |
| PUTraceAD | 1715s | 548 traces/s |
| TraceVAE | 23497s | 257 traces/s |
| **GTrace** | **1641s** | **10211 traces/s** ← 远超其他 |

**结论**：TraceVAE 性能最强但训练 6.5 小时，GTrace 性能优且快 40 倍。生产部署 GTrace > TraceVAE，论文打榜两者都要试。

---

## 3. 与 wzc + linyu工作的契合度（更新版）

### 3.1 指标 3.2（轨迹单模态）— **师门内部对照**

由于 TADBench 是裴丹组合作产出，**TrUST 与 TADBench 7 个 baseline 的对照实验本质上是同门技术演进路线对照**。建议路径：

| 任务 | 你们的方法 | TADBench 对照 baseline | 论文叙事 |
|---|---|---|---|
| AD | TrUST（linyu） | TraceVAE / GTrace / PUTraceAD | "在 TraceVAE/GTrace 基础上集成 entropy 优化 + 层次结构" |
| RCA | TrUST 柔性聚合 | （TADBench 严格无 RCA 任务）| 相对优势 |
| CLS | trace_sv_diag（wzc） | PUTraceAD | "PUTraceAD 是 PU 学习半监督；wzc 用全监督 + Tree-LSTM" |

### 3.2 指标 3.4（轨迹+拓扑融合）— 仍是消融叙事

TADBench 7 个 baseline **全部 trace-only**——
- ❌ 不是 TraTopoRca 的同维度对手
- ✅ 但作为消融基线效果反而更强：你可以说"我们的方法在 7 个 SOTA trace-only baseline 之外，**新增了 host metric + 拓扑通道**，得到 X% 增量"——这个叙事比之前更有说服力。

### 3.3 公开数据集补充（论文 Table I + II）

| Dataset | Avg Trace Depth | Avg Span | Granularity | 推荐用法 |
|---|---|---|---|---|
| TrainTicket | 3.3 | 39.0 | Operation | 跑 wzc trace_sv_diag CLS 对照（PUTraceAD 95.4%、TraceVAE 97.3%、GTrace 99.4%）|
| GAIA | 4.7 | 9.3 | Service | 浅 trace，TraceVAE 强（90.9%） |
| **AIOps2020** | 5.5 | 23.1 | Service | ★ 最小（74MB），优先尝试 |
| AIOps2022 | 4.2 | 21.7 | Operation | TraceVAE 78.9% |
| AIOps2023 | 3.8 | 14.5 | Service | **README 提到但实际不在仓库**——仓库下载或问作者 |

### 3.4 数据格式（论文 Fig. 9 + 数据契约）

TADBench 的 Trace + Span 标准格式：
```
Trace: { trace_id, root_span, span_count,
         anomaly_type ∈ {0,1,2,3}, source }

Span:  { trace_id, span_id, parent_span_id, children_span_id,
         start_time, duration,
         service_name, operation_name,
         anomaly ∈ {0,1}, status_code,
         latency ∈ {0,1},     // 1 = latency anomaly
         structure ∈ {0,1},   // 1 = structural anomaly
         extra }
```

**与你们 16 列 CSV 的核心差异**：
- 多了 `anomaly_type` 4 类标注（normal / latency / structure / both）→ 适合 wzc trace_sv_diag CLS 头任务
- 多了 `latency` 和 `structure` 两个独立字段 → 适合linyu TrUST 的双子网络（结构 + 时序分别建模）
- 没有 fault_type 细类（你们的"network delay / cpu stress / dns error"等），只有粗粒度二分类
- 没有 fault_instance 字段（不能直接做 RCA 评测）

**结论**：TADBench 的 schema 更适合做 AD 任务，做 RCA 和细类 CLS 时需要补充字段。

---

## 4. 整合路径（按 ROI 排序，更新版）

### 🟢 高 ROI 短期可做（1~3 周）

#### 路径 A：补论文叙事（半天）

- 在中期/结题报告 3.3 节"已有方法对比"小节，把 TADBench 论文 Table III + Fig. 18 决策树纳入作为对比 baseline 标准。
- 把 TrUST 定位为 *"对 [TADBench] 提出的 future direction 的具体实现"*。
- 引用作者：Yongqian Sun, Shenglin Zhang (南开), Changhua Pei (CNIC), **Dan Pei (清华，本课题 PI)** —— 师门内部对照。

#### 路径 B：在 AIOps2020 上跑 TrUST + trace_sv_diag（1~2 天）

- 数据：解压 `Datasets/AIOps2020.tar.gz`（74MB）。
- 流程：
  1. 写 `trace_tools/dataprocess/scripts/tadbench_adapter.py`：TADBench Trace/Span ↔ 你们 16 列 CSV 双向转换。
  2. linyu TrUST 跑 → 得到 F1 vs TADBench 表 III 的 71.8%（GTrace 在 AIOps2020 上的 SOTA）。
  3. wzc trace_sv_diag 跑 → 得到 Acc vs TADBench 表 III 的 81.9%（GTrace ACC）。
- 输出：`results/tadbench_aiops2020_comparison.md`。

#### 路径 C：实现 PUTraceAD 的子任务对照（结题前必做）

- 论文决策树（§E）显示：你们故障注入数据异常比 ~10%+ 时**应该用 PUTraceAD**。
- 在你们自己测试集上跑 PUTraceAD（CCF AIOps + 阿里云原生）。
- 与 trace_sv_diag CLS 直接对照 P/R/F1。
- 工程量：1 周（PUTraceAD 用 PyG + transformers，环境与你们兼容）。

### 🟡 中 ROI 中期可做（结题前 1 个月）

#### 路径 D：跑 TraceVAE 在你们测试集（论文打榜素材）

- TraceVAE 在 GAIA / AIOps2022 上是 SOTA（90.9% / 78.9%），训练慢但是论文最强对手。
- 工程量：1~2 周（DGL 环境与你们一致，但训练 6.5 小时/数据集要预留 GPU 时间）。

#### 路径 E："拓扑+主机指标" 增量贡献叙事（指标 3.4）

- TraTopoRca 关 Host 通道 → 与 GTrace 比（同样三通道但无 host）。
- 用差距证明 host metric + 拓扑的增量。
- 工程量：1 周（模型分支已有，主要是实验 + 报告）。

### 🔴 低 ROI 不建议短期做

- ❌ 完整 TADBench SDK 对接（SDK 是空壳，不维护）
- ❌ 替换内部 schema 为 Trace/Span dataclass（赛西已认证，不动）
- ❌ 跑 TraceCRL / Multimodal LSTM / TraceAnomaly / CRISP（决策树都不推荐这些 baseline 在你们的异常比下）

---

## 5. 推荐的落地清单（更新版）

### 5.1 短期（1~3 周）—— 重点是论文叙事 + AIOps2020 对照

| # | 任务 | 工程量 | 输出 |
|---|---|---|---|
| 1 | 写 `trace_tools/dataprocess/scripts/tadbench_adapter.py` | 半天 | TADBench↔16 列 CSV 转换器 |
| 2 | 解压 AIOps2020 + 跑 TrUST → AD F1 | 1 天 | 数据对照表 |
| 3 | 同上跑 wzc trace_sv_diag → CLS Acc | 1 天 | 同上 |
| 4 | 跑 PUTraceAD 在你们测试集（决策树推荐对手） | 1 周 | `results/putracead_vs_ours.md` |
| 5 | 写 `results/tadbench_summary.md`，引用论文表 III 决策树 + 你们对照数字 | 半天 | 结题素材 |
| 6 | 中期/结题报告 3.3 节加 TrUST 论文谱系叙事 | 半天 | 论文 framing |

### 5.2 中期（结题前 1 个月）

| # | 任务 | 工程量 |
|---|---|---|
| 7 | 跑 TraceVAE 在你们测试集 | 1~2 周（GPU 时间） |
| 8 | 关 Host 通道的 TraTopoRca vs GTrace 消融 | 1 周 |
| 9 | 写论文 / 结题报告对比章节 | 1 周 |

### 5.3 不做

- 完整 TADBench SDK 对接、schema 替换、AIOps2023 数据（仓库没有）。

---

## 6. 关键引用清单（直接可复制到论文/结题报告）

### 6.1 BibTeX（建议格式，等论文 DOI 出来后补全）

```bibtex
@article{sun2025tadbench,
  title={A Comprehensive Benchmark and Empirical Study of Trace Anomaly Detection},
  author={Sun, Yongqian and Shao, Minyi and Nie, Xiaohui and Yang, Kaiwen and
          Li, Xingda and Hao, Bowen and Zhang, Shenglin and Pei, Changhua and
          He, Dongbiao and Li, Yanbiao and Pei, Dan},
  journal={IEEE Transactions on Services Computing},
  year={2025},
  note={Code: https://github.com/nkalgo/TADBench.git}
}
```

### 6.2 论文叙事关键句（中文，给linyu用）

> 近期，Sun 等人 [TADBench] 系统评测了七种主流 trace 异常检测算法（TraceAnomaly、CRISP、TraceVAE、GTrace、PUTraceAD、TraceCRL、Multimodal LSTM），并指出未来方向应将 TraceVAE 的 entropy 减小策略（Bernoulli-Categorical Scaling、Node Count Normalization、Gaussian Std-Limit）与 GTrace 的层次化图编码架构协同集成。本工作（TrUST）沿这一方向开展进一步探索，将上述评分优化策略与层次化图编码架构在统一框架内协同建模……

（注：TADBench 论文与 TrUST 同期独立开展，本叙事强调研究方向的一致性而非依赖关系。）

### 6.3 决策树推荐对照表（给 wzc 选 CLS 对手用）

| 你们的数据集 | 异常比 | 决策树推荐 |
|---|---|---|
| 阿里云原生 12000 trace（YS02-013）| ~10% | 临界——PUTraceAD 或 GTrace 都跑 |
| 阿里云原生 6229 trace（YS02-009）| ~70%+ | PUTraceAD（≥10% 必选）|
| CCF AIOps 30000 trace（中期报告）| ~36% | PUTraceAD |
| AIOps2020（论文）| 21.8% | PUTraceAD |

---

## 7. 风险与注意

1. **AIOps2023 缺失**：论文 Table III 列了 AIOps2023 的数字，但你的本地仓库 `Datasets/` 目录下没有这个数据。需要从 https://github.com/nkalgo/TADBench.git 完整 clone 或问作者要——3.6GB 总体积。
2. **TraceAnomaly / CRISP 用 TF1**：Python 3.6 + 老 TensorFlow，与你们 PyTorch + DGL 2.2.1 + Python 3.10 环境冲突。决策树都不推荐这两个 baseline，可以放弃。
3. **TraceVAE 训练慢**：6.5 小时/数据集，GPU 时间要提前预约。
4. **TADBench 不评 RCA**：linyu TrUST 的 RCA 部分（柔性聚合 Top-K）在 TADBench 里没有直接对手，需要自找 baseline（OpsAug 家族）。
5. **GTrace 你们已经有了**：直接复用 `GTrace-LateFusion/`，不要在 TADBench 那份里重新实现。
6. **TADBench schema 没有 fault_type 细类**：你们 trace_svnd_diag 的 11 类细分类无法直接在 TADBench 数据集上评测——只能评 binary AD 或 4 类 anomaly_type。
