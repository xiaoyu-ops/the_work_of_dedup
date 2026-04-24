# MMdedup 论文修改计划

**基于**：VLDB 2026 审稿意见（R1 Weak Reject / R2 Reject / R3 Weak Reject）
**目标投稿**：AI 相关会议（NeurIPS / ACM MM / ICML 优先）
**已确认路线**：✅ 路线 A — 补充 MLLM 下游训练实验 + Q-SemDeDup 统一框架扩展
**核心创新重定位**：从"系统集成"升级为"提出统一质量感知多模态去重框架"
**文档状态**：草稿，分工待填写

---

## 一、目标会议

> AI 会议审稿人最关心：**去重后模型性能提升了多少**、**方法是否有 novelty**。
> VLDB 的工程细节批评（MinHash 过时、参数 hardcode）在 AI 会议中大部分不构成问题。

| 目标 | 截稿参考 | 契合度 | 说明 |
|---|---|---|---|
| **NeurIPS 2026**（Datasets & Benchmarks Track） | ~2026年5月 | ⭐⭐⭐⭐⭐ | 最对口。D&B Track 专收数据处理/质量工作，强调下游模型效果 |
| **ACM MM 2026** | ~2026年4月 | ⭐⭐⭐⭐ | 多媒体方向最自然对齐，竞争压力比 NeurIPS 小，可作保底 |
| **ICML 2027**（Datasets Track） | ~2027年1月 | ⭐⭐⭐⭐ | 与 NeurIPS 类似，竞争稍弱 |
| **ICLR 2027** | ~2026年10月 | ⭐⭐⭐ | 审稿口味偏理论，需下游实验结果足够强 |
| **AAAI 2027** | ~2026年8月 | ⭐⭐⭐ | 综合 AI，系统类工作有受众，影响力稍低 |

**推荐顺序**：NeurIPS 2026 D&B → ACM MM 2026 → ICML 2027 → ICLR 2027

### VLDB vs AI 会议审稿差异

| 关注点 | VLDB（已拒） | AI 会议（目标） |
|---|---|---|
| 下游模型效果 | 未要求 | **核心必须项** |
| 算法 novelty | 严格（MinHash 被批） | 统一框架即可 |
| 系统工程贡献 | 核心 | 次要 |
| 数据集真实性 | 严格 | 严格 |
| Related Work | 需引用 VLDB 数据库工作 | 引用 AI/CV/NLP 工作即可 |

---

## 二、核心创新重定位

### 从"系统集成"到"统一质量感知去重框架"

图像端已实现 Q-SemDeDup，核心公式：

```
Score = α · Sim(x, C) + (1-α) · Norm(Quality(x))     [α = 0.7]
```

**新定位**：将此框架统一推广至文本和音频，形成三模态一致的技术贡献。

| 模态 | Embedding（Sim） | Quality 指标 | 状态 |
|---|---|---|---|
| 图像 | CLIP-ViT-B-16 | 文件大小 / 分辨率 | ✅ 已实现 |
| 文本 | SBERT 语义相似度 | 文本信息熵 / 困惑度 | ❌ 待实现 |
| 音频 | CLAP embedding | 信噪比 / 有效时长 | ❌ 待实现 |

**论文包装**：
- Section 3 标题改为 "Unified Quality-Aware Multimodal Deduplication Framework"
- Q-SemDeDup 公式作为全文核心，三个模态各自实例化
- Contribution list 新增："We propose a unified quality-aware deduplication framework extending Q-SemDeDup across image, text, and audio modalities"

---

## 三、修改任务清单

### 🔴 P0 — 必须完成，缺一不可

#### 1. MLLM 下游训练实验（最核心）

AI 会议的生死线：没有下游实验等于没有论文。

**实验设计**：
- 基础模型：LLaVA-1.5-7B（LoRA 方案可压至 24GB 显存）
- 数据集：CC3M 子集（~300K 图文对，天然含重复）
- 对比组：
  1. 原始数据训练（baseline）
  2. MMdedup 全模态去重后训练
  3. 仅图像去重（消融）
  4. 仅文本去重（消融）
- 评测：VQAv2、TextVQA（优先）；MMMU（加分）
- 资源：A100 40G+ × 1，约 1-2 周

**步骤**：
- [ ] 下载 CC3M 子集，转换为 LLaVA 训练格式
- [ ] 用 MMdedup 对 CC3M 执行去重，记录各模态去重率
- [ ] 跑 4 组 LLaVA fine-tune
- [ ] 在 VQAv2 / TextVQA 上评测，整理结果表格
- [ ] 写 Section 6（下游实验）

#### 2. 真实数据集替换合成数据

**问题**：合成数据"自问自答"在 AI 会议同样不被接受（R2 D4）

- [ ] 用 CC3M / LAION 子集替换现有合成评测集（天然含重复，有 ground truth 可挖）
- [ ] 或：从爬取数据中人工标注 1000 对，建立小规模真实 ground truth

#### 3. 文本去重升级（Q-SemDeDup 文本实例化）

- [ ] 实现两阶段流程：MinHash LSH 粗过滤 → SBERT 精排
- [ ] Quality 指标：信息熵（高熵优先保留）
- [ ] 补充效率对比实验：SBERT vs MinHash，论证 TB 级场景下两阶段的必要性
- [ ] 新增依赖：`sentence-transformers>=2.6`、`faiss-cpu>=1.7`

#### 4. 音频去重升级（Q-SemDeDup 音频实例化）

- [ ] 替换 embedding：频谱指纹 → CLAP（Microsoft CLAP `msclap` 或 LAION CLAP）
- [ ] Quality 指标：信噪比（SNR）
- [ ] 对比实验：CLAP vs 原 spectrum fingerprint
- [ ] 新增依赖：`msclap>=1.3`

#### 5. 数字/图表一致性

**问题**：摘要、Introduction、Table 数字矛盾（R1 最强调，AI 会议同样不容忍）

- [ ] 建立数字溯源表：所有关键数字标注来源实验
- [ ] 逐条核对 Introduction 贡献声明与实验小节对应关系
- [ ] Table 5/6/7 加粗逻辑统一，caption 标注"粗体表示最优值"
- [ ] 补充 F1 到 Table 5/6/7

#### 6. 补充 SSCD baseline

**问题**：SimCLR 不是图像近重复检测的合适 baseline，SSCD（CVPR 2022）更对口

- [ ] 加载 SSCD 公开权重（Meta Research）
- [ ] 在测试集上跑 SSCD 对比，加入 Table 5

#### 7. 处理本文不占优的实验结果

**问题**：Table 5 中 pHash/SemDeDup 优于本文，正文没有讨论（AI 会议审稿人同样会质疑）

- [ ] Table 5：pHash 在多项指标优于本文 → 正文分析：pHash 只适合完全重复，近重复场景失效
- [ ] Table 6：MD5 F-Measure 高于 SemDeDup → 说明 MD5 应用边界
- [ ] Table 7：MD5 速度优势 → 同上

---

### 🟡 P1 — 重要，显著影响审稿人信心

#### 8. Related Work 重写（面向 AI 会议）

**VLDB 版 Related Work 需要大幅调整**：删除数据库领域专用引用，换成 AI/CV/NLP 领域工作。

- [ ] 压缩到 1 页以内
- [ ] 删除或淡化 entity resolution、PPJoin 等数据库领域工作
- [ ] 新增 AI 领域相关工作：
  - 数据质量对 LLM/MLLM 训练的影响（DataComp、LAION-5B、The Pile 等）
  - 语义去重方法（SemDeDup 2023、D4 2023）
  - 多模态数据清洗（FairDeDup）
- [ ] 每个子节末尾加"MMdedup 与之的区别"

#### 9. 修改 Section 2.6 自我否定表述

- [ ] 删除"Rather than proposing novel algorithms..."
- [ ] 改为：介绍统一框架如何解决单模态方案无法处理的挑战

#### 10. 添加正式问题定义

- [ ] Section 3 开头加 Problem Definition（含数学符号）
  - 输入：多模态语料 D，含模态标签
  - 输出：去重后子集 D' ⊆ D
  - 目标：保留语义多样性的同时最小化 |D'|，最大化下游模型性能

#### 11. FairDeDup 讨论

- [ ] 阅读 FairDeDup，整理与 MMdedup 核心差异
- [ ] Related Work 新增对比段落，修改 Introduction 中的 novelty claim

---

### 🟢 P2 — 写作细节，锦上添花

- [ ] Section 1 动机段补充引用（数据质量影响模型性能的 AI 文献 2-3 篇）
- [ ] 删除 LSHBloom "SOTA" 表述，或替换为同行评审引用
- [ ] 224×224 参数加说明：引用 CLIP 原文
- [ ] 补充 cluster 数量消融实验（500/1000/2000/5000 对 recall/速度的影响）

---

## 四、修改优先级总览

```
立即启动（阻塞后续工作）
├── CC3M 数据准备
├── 文本 SBERT 升级实现
└── 音频 CLAP 升级实现

并行推进
├── LLaVA fine-tune 实验（依赖数据准备）
├── SSCD baseline 实验
├── FairDeDup 阅读 + Related Work 重写
└── 数字溯源表 + 图表修正

最后收尾
├── 论文本体写作修改
├── Section 3 统一框架重写
└── P2 写作细节
```

---

## 五、分工（待填写）

| 任务 | 负责人 | 预计完成时间 |
|---|---|---|
| CC3M 数据下载 + 格式转换 | | |
| LLaVA fine-tune 实验（4 组） | | |
| VQAv2 / TextVQA 评测 | | |
| 文本 SBERT + 两阶段流程实现 | | |
| 音频 CLAP embedding 实现 | | |
| SSCD baseline 实验 | | |
| FairDeDup 阅读 + Related Work 重写 | | |
| 数字溯源表 + 图表修正 | | |
| Section 3 统一框架重写 | | |
| 论文本体写作修改 | | |
