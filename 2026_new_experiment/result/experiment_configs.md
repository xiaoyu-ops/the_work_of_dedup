# 图像去重实验配置记录
日期: 2026-01-27
环境: Windows / NVIDIA GPU (如果可用)
数据集: imagenet_bloated (10,000 样本子集)

## 1. 基线方法 (Baselines)

### MD5 哈希 (MD5 Hash)
- **方法**: 精确二进制匹配 (hashlib.md5)
- **参数**: 无 (严格相等)
- **备注**: 仅检测完全一致的文件副本，无语义理解能力。

### 感知哈希 (pHash)
- **方法**: imagehash.phash
- **阈值**: 汉明距离 (Hamming Distance) <= 5
- **库**: `imagehash`
- **备注**: 对轻微缩放/压缩有抵抗力，但对旋转/裁剪抵抗力较弱。

### SimCLR (Embeddings)
- **模型**: ResNet50 (自监督/SimCLR 预训练)
- **相似度度量**: 余弦相似度 (Cosine Similarity)
- **阈值**: 0.93
- **图像大小**: 256x256 (Resized)
- **批次大小**: 32

---

## 2. 我的流水线 (My Pipeline - Ours)

### 核心配置
- **算法**: SemDeDup (语义去重)
- **Embedding 后端**: `open_clip`
- **模型架构**: `ViT-B-16` (laion2B-s34B-b88K)
- **设备**: 自动 (CUDA/CPU)

### 关键参数 (2026-01-27 调优)
- **Epsilon (eps)**: `0.07` 
  - 对应相似度阈值: `1 - 0.07 = 0.93`
- **批次大小**: `256`
- **最大候选数**: `50000`

### 执行上下文
- **配置文件**: `configs/my_pipeline.yaml` & `configs/image_config.yaml`
- **输入数据**: 10,000 张图像 (子集清单)
- **结果**: 
  - 准确率 (Precision): 85.24%
  - 召回率 (Recall): 69.25%
  - 吞吐量 (Throughput): 86.8 imgs/s

---

## 3. 实验数据汇总表格

| Method | Throughput (imgs/s) | Precision | Recall | GPU Mem (GB) |
| :--- | :---: | :---: | :---: | :---: |
| MD5 Hash | 2640.2 | 100.00% | 0.00% | 0 |
| pHash | 356.2 | 100.00% | 99.97% | 0 |
| SimCLR Embed | 45.0 | 18.26% | 99.68% | 0.00 |
| My Pipeline | 86.8 | 85.24% | 69.25% | 0.00 |
