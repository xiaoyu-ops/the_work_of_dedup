# Experiment Summary — the_compare_image

路径与数据集
- 原始训练数据（磁盘路径）：
  - `D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated\train`
- 结果 / keep-lists 存放：
  - `D:\Deduplication_framework\2026_new_experiment\result`
- 数据读取方式：
  - 默认使用 `DedupJsonDataset` 从文件系统递归扫描图片（支持 JSON keep-list 文件以只保留特定样本）。
  - 支持可选的 WebDataset sharded 输入（由变量 `USE_WDS` 控制），shard 根目录为 `D:\Deduplication_framework_wds_shards`。

训练与评估设置
- 训练脚本：`compare_dedup_accuracy.py`
- 比较方法（当前默认启用）:
  - `Ours (Pipeline)`（使用 `our_pipeline_keep_list.json`，其他方法在脚本中被注释/可选）

关键超参数
- `BATCH_SIZE`：512
- `EPOCHS`：4
- `NUM_CLASSES`：1000
- `LR`（学习率）：1e-4
- `VAL_SPLIT`：0.2（固定随机种子 `SPLIT_SEED=42` 做稳定划分）
- 设备：自动选择 `cuda`（若可用）否则 `cpu`

数据加载与加速
- 数据增强 / 预处理：
  - Resize 到 (224,224)
  - ToTensor + ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- DataLoader（文件系统模式）:
  - `num_workers`: 24
  - `pin_memory`: True
  - `persistent_workers`: True
  - `prefetch_factor`: 4
  - `collate_fn`: `safe_collate`（会过滤掉返回 None 的样本）
- WebDataset 模式（若启用）:
  - `WDS_NUM_WORKERS`: 0 on Windows, otherwise 8
  - `WDS_SHUFFLE`: 1000

模型与优化器
- 基模型：`torchvision.models.resnet18`，加载 `ResNet18_Weights.IMAGENET1K_V1` 预训练权重
- 分类头替换：`model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)`
- 全网络可训练（脚本中没有冻结 backbone），即 `optimizer` 更新 `model.parameters()`
- 优化器：`optim.Adam(model.parameters(), lr=LR)`（未设置 `weight_decay`）
- 损失：`nn.CrossEntropyLoss()`
- AMP：使用 `torch.cuda.amp.GradScaler` 在 GPU 上启用混合精度

训练行为与度量
- 训练模式为“unfrozen / fine-tuning”（脚本注释表明全部层参与训练）
- 每 epoch 在训练后对验证集评估并打印 `val_acc`，脚本内部维护 `best_val`
- 最终返回 `(train_acc, time_s, dataset_size, best_val)` 并把 summary 保存到 `final_ours_full_run.csv`

注意事项与建议
- 当前配置批量较大（512）且所有层均被训练，容易导致过拟合；若出现训练 acc >> val acc，建议：
  - 先冻结 backbone，只训练 `fc`，再逐步解冻并微调；
  - 添加 `weight_decay`（例如 1e-4）或降低 `LR`；
  - 减小 `BATCH_SIZE` 或增加数据增强。 
- 脚本不自动保存训练 checkpoint（默认无 `.pth` 输出），建议在训练循环中保存 `model.state_dict()` 以便后续单独评估。

运行命令（示例）
```powershell
# 运行对比训练（使用默认方法列表）
python compare_dedup_accuracy.py

# 若使用 WebDataset，请先在脚本中将 USE_WDS=True 并准备 shards，然后运行
python compare_dedup_accuracy.py
```

文件位置
- 本文件：`the_compare_image/SUMMARY.md`
- 训练脚本：`the_compare_image/compare_dedup_accuracy.py`

生成时间： 2026-02-18
