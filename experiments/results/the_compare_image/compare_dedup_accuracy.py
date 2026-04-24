import os
import json
import re
import time
import csv
import math
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from torch.utils.data.dataloader import default_collate
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights 
from PIL import Image
from tqdm import tqdm
try:
    import webdataset as wds
except Exception:
    wds = None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

# ================= 1. 实验配置 =================
RAW_DATA_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated\train"
JSON_RESULT_DIR = r"D:\Deduplication_framework\2026_new_experiment\result" 

METHODS_TO_COMPARE = [
    # ("No Dedup",       None),
    # ("MD5",            "md5_keep_list.json"),
    # ("pHash",          "phash_keep_list.json"),
    # ("SimCLR",         "simclr_keep_list.json"),
    # ("SemDeDup",       "semdedup_keep_list.json"),
    ("Ours (Pipeline)", "our_pipeline_keep_list.json") 
]

BATCH_SIZE = 512
EPOCHS = 4

NUM_CLASSES = 1000 
LR = 0.0001         
VAL_SPLIT = 0.2
SPLIT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_FILENAME = "cached_no_dedup_list.json"
# 加速设置
torch.backends.cudnn.benchmark = True
# WebDataset 设置
USE_WDS = False
WDS_ROOT = r"D:\Deduplication_framework_wds_shards"
WDS_SHUFFLE = 1000
WDS_NUM_WORKERS = 0 if os.name == "nt" else 8
# ===============================================

class DedupJsonDataset(Dataset):
    def __init__(self, root_dir, json_path=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        cache_path = os.path.join(JSON_RESULT_DIR, CACHE_FILENAME)
        
        if json_path:
            full_json_path = os.path.join(JSON_RESULT_DIR, json_path)
            print(f"   [Load] Reading Keep-List from: {os.path.basename(full_json_path)}...")
            if os.path.exists(full_json_path):
                try:
                    with open(full_json_path, 'r') as f:
                        self.image_paths = json.load(f)
                except Exception as e:
                    print(f"   [ERROR] Failed to load JSON: {e}")
            else:
                print(f"   [ERROR] File not found: {full_json_path}")
        else:
            if os.path.exists(cache_path):
                print(f"   [Cache] Loading from {CACHE_FILENAME}...")
                with open(cache_path, 'r') as f: self.image_paths = json.load(f)
            elif os.path.exists(root_dir):
                print(f"   [Scan] Scanning directory recursively...")
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            self.image_paths.append(os.path.join(root, file))
                with open(cache_path, 'w') as f: json.dump(self.image_paths, f)
        
        print(f"   [Ready] Dataset size: {len(self.image_paths)} images")

    def _extract_label(self, filepath):
        # 正确标签来自父目录名（0-999）
        try:
            parent = os.path.basename(os.path.dirname(filepath))
            if parent.isdigit():
                return int(parent)
        except:
            pass
        # 兜底：兼容旧的文件名规则
        try:
            filename = os.path.basename(filepath)
            match = re.search(r'_(\d+)(?:_aug|\.)', filename)
            if match:
                return int(match.group(1)) % NUM_CLASSES
        except:
            pass
        return 0

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        try:
            path = self.image_paths[index]
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, self._extract_label(path)
        except:
            return None

def run_training(method_name, json_file):
    print(f"\n" + "="*50)
    print(f"START TRAINING: {method_name} (UNFROZEN / FINE-TUNING)")
    print("="*50)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    def get_wds_dir(json_path):
        name = "no_dedup" if not json_path else os.path.splitext(json_path)[0]
        return os.path.join(WDS_ROOT, name)

    def build_wds_loader(json_path):
        if not USE_WDS or wds is None:
            return None, None
        wds_dir = get_wds_dir(json_path)
        shard_fs_pattern = os.path.join(wds_dir, "shard-*.tar")
        shard_list = sorted(glob.glob(shard_fs_pattern))
        if not shard_list:
            return None, None
        shard_urls = ["file:" + p.replace("\\", "/") for p in shard_list]

        count = None
        index_path = os.path.join(wds_dir, "index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    count = json.load(f).get("count")
            except Exception:
                count = None

        dataset = (
            wds.WebDataset(shard_urls, handler=wds.warn_and_continue, shardshuffle=False, empty_check=False)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(transform, lambda x: int(x))
        )
        if WDS_SHUFFLE and WDS_SHUFFLE > 0:
            dataset = dataset.shuffle(WDS_SHUFFLE)
        if count:
            dataset = dataset.with_length(count)
        dataset = dataset.batched(BATCH_SIZE, partial=False)
        if count:
            dataset = dataset.with_length(math.ceil(count / BATCH_SIZE))

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=WDS_NUM_WORKERS,
            pin_memory=True,
            persistent_workers=(WDS_NUM_WORKERS > 0),
            prefetch_factor=4 if WDS_NUM_WORKERS > 0 else None
        )
        return loader, count

    train_loader, dataset_size = build_wds_loader(json_file)
    val_loader = None
    if train_loader is None:
        dataset = DedupJsonDataset(RAW_DATA_ROOT, json_file, transform)
        if len(dataset) == 0:
            return None
        dataset_size = len(dataset)

        # 8:2 划分
        rng = torch.Generator().manual_seed(SPLIT_SEED)
        indices = torch.randperm(dataset_size, generator=rng).tolist()
        split_idx = int(dataset_size * (1 - VAL_SPLIT))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        # Windows IO 设置
        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=24,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=safe_collate
        )
        val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=safe_collate
        )

    print("   [Model] Loading Pretrained Weights (Unfrozen)...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # 【关键点3】这里没有任何冻结代码！让模型自由学习！
    # (之前的 param.requires_grad = False 已被移除)
    
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    
    # 【关键点4】优化器负责更新 model.parameters() (所有层)，不仅仅是 fc 层
    optimizer = optim.Adam(model.parameters(), lr=LR)
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    model.train()
    start_time = time.time()
    best_val = 0.0

    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

        # 验证
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    v_inputs, v_labels = batch
                    v_inputs = v_inputs.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                    v_labels = v_labels.to(DEVICE, non_blocking=True)
                    v_outputs = model(v_inputs)
                    _, v_pred = torch.max(v_outputs.data, 1)
                    val_total += v_labels.size(0)
                    val_correct += (v_pred == v_labels).sum().item()
            if val_total > 0:
                val_acc = 100 * val_correct / val_total
                best_val = max(best_val, val_acc)
                print(f"   [Val] Epoch {epoch + 1}: {val_acc:.2f}%")
            model.train()

    real_time = time.time() - start_time
    if total == 0:
        print(f"[ERROR] {method_name}: No samples were read. Please check WDS shards/path.")
        return None
    final_acc = 100 * correct / total
    
    print(f"DONE: {method_name}")
    print(f"   -> Time Cost:  {real_time:.1f}s ({(real_time/3600):.2f} hours)")
    print(f"   -> Train Acc:  {final_acc:.2f}%")
    if val_loader is not None:
        print(f"   -> Best Val:   {best_val:.2f}%")
    
    return final_acc, real_time, dataset_size, best_val

if __name__ == "__main__":
    results = []
    print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | LR: {LR} | Mode: Unfrozen (Fair)")
    
    for name, json_file in METHODS_TO_COMPARE:
        if json_file and not os.path.exists(os.path.join(JSON_RESULT_DIR, json_file)):
             print(f"[SKIP] {name}: File not found")
             continue
        
        res = run_training(name, json_file)
        if res:
            acc, time_s, size, best_val = res
            results.append({
                "Method": name,
                "Size": size,
                "Time(s)": f"{time_s:.1f}",
                "Train Acc": f"{acc:.2f}%",
                "Best Val": f"{best_val:.2f}%" if best_val else ""
            })

    print("\n" + "="*60)
    print(f"{'Method':<15} | {'Size':<10} | {'Time(s)':<15} | {'Train Acc':<10} | {'Best Val':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['Method']:<15} | {r['Size']:<10} | {r['Time(s)']:<15} | {r['Train Acc']:<10} | {r['Best Val']:<10}")
    print("="*60)
    
    try:
        with open("final_ours_full_run.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Method", "Size", "Time(s)", "Train Acc", "Best Val"])
            writer.writeheader()
            writer.writerows(results)
        print("\n[Success] Results saved to final_ours_full_run.csv")
    except: pass