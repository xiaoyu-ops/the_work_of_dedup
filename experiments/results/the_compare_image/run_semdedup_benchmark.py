import os
import sys
import random
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

# ================= 配置区域 =================
# 指向 imagenet_bloated 文件夹
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 256
# SemDeDup 论文推荐 epsilon=0.07 对应阈值 0.93
SIMILARITY_THRESHOLD = 0.93 
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
# ===========================================

def get_all_images(root_dir, limit=None):
    image_files = []
    # 使用 topdown=True 可以修改 dirs 列表，从而打乱遍历顺序，实现随机采样
    count = 0
    for root, dirs, files in os.walk(root_dir, topdown=True):
        random.shuffle(dirs)  # 打乱子文件夹遍历顺序
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                count += 1
                if count % 1000 == 0:
                    print(f"      [Scan] Found {count} images...", end='\r')
                if limit and len(image_files) >= limit:
                    print(f"      [Scan] Reached limit of {limit} images.")
                    return image_files
    return image_files

def parse_id(filename):
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result(method, throughput, precision, recall, gpu_mem):
    file_exists = os.path.isfile(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
        writer.writerow([method, f"{throughput:.1f}", f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{gpu_mem:.2f}"])
    print(f"[成功] {method} 结果已写入文件")

class BenchDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            return self.transform(img), self.files[idx]
        except:
            return torch.zeros(3,224,224), self.files[idx]

def run_semdedup_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] SemDeDup (ResNet50) 评测...")
    
    # 1. 准备数据
    print(f"   正在快速扫描前 {SAMPLE_SIZE} 个文件...")
    all_files = get_all_images(IMAGE_DIR, limit=SAMPLE_SIZE)
    if not all_files: return
    # 截取前 SAMPLE_SIZE 个样本 (其实已经是了，但保持逻辑)
    test_files = all_files[:SAMPLE_SIZE]
    
    # 2. 加载模型 (ResNet50 Feature Extractor)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1]) 
    model.to(device).eval()

    dataset = BenchDataset(test_files)
    # 增加 num_workers 提高加载速度，Windows 下建议 4 或 8
    # 优化：针对 24G 显存 GPU，增加 batch_size 和 num_workers 以提升利用率
    # 注意：为了避免 Windows 启动 16 个进程导致初始化过久看似卡死，改为 8
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    feats_list = []
    paths_list = []
    
    # 开始计时 (包含特征提取 + SemDeDup 算法计算)
    start_time = time.time()
    
    # A. 特征提取
    print("   正在提取特征 (Initializing DataLoader for batch processing)...")
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Extracting", file=sys.stdout):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze()
            # 关键：SemDeDup 需要 L2 归一化
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_list.append(feats.cpu().numpy())
            paths_list.extend(paths)
            
    embeddings = np.concatenate(feats_list)
    
    # B. 运行 SemDeDup 核心算法
    print("   正在执行 SemDeDup K-Means 聚类筛选...")
    
    # 使用 K-Means 进行聚类
    # 聚类数量 heuristic: 假设平均每个簇有 50 张图片 (根据你的数据分布可调整)
    # 对于 10000 张图，大约 200 个簇。这比按文件夹(类别)聚类更接近真实无监督场景。
    n_clusters = max(1, len(embeddings) // 50)
    print(f"      Running KMeans with k={n_clusters} ...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # 将 embedding 和 path 按聚类标签分组
    cluster_groups = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_groups[label].append(i) # 存索引
        
    removed_indices = set()
    
    for label, indices in tqdm(cluster_groups.items(), desc="Filtering Clusters", file=sys.stdout):
        if len(indices) < 2: continue
        
        # 获取当前簇的特征 [K, 2048]
        cluster_feats = embeddings[indices]
        
        # 1. 计算中心 (Centroid)
        centroid = np.mean(cluster_feats, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # 2. 计算到中心的相似度并降序排列
        sim_to_center = np.dot(cluster_feats, centroid)
        # argsort是从小到大，[::-1]反转为从大到小
        sorted_local_indices = np.argsort(sim_to_center)[::-1]
        
        # 3. 动态去重筛选
        kept_local_feats = []
        
        for local_idx in sorted_local_indices:
            global_idx = indices[local_idx]
            current_feat = cluster_feats[local_idx]
            
            if not kept_local_feats:
                # 离中心最近的，无条件保留
                kept_local_feats.append(current_feat)
            else:
                # 和已保留的对比
                # [M, 2048] @ [2048, 1] -> [M]
                sims = np.dot(np.array(kept_local_feats), current_feat)
                max_sim = np.max(sims)
                
                if max_sim < SIMILARITY_THRESHOLD:
                    # 不相似，保留
                    kept_local_feats.append(current_feat)
                else:
                    # 太相似，删除
                    removed_indices.add(global_idx)

    # 算法结束，停止计时
    total_time = time.time() - start_time
    throughput = len(test_files) / total_time
    
    # C. 统计 Precision/Recall
    print("   正在计算准确率指标...")
    
    # 1. 构建 Ground Truth
    # 统计每个 ID 在样本中出现的总次数
    id_list = [parse_id(os.path.basename(p)) for p in paths_list]
    id_counts = Counter(id_list)
    
    # 总共应该删除的数量 (Ground Truth Duplicates)
    # 如果一个 ID 有 N 个，应该删除 N-1 个
    total_gt_duplicates = sum([count - 1 for count in id_counts.values()])
    
    # 2. 统计删除是否正确 (修复版逻辑)
    removed_counts = Counter() # 记录每个 ID 被删除了多少次
    for idx in removed_indices:
        fid = id_list[idx]
        removed_counts[fid] += 1
        
    tp = 0
    fp = 0 # 包含了误删唯一图片 + 过度删除副本
    
    # 遍历所有出现过的 ID
    for fid, total_count in id_counts.items():
        # 这个 ID 实际被删了多少个
        rem_count = removed_counts[fid]
        
        # 它可以被删除的最大数量 (即副本数，保留1个)
        # 如果 total_count=1, max_removable=0
        max_removable = max(0, total_count - 1)
        
        if rem_count <= max_removable:
            # 删的数量 <= 副本数量，说明留住了至少一个，全是 TP
            tp += rem_count
        else:
            # 删多了 (过度删除，或者本来就只有一个被删了)
            # 有效部分是 max_removable (可能为0)
            tp += max_removable
            # 多删的部分是 FP (把本该保留的“独苗”也删了)
            fp += (rem_count - max_removable)
            
    # 计算 Precision (删得准不准)
    precision = tp / len(removed_indices) if len(removed_indices) > 0 else 0
    
    # 计算 Recall (删得干不干净)
    recall = tp / total_gt_duplicates if total_gt_duplicates > 0 else 0
    
    # D. 获取显存占用
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

    # E. 写入结果
    log_result("SemDeDup", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    run_semdedup_benchmark()