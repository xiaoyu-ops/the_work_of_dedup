import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

# ================= 配置区域 =================
# 指向 imagenet_bloated 文件夹
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 32
SIMILARITY_THRESHOLD = 0.93
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
# ===========================================

def get_all_images(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
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

def run_simclr_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline 2] SimCLR (ResNet50) 评测...")
    
    all_files = get_all_images(IMAGE_DIR)
    if not all_files: return
    test_files = all_files[:SAMPLE_SIZE]
    
    # 加载 ResNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1]) 
    model.to(device).eval()
    
    dataset = BenchDataset(test_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    feats_list = []
    paths_list = []
    
    start_time = time.time()
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze()
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            feats_list.append(feats.cpu().numpy())
            paths_list.extend(paths)
            
    extract_time = time.time() - start_time
    throughput = len(test_files)/extract_time
    embeddings = np.concatenate(feats_list)
    
    # 统计 Precision/Recall
    print("   正在计算相似度矩阵...")
    id_list = [parse_id(os.path.basename(p)) for p in paths_list]
    from collections import Counter
    cnt = Counter(id_list)
    total_gt_pairs = sum([(c*(c-1))//2 for c in cnt.values() if c > 1])
        
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    pairs = np.where(np.triu(sim_matrix, k=1) > SIMILARITY_THRESHOLD)
    
    tp, fp = 0, 0
    for i, j in zip(pairs[0], pairs[1]):
        if id_list[i] == id_list[j]: tp += 1
        else: fp += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt_pairs if total_gt_pairs > 0 else 0
    
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated()/(1024**3)

    log_result("SimCLR Embed", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    run_simclr_benchmark()