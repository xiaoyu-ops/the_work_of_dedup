import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
import multiprocessing

# ================= 配置区域 =================
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\simclr_keep_list.json"
THRESHOLD = 0.93 # 保持和论文一致
BATCH_SIZE = 128
NUM_WORKERS = 4 # Windows 下通常设置为 0-8，根据 CPU 而定
# ============================================

class ImageListDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, idx
        except Exception:
            # 返回全黑图作为占位符，后续会过滤掉 norm 极小的特征
            return torch.zeros((3, 224, 224)), idx

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print("加载 ResNet50 模型...")
    # 使用旧版 weights 参数或新版 weights枚举，这里兼容处理
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except:
        model = models.resnet50(pretrained=True)
        
    # 去掉最后的全连接层，保留特征提取
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 扫描文件夹并分组
    class_groups = {}
    print("正在扫描文件结构...")
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory NOT FOUND: {IMAGE_DIR}")
        return

    for r, d, f in os.walk(IMAGE_DIR):
        imgs = [os.path.join(r, x) for x in f if x.lower().endswith(('.jpg', '.png'))]
        if imgs: 
            class_groups[r] = imgs

    print(f"找到 {len(class_groups)} 个文件夹。")

    # 2. 准备需要计算特征的图片列表
    # 如果文件夹内只有1张图，直接保留，不需要计算特征
    final_keep = []
    files_to_infer = []      # List[path]
    files_to_infer_map = {}  # folder -> List[path indices in files_to_infer]
    
    sorted_folders = sorted(list(class_groups.keys()))
    
    current_idx = 0
    for folder in sorted_folders:
        files = class_groups[folder]
        if len(files) < 2:
            final_keep.extend(files)
        else:
            # 记录这个文件夹对应 files_to_infer 中的哪些索引范围
            start = current_idx
            files_to_infer.extend(files)
            end = len(files_to_infer)
            files_to_infer_map[folder] = list(range(start, end))
            current_idx = end
            
    total_infer = len(files_to_infer)
    print(f"需要计算特征的图片数量: {total_infer} (直接保留单张文件夹图片: {len(final_keep)})")

    if total_infer == 0:
        print("没有需要去重的文件夹。")
        with open(OUTPUT_JSON, "w") as f:
            json.dump(final_keep, f)
        return

    # 3. 批量推理 (Batch Inference)
    # 使用 DataLoader 实现多进程数据加载，这是深度学习中标准的“多进程加速”方式
    print(f"开始批量特征提取 (Batch Size={BATCH_SIZE}, Workers={NUM_WORKERS})...")
    dataset = ImageListDataset(files_to_infer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 我们需要存储所有特征，虽然 3.8M 也还能接受 (3.8M * 2048 * 4B ≈ 30GB)，但还是有点大
    # 为了安全，我们还是存下来。如果内存不足，可以只存当前批处理的文件夹，但那样逻辑复杂。
    # 这里假设 32GB 内存足够处理目前的数据量。如果 OOM，可以将 features 基于 mmap 或者分块处理。
    # 为了避免 OOM，我们使用 numpy array 存储，并在循环中直接写入。
    # 实际上，我们可以只存储 "path -> feature" 的映射，或者按顺序存储一个大矩阵
    
    all_features = np.zeros((total_infer, 2048), dtype=np.float32)
    valid_mask = np.zeros(total_infer, dtype=bool) # 标记是否成功读取且非全黑

    with torch.no_grad():
        for imgs, idxs in tqdm(dataloader, desc="Inference"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze() # [B, 2048]
            
            # 归一化
            if len(feats.shape) == 1:
                feats = feats.unsqueeze(0)
            
            norms = torch.norm(feats, p=2, dim=1, keepdim=True)
            feats = feats / (norms + 1e-8)
            feats_np = feats.cpu().numpy()
            
            idxs_np = idxs.numpy()
            norms_np = norms.cpu().numpy()
            
            # 存入大数组
            all_features[idxs_np] = feats_np
            
            # 简单的有效性检查，如果 norm 太小说明是黑图或读取失败(我们在 Dataset 里返回了全黑)
            # 正常 ResNet 特征 norm 应该比较大，归一化前。
            # 这里只要 norm > 0.001 即可
            valid_batch = (norms_np.squeeze() > 0.001)
            valid_mask[idxs_np] = valid_batch

    # 4. 按文件夹进行去重
    print("开始按文件夹计算相似度矩阵并去重...")
    
    # 我们遍历记录了索引映射的文件夹
    for folder, indices in tqdm(files_to_infer_map.items(), desc="Deduplicating Groups"):
        # 取出该文件夹对应的特征和路径
        # indices 是 files_to_infer 中的下标
        folder_indices = [i for i in indices if valid_mask[i]]
        
        if not folder_indices:
            continue
            
        folder_feats = all_features[folder_indices] # (N, 2048)
        folder_paths = [files_to_infer[i] for i in folder_indices]
        
        N = len(folder_paths)
        if N < 2:
            final_keep.extend(folder_paths)
            continue
            
        # 计算相似度矩阵
        # (N, 2048) @ (2048, N) -> (N, N)
        sim_mat = np.dot(folder_feats, folder_feats.T)
        np.fill_diagonal(sim_mat, 0)
        
        to_remove_local = set()
        
        for i in range(N):
            if i in to_remove_local: continue
            # 找到相似度高的
            dups = np.where(sim_mat[i] > THRESHOLD)[0]
            for j in dups:
                if j > i: # 只移除后面的
                    to_remove_local.add(j)
        
        for i in range(N):
            if i not in to_remove_local:
                final_keep.append(folder_paths[i])

    print(f"总计保留文件数: {len(final_keep)}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_keep, f)
    print(f"完成！已保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    # Windows 下 DataLoader num_workers > 0 需要 freeze_support
    multiprocessing.freeze_support()
    main()