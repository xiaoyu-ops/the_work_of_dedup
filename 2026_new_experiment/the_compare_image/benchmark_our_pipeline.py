
import os
import sys
import time
import csv
import json
from pathlib import Path
from tqdm import tqdm

# --- 引入项目根目录以调用真实 Pipeline API ---
PROJECT_ROOT = r"D:\Deduplication_framework"
sys.path.insert(0, PROJECT_ROOT)

# 动态添加 image 模块的搜索路径，防止相对导入错误
sys.path.insert(0, os.path.join(PROJECT_ROOT, "image"))

import torch
import open_clip

# 手动 Patch Pipeline API，防止 import 失败
# 因为有时候相对导入或环境路径问题，pipeline_api 内部的 try-import 可能会失败
# 我们在这里强行注入
import image.method.pipeline_api as pipeline_api
pipeline_api.torch = torch
pipeline_api.open_clip = open_clip

from image.method.pipeline_api import (
    ImagePipelineConfig,
    EmbeddingConfig,
    DedupConfig,
    _compute_embeddings_open_clip,
    _run_deduplication
)

# ================= 配置区域 =================
# 指向与 benchmark_simclr_updated 相同的数据集
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 64
EPS = 0.07 # 对应流水线默认配置
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
# ===========================================


from image.method.pipeline_api import (
    ImagePipelineConfig,
    EmbeddingConfig,
    DedupConfig,
    _compute_embeddings_open_clip,
    _run_deduplication
)

# ================= 配置区域 =================
# 指向与 benchmark_simclr_updated 相同的数据集
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
BATCH_SIZE = 64
EPS = 0.07 # 对应流水线默认配置
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"
MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K" # 必须与 configs/image_config.yaml 一致
# ===========================================

def get_all_images(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# Ground Truth 逻辑：文件名包含 _aug 视为同一组
def parse_id(filename):
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result(method, throughput, precision, recall, gpu_mem):
    file_exists = os.path.isfile(RESULT_FILE)
    try:
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            writer.writerow([method, f"{throughput:.1f}", f"{precision*100:.2f}%", f"{recall*100:.2f}%", f"{gpu_mem:.2f}"])
        print(f"[Success] '{method}' results written to {RESULT_FILE}")
    except Exception as e:
        print(f"[Error] Failed to write CSV: {e}")

def run_real_pipeline_benchmark():
    print(f"[Benchmark] Ours (Real Pipeline Logic: SemDeDup) on imagenet_bloated subset...")
    
    # 1. 准备数据
    all_files = get_all_images(IMAGE_DIR)
    if not all_files:
        print("No images found.")
        return

    # 这里我们模拟流水线输入，使用 Path 对象
    test_paths = [Path(f) for f in all_files[:SAMPLE_SIZE]]
    print(f"Total files to test: {len(test_paths)}")
    
    # 2. 配置 Pipeline
    # 我们构造与 YAML 配置相同的对象
    emb_config = EmbeddingConfig(
        backend="open_clip",
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device="auto"
    )
    
    # 关键点：这里我们要开启 SemDeDup，而非 pairwise
    # 注意：SemDeDup 需要聚类。如果数据量太小(<1000)，pipeline 可能会回退或聚类数很少，这符合预期。
    dedup_config = DedupConfig(
        method="semdedup", 
        eps=EPS,
        legacy_cluster_dir=None, # 强制重新计算聚类
        legacy_keep_indices_file=None
    )
    
    start_time = time.time()
    gpu_mem = 0
    
    try:
        # 3. 调用真实 API 提取特征
        # _compute_embeddings_open_clip 返回 (embeddings, valid_paths, failed_paths, backend_name)
        embeddings, valid_paths, failed_paths, _ = _compute_embeddings_open_clip(test_paths, emb_config)
        
        extract_time = time.time()
        print(f"Embedding extraction done. Shape: {embeddings.shape}")
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # 4. 调用真实 API 进行 SemDeDup 去重
        # 这会自动触发 K-Means 聚类和簇内去重
        dedup_result = _run_deduplication(
            valid_paths,
            embeddings,
            dedup_config,
            indices=None # 不传递 indices，让它根据 embedding 动态聚类
        )
        
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    total_time = time.time() - start_time
    throughput = len(test_paths) / total_time if total_time > 0 else 0
    print(f"Pipeline finished. Total Throughput (Extract+SemDeDup): {throughput:.2f} imgs/s")
    
    # 5. 评估结果 vs Ground Truth
    
    kept_set = set([str(p) for p in dedup_result['keepers']])
    # 被删掉的文件集合
    
    # 简单起见，从 valid_paths 中减去 kept_set 即可得到的被删列表
    valid_paths_str = [str(p) for p in valid_paths]
    removed_files = [p for p in valid_paths_str if p not in kept_set]
    
    print(f"Kept: {len(kept_set)}, Removed: {len(removed_files)}")
    
    # 开始算 P/R (Action-Based)
    print("Evaluating Precision & Recall...")
    
    # Parse IDs
    id_map = {str(p): parse_id(os.path.basename(str(p))) for p in valid_paths}
    all_labels = [id_map[p] for p in valid_paths_str]
    
    # A. 计算 Total Ground Truth Pairs
    from collections import Counter
    cnt = Counter(all_labels)
    # total_gt_pairs = sum([(c*(c-1))//2 for c in cnt.values() if c > 1])
    # print(f"Total Ground Truth Duplicate Pairs: {total_gt_pairs}")
    
    # --- 采用 Action-Based Evaluation (更适合 SemDeDup) ---
    # Precision = (被成功去除的冗余) / (所有被去除的文件)
    # Recall = (被成功去除的冗余) / (Ground Truth 中应该被去除的总数)
    
    unique_ids_count = len(cnt)
    total_should_remove = len(valid_paths) - unique_ids_count
    
    tp_action = 0 # 删得对
    
    # 建立 Keepers 的 ID 集合
    keeper_ids = set()
    for k in kept_set:
        keeper_ids.add(id_map[k])
        
    for removed in removed_files:
        rid = id_map[removed]
        if rid in keeper_ids:
            # 删了它，但这个 ID 还有别的副本留着 -> 删对了 (冗余)
            tp_action += 1
        else:
            # 删了它，结果这个 ID 全军覆没（或者本来就只有一个） -> 删错了
            pass
            
    actual_precision = tp_action / len(removed_files) if len(removed_files) > 0 else 0
    actual_recall = tp_action / total_should_remove if total_should_remove > 0 else 0
    
    print(f"Precision (Reduction): {actual_precision*100:.2f}%")
    print(f"Recall (Reduction): {actual_recall*100:.2f}%")
    print(f"Peak GPU Mem: {gpu_mem:.2f} GB")

    log_result("Ours (Real SemDeDup 10k)", throughput, actual_precision, actual_recall, gpu_mem)

if __name__ == "__main__":
    run_real_pipeline_benchmark()
