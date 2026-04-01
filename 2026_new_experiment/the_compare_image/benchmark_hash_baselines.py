import os
import time
import hashlib
import imagehash
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import csv

# ================= 配置区域 =================
# 指向 imagenet_bloated 文件夹
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

SAMPLE_SIZE = 10000 
HASH_THRESHOLD = 5 
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv" 
# ===========================================

def get_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def parse_id(filename):
    """
    解析文件名逻辑：
    train-xxx_229.jpg           -> ID: train-xxx_229
    train-xxx_229_aug_noise.jpg -> ID: train-xxx_229
    """
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def get_all_images(root_dir):
    image_files = []
    print(f"正在扫描目录: {root_dir} ...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

def log_result(method, throughput, precision, recall, gpu_mem="0"):
    """写入 CSV"""
    file_exists = os.path.isfile(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
        writer.writerow([method, f"{throughput:.1f}", f"{precision*100:.2f}%", f"{recall*100:.2f}%", gpu_mem])
    print(f"[成功] {method} 结果已写入文件")

def run_benchmark():
    print(f"[Baseline 1] 开始传统哈希评测...")
    
    all_files = get_all_images(IMAGE_DIR)
    if not all_files:
        print("[错误] 未找到图片，请检查路径！")
        return
        
    test_files = all_files[:SAMPLE_SIZE]
    print(f"[信息] 采样: {len(test_files)} / 总数: {len(all_files)}")

    # --- MD5 ---
    print("\n[1/2] 测试 MD5...")
    start_time = time.time()
    md5_groups = defaultdict(list)
    
    for f in tqdm(test_files, desc="MD5"):
        h = get_md5(f)
        fid = parse_id(os.path.basename(f))
        md5_groups[h].append(fid)
    
    cost = time.time() - start_time
    throughput = len(test_files) / cost
    
    # 统计 Recall
    id_counts = defaultdict(int)
    for f in test_files:
        fid = parse_id(os.path.basename(f))
        if fid: id_counts[fid] += 1
    total_gt_pairs = sum([(c*(c-1))//2 for c in id_counts.values() if c > 1])
    
    tp = 0
    for ids in md5_groups.values():
        if len(ids) > 1:
            n = len(ids)
            tp += (n * (n - 1)) // 2
            
    recall = tp / total_gt_pairs if total_gt_pairs > 0 else 0
    log_result("MD5 Hash", throughput, 1.0, recall)

    # --- pHash ---
    print("\n[2/2] 测试 pHash...")
    start_time = time.time()
    phash_dict = {}
    
    for f in tqdm(test_files, desc="pHash"):
        try:
            phash_dict[f] = imagehash.phash(Image.open(f))
        except: pass
        
    cost = time.time() - start_time
    throughput = len(test_files) / cost
    
    # 两两比对
    paths = list(phash_dict.keys())
    hashes = list(phash_dict.values())
    n = len(paths)
    
    # 重新计算有效 GT
    sub_ids = [parse_id(os.path.basename(p)) for p in paths]
    from collections import Counter
    cnt = Counter(sub_ids)
    valid_gt = sum([(c*(c-1))//2 for c in cnt.values() if c > 1])
    
    tp, fp = 0, 0
    print("   正在计算 Precision/Recall (O(N^2))...")
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= HASH_THRESHOLD:
                if sub_ids[i] == sub_ids[j]: tp += 1
                else: fp += 1
                
    rec = tp / valid_gt if valid_gt > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    log_result("pHash", throughput, prec, rec)

if __name__ == "__main__":
    run_benchmark()