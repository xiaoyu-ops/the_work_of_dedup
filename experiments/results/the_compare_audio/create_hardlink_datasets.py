import os
import json
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ================= 配置区域 =================
# 1. 原始数据的根目录 (用于计算相对路径)
# 注意：硬链接要求源文件和目标文件夹必须在【同一个硬盘分区】(比如都在 D 盘)
BASE_DIR = r"D:\Deduplication_framework\2026_new_experiment"

# 2. 输出目录 (只保留这一个正确的定义)
TARGET_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_deduped_datasets"

# 3. 任务列表 (包含 Ours, MFCC, MD5)
TASKS = [
    # === 任务 1: Ours (Spectrogram + pHash) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_ours_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_ours_deduped"
    },
    
    # === 任务 2: MFCC (Baseline) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_mfcc_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_mfcc_deduped"
    },

    # === 任务 3: MD5 (Baseline) ===
    {
        "json": r"D:\Deduplication_framework\2026_new_experiment\result\audio_md5_keep_list.json",
        "src_root": r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio",
        "target_name": "audio_md5_deduped"
    }
]
# ===========================================

# 全局变量用于多进程传参
GLOBAL_SRC_ROOT = None
GLOBAL_TARGET_DIR = None

def init_worker(src_root, target_dir):
    """初始化子进程"""
    global GLOBAL_SRC_ROOT, GLOBAL_TARGET_DIR
    GLOBAL_SRC_ROOT = Path(src_root)
    GLOBAL_TARGET_DIR = Path(target_dir)

def worker_link(file_path):
    """子进程：创建硬链接"""
    try:
        # 1. 计算相对路径
        try:
            p = Path(file_path)
            rel_path = p.relative_to(GLOBAL_SRC_ROOT)
        except ValueError:
            p = Path(file_path)
            rel_path = p.name

        # 2. 目标路径
        dest_path = GLOBAL_TARGET_DIR / rel_path
        
        # 3. 如果已存在则跳过
        if dest_path.exists():
            return True

        # 4. 创建父目录
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 5. 创建链接
        try:
            os.link(file_path, dest_path)
        except OSError:
            shutil.copy2(file_path, dest_path)
        
        return True
    except Exception:
        return False

def run_multicore():
    # 启用多核
    num_cores = max(1, cpu_count() - 2)
    print(f"[Info] 启用 {num_cores} 核心并行构建数据集...")

    for task in TASKS:
        json_path = task['json']
        target_name = task['target_name']
        src_root = task['src_root']
        
        if not os.path.exists(json_path):
            print(f"[Warning] 跳过: 找不到 JSON {json_path}")
            continue

        target_dir = os.path.join(TARGET_ROOT, target_name)
        
        # 清理旧目录
        if os.path.exists(target_dir):
            print(f"[Info] 正在重建目录: {target_name} ...")
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        # 读取列表
        with open(json_path, 'r') as f:
            keep_list = json.load(f)

        print(f"[Info] 开始构建: {target_name} ({len(keep_list)} 个文件)...")
        
        # 并行执行
        with Pool(processes=num_cores, initializer=init_worker, initargs=(src_root, target_dir)) as pool:
            results = list(tqdm(pool.imap_unordered(worker_link, keep_list, chunksize=100), 
                              total=len(keep_list), 
                              desc=f"Linking"))
            
        print(f"[Success] 完成! 成功: {sum(results)}/{len(keep_list)}\n")

if __name__ == "__main__":
    run_multicore()