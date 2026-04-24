import os
import json
import hashlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 配置区域 =================
# 音频源数据路径
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"

# 结果保存路径
RESULT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\audio_md5_keep_list.json"
# ===========================================

def get_files():
    files = []
    print("正在扫描音频文件...")
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

def worker_md5(file_path):
    """子进程：计算单个文件的 MD5"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # 分块读取，防止内存溢出（虽然音频文件通常不大）
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return (file_path, hash_md5.hexdigest())
    except Exception:
        # 如果读取失败，为了安全起见返回 None，主进程会处理
        return None

if __name__ == "__main__":
    # 1. 扫描文件
    all_files = get_files()
    if not all_files:
        print("错误：未找到音频文件！")
        exit()

    # 2. 准备多进程
    num_cores = max(1, cpu_count() - 16)
    print(f"启用 {num_cores} 核心并行计算 MD5...")

    # 3. 并行计算
    results = []
    with Pool(processes=num_cores) as pool:
        # 使用 imap_unordered 让进度条实时滚动
        for res in tqdm(pool.imap_unordered(worker_md5, all_files, chunksize=50), 
                       total=len(all_files), 
                       desc="Calculating MD5"):
            if res is not None:
                results.append(res)

    # 4. 去重逻辑 (单线程汇总)
    print("正在进行去重筛选...")
    seen_hashes = set()
    keep_list = []
    
    # 结果包含 (path, md5)
    # 我们遍历结果，只有第一次见到的 hash 才保留 path
    for file_path, md5_val in results:
        if md5_val not in seen_hashes:
            seen_hashes.add(md5_val)
            keep_list.append(file_path)

    # 5. 保存 JSON
    os.makedirs(os.path.dirname(RESULT_JSON), exist_ok=True)
    with open(RESULT_JSON, 'w') as f:
        json.dump(keep_list, f)
        
    print(f"MD5 列表生成完毕！")
    print(f"   总文件: {len(all_files)}")
    print(f"   保留文件: {len(keep_list)}")
    print(f"   保存路径: {RESULT_JSON}")