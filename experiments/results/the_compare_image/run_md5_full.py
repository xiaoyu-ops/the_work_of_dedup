import os
import hashlib
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 配置区域 =================
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\md5_keep_list.json"
# ============================================

def get_md5_worker(file_path):
    """单独的 Worker 函数，计算单个文件的 MD5"""
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash, file_path
    except Exception:
        # 如果读取出错（如文件损坏），返回 None
        return None

def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory NOT FOUND: {IMAGE_DIR}")
        return

    files = []
    print(f"正在扫描文件结构: {IMAGE_DIR} ...")
    for r, d, f in os.walk(IMAGE_DIR):
        for file in f:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                files.append(os.path.join(r, file))

    total_files = len(files)
    print(f"找到 {total_files} 个图片文件。")

    seen = set()
    keep = []
    
    # 根据 CPU 核心数设定进程数，保留两个核心给系统
    num_processes = max(1, cpu_count() - 2)
    print(f"启动 {num_processes} 个进程进行 MD5 计算...")

    # 使用 Pool 进行多进程处理
    # chunksize 设置大一点可以减少进程间通信开销
    with Pool(processes=num_processes) as pool:
        # imap_unordered 会乱序返回结果，但因为我们需要全部处理，顺序不重要且能更快且实时显示进度
        results = list(tqdm(pool.imap_unordered(get_md5_worker, files, chunksize=100), total=total_files, desc="Calculating MD5"))

    print("正在进行去重筛选...")
    for res in results:
        if res is None:
            continue
        h, f_path = res
        if h not in seen:
            seen.add(h)
            keep.append(f_path)

    print(f"原始数量: {total_files} -> 保留数量: {len(keep)}")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(keep, f)
    print(f"完成！已保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()