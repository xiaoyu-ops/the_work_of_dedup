import os
import imagehash
from PIL import Image
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 配置区域 =================
IMAGE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"
OUTPUT_JSON = r"D:\Deduplication_framework\2026_new_experiment\result\phash_keep_list.json"
# ============================================

def get_phash_worker(file_path):
    """单独的 Worker 函数，计算单个文件的 pHash"""
    try:
        img = Image.open(file_path)
        # 只要 hash 值字符串一样就算重复 (精确匹配以提升速度)
        file_hash = str(imagehash.phash(img))
        return file_hash, file_path
    except Exception:
        return None

def main():
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
    
    num_processes = max(1, cpu_count() - 2)
    print(f"启动 {num_processes} 个进程进行 pHash 计算...")

    with Pool(processes=num_processes) as pool:
        # pHash 计算是 CPU 密集型，多进程提升会很显著
        results = list(tqdm(pool.imap_unordered(get_phash_worker, files, chunksize=50), total=total_files, desc="Calculating pHash"))

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