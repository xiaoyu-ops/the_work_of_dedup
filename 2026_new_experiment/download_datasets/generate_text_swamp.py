import os
import json
import time
import random

SOURCE_FILE = "D:\\桌面\\Deduplication_framework\\2026_new_experiment\\datasets\\text_seed_data\\amazon_reviews_seed.csv"
OUTPUT_DIR = "D:\\桌面\\Deduplication_framework\\2026_new_experiment\\datasets\\final_swamp_data\\digital_swamp_text"  # 最终的 500GB 数据目录
TARGET_SIZE_GB = 500  # 目标总大小
NUM_COPIES = 300      # 预计复制份数 (1.6GB * 300 ≈ 480GB)
NEAR_DUP_COUNT = int(NUM_COPIES * 0.2)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 召回率计算-记录ground truth
ground_truth_log = {
    "exact_duplicates": [],  # 存完全一样的文件列表
    "near_duplicates": [],   # 存修改过的文件列表
    "original_seed": SOURCE_FILE
}

def inject_noise(content_bytes):
    """
    修改版：注入带随机ID的注释或乱码，确保 MinHash 也能感知到微小变化，
    或者确保它不会被简单的 whitespace normalization 过滤掉。
    """
    # 生成一个随机ID
    rand_id = random.randint(100000, 999999)
    
    # 注入一个带随机ID的注释（示例：<!--noise id:123456-->），包含字母和数字
    # 这样 MD5 肯定变了，MinHash 也会捕捉到 token 的增加，但 Jaccard 相似度依然很高 (>0.95)
    noise = f"\n<!--noise id:{rand_id}-->\n".encode('utf-8')
    return content_bytes + noise

print(f"开始构建 {TARGET_SIZE_GB}GB 数字沼泽...")
start_time = time.time()

# 读取种子文件到内存 (1.6GB 内存通常没问题，如果爆内存请改成分块读取)
print("正在读取种子文件...")
with open(SOURCE_FILE, 'rb') as f:
    seed_content = f.read()

current_size = 0
file_count = 0

for i in range(NUM_COPIES):
    file_name = f"part_{i:04d}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    
    # 策略：前 20% 做成“近似重复”，后 80% 做成“精确重复”
    # 这样你可以分别测试 MinHash 和 Hash 去重的能力
    if i < NEAR_DUP_COUNT:
        # --- 近似重复 (Near Duplicate) ---
        modified_content = inject_noise(seed_content)
        with open(file_path, 'wb') as f_out:
            f_out.write(modified_content)
        ground_truth_log["near_duplicates"].append(file_name)
        mode = "Near-Dup"
    else:
        # --- 精确重复 (Exact Duplicate) ---
        # 直接写入原内容
        with open(file_path, 'wb') as f_out:
            f_out.write(seed_content)
        ground_truth_log["exact_duplicates"].append(file_name)
        mode = "Exact-Dup"
        
    # 更新统计
    file_size = os.path.getsize(file_path)
    current_size += file_size
    file_count += 1
    # 如果达到了目标大小则提前结束
    if current_size >= TARGET_SIZE_GB * (1024**3):
        print(f"已达到目标大小 {TARGET_SIZE_GB} GB，提前停止生成。")
        break
    
    # 每生成 10 个文件打印一次进度
    if i % 10 == 0:
        total_gb = current_size / (1024**3)
        print(f"[{i}/{NUM_COPIES}] 生成: {mode} | 当前总大小: {total_gb:.2f} GB")

# 保存 Ground Truth 日志
GROUND_TRUTH_PATH = "D:\\桌面\\Deduplication_framework\\2026_new_experiment\\datasets\\ground_truth\\text_ground_truth.json"
os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
with open(GROUND_TRUTH_PATH, "w") as f:
    json.dump(ground_truth_log, f, indent=4)

end_time = time.time()
print(f"\n数字沼泽构建完成！")
print(f"总文件数: {file_count}")
print(f"总大小: {current_size / (1024**3):.2f} GB")
print(f"耗时: {(end_time - start_time)/60:.2f} 分钟")
print(f"数据位置: {OUTPUT_DIR}")
print(f"标准答案已保存至: D:\\桌面\\Deduplication_framework\\2026_new_experiment\\datasets\\ground_truth\\text_ground_truth.json (用于计算Recall)")