import os
import time
import csv
import json
import torch
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------------
# 0. 配置区域
# -------------------------------------------------------------

# pipeline 输出目录（根据你 big_run 的配置）
# 我们需要从中读取 summary.json 来找到 duplicates 信息
PIPELINE_SUMMARY_FILE = Path(r"D:\Deduplication_Framework\outputs\image_experiment\summary.json")

# Ground Truth 原始图片目录
IMAGE_SOURCE_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

# 结果追加写入的文件
RESULT_FILE = r"D:\Deduplication_framework\2026_new_experiment\result\image_benchmark_results.csv"

# -------------------------------------------------------------
# 1. 辅助函数
# -------------------------------------------------------------

def parse_id(filename):
    """
    文件名解析逻辑，需与基线一致。
    例如 train-0_0_aug_noise.jpg -> train-0_0
    """
    name = os.path.splitext(filename)[0]
    if "_aug" in name:
        return name.split("_aug")[0]
    return name

def log_result_csv(method, throughput, precision, recall, gpu_mem):
    """追加写入 CSV"""
    file_exists = os.path.isfile(RESULT_FILE)
    try:
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Method", "Throughput (imgs/s)", "Precision", "Recall", "GPU Mem (GB)"])
            
            # 如果是无效值，格式化一下
            tp_str = f"{throughput:.1f}" if throughput >= 0 else "N/A"
            mem_str = f"{gpu_mem:.2f}" if gpu_mem >= 0 else "0.00"
            
            writer.writerow([method, tp_str, f"{precision*100:.2f}%", f"{recall*100:.2f}%", mem_str])
        print(f"[成功] 结果已追加到 {RESULT_FILE}")
    except Exception as e:
        print(f"[错误] 写入CSV失败: {e}")

# -------------------------------------------------------------
# 2. 主逻辑
# -------------------------------------------------------------

def main():
    print(f"[My Pipeline Evaluation] 开始评估...")

    # A. 加载 Pipeline 产出的 duplicates
    if not PIPELINE_SUMMARY_FILE.exists():
        print(f"[错误] 找不到 summary 文件: {PIPELINE_SUMMARY_FILE}")
        print("请确认 pipeline 是否已运行完毕 (check outputs/big_run)。")
        return
    
    duplicates_map = {} # keeper -> [dup1, dup2...]
    duration_total = 0.0
    processed_count = 0
    
    try:
        with open(PIPELINE_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        # 1. 寻找 image stage 的信息
        image_stage = None
        if "stages" in summary:
            for stage in summary["stages"]:
                if stage.get("stage_name") == "stage2_image":
                    image_stage = stage
                    break
        
        if not image_stage:
            print("[错误] summary.json 中未找到 'stage2_image' 阶段的信息。")
            return

        # 2. 从 stage 信息中获取 duplicates 文件列表
        output_paths = image_stage.get("output_paths", {})
        dup_files = output_paths.get("duplicates", [])
        
        if not dup_files:
            print("[警告] stage2_image 报告中没有 duplicates 文件输出。可能没有发现重复？")
        
        print(f"[Info] 找到 {len(dup_files)} 个分片结果文件，开始合并...")

        # 3. 遍历读取每个分片文件
        for dup_file_path in dup_files:
            if not os.path.exists(dup_file_path):
                print(f"[警告] 找不到结果文件: {dup_file_path}，跳过。")
                continue
                
            with open(dup_file_path, 'r', encoding='utf-8') as df:
                items = json.load(df)
                # items 结构: [{"original": "pathA", "duplicates": [{"path": "pathB", "similarity": 0.99}, ...]}, ...]
                
                for item in items:
                    keeper = item.get("original")
                    if not keeper: continue
                    
                    dups_list = item.get("duplicates", [])
                    if not dups_list: continue

                    if keeper not in duplicates_map:
                        duplicates_map[keeper] = []
                    
                    for dup_info in dups_list:
                        d_path = dup_info.get("path")
                        if d_path:
                            duplicates_map[keeper].append(d_path)

        # 4. 获取统计信息 (耗时等)
        runner_stats = image_stage.get("metadata", {}).get("runner_summary", {}).get("stats", {})
        processed_count = runner_stats.get("processed", 0)
        # 优先使用 metadata 里的 elapsed_seconds (通常更准)，否则用 stage 的
        duration_total = runner_stats.get("elapsed_seconds", 0)
        if duration_total == 0:
             duration_total = image_stage.get("elapsed_seconds", 0)
        
    except Exception as e:
        print(f"[严重错误] 解析过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Pipeline 报告耗时: {duration_total:.2f}s, 处理文件: {processed_count}")
    print(f"发现重复组 (Clusters): {len(duplicates_map)} (含 Keeper)")

    # B. 建立 Ground Truth (基于 Pipeline 处理的实际文件列表)
    print(f"正在读取 Pipeline 输入 Manifest 以建立 Ground Truth...")
    
    all_files_gt = []
    
    # 从 image_stage 里的 output_paths 获取输入 manifest
    # 注意：stage2_image 的 output_paths['manifests'] 其实是它处理的分片输入 manifest
    input_manifests = image_stage.get("output_paths", {}).get("manifests", [])
    
    if not input_manifests:
        # 如果没有找到具体的 manifests 列表，尝试 fallback 到 metadata 或者直接报错
        print("[错误] 无法在 summary 中找到 input manifests 列表，无法确定 GT 范围。")
        return

    for mani_path in input_manifests:
        if not os.path.exists(mani_path):
            print(f"[警告] 找不到 Manifest 文件: {mani_path}")
            continue
            
        with open(mani_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 获取文件名
                    fname = os.path.basename(line)
                    all_files_gt.append(fname)
    
    if not all_files_gt:
        print("[错误] 未能从 Manifest 中读取到任何文件。")
        return

    # 统计 GT Pairs
    id_counts = defaultdict(int)
    for name in all_files_gt:
        fid = parse_id(name)
        id_counts[fid] += 1
        
    total_gt_pairs = 0
    for count in id_counts.values():
        if count > 1:
            total_gt_pairs += (count * (count - 1)) // 2
            
    print(f"Ground Truth: 总文件 {len(all_files_gt)}, 真实重复对 {total_gt_pairs}")

    # C. 计算 TP / FP
    tp = 0
    fp = 0
    
    # 遍历每个聚类
    for keeper_path, dup_paths in duplicates_map.items():
        # 一个聚类包含 [keeper, dup1, dup2...]
        # 只需要文件名
        cluster_files = [os.path.basename(keeper_path)] + [os.path.basename(d) for d in dup_paths]
        
        # 解析 ID
        ids = [parse_id(f) for f in cluster_files]
        n = len(ids)
        if n < 2: continue
        
        # 两两比对
        # 这里的 cluster 是依据算法得出的“认为这一组都是一样的”
        # 所以这组里任意两张图片，都被算法判断为“重复”
        # 我们要检查这一组里的 Pair 是否真正拥有相同的 ID
        
        # 注意：这里的逻辑是 Pairwise 的
        # 如果一组有 k 个元素，算法声称发现了 k(k-1)/2 个 Pairs
        # 我们一个个检查这些 Pairs 是否正确
        for i in range(n):
            for j in range(i+1, n):
                if ids[i] == ids[j]:
                    tp += 1
                else:
                    fp += 1
    
    # D. 计算最后指标
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    
    # 吞吐量
    throughput = (processed_count / duration_total) if duration_total > 0 else 0
    
    # 显存 (如果是离线跑，无法获取峰值，这里填 0)
    gpu_mem = 0.0
    
    print("-" * 40)
    print(f"结果: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%")
    print(f"TP={tp}, FP={fp}, Total GT Pairs={total_gt_pairs}")
    print("-" * 40)

    # 写入 CSV
    log_result_csv("My Pipeline", throughput, precision, recall, gpu_mem)

if __name__ == "__main__":
    main()
