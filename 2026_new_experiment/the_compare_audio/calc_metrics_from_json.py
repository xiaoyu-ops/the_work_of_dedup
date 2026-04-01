import os
import json

# ================= Configuration =================
# 1. JSON List Files (Check file names carefully)
JSON_FILES = {
    "MD5 Hash":       r"D:\Deduplication_framework\2026_new_experiment\result\audio_md5_keep_list.json",
    "MFCC+Cosine":    r"D:\Deduplication_framework\2026_new_experiment\result\audio_mfcc_keep_list.json",
    "Ours (LSH)":     r"D:\Deduplication_framework\2026_new_experiment\result\audio_ours_keep_list.json"
}

# 2. Ground Truth File
GT_PATH = r"D:\Deduplication_framework\2026_new_experiment\datasets\ground_truth\audio_ground_truth.json"

# 3. Source Audio Directory (For counting total files)
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"
# =============================================

def load_ground_truth():
    print(f"[INFO] Loading Ground Truth: {os.path.basename(GT_PATH)}...")
    if not os.path.exists(GT_PATH):
        print(f"[ERROR] Ground Truth file not found: {GT_PATH}")
        return set()
        
    with open(GT_PATH, 'r') as f:
        data = json.load(f)
    # Get "exact_duplicates" list
    return set(data.get("exact_duplicates", []))

def get_total_files():
    # Count .wav files in source directory
    count = 0
    all_filenames = set()
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                count += 1
                all_filenames.add(file)
    return count, all_filenames

def calculate(name, json_path, gt_duplicates, total_count, all_filenames_set):
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: {json_path}")
        return None

    with open(json_path, 'r') as f:
        keep_list = json.load(f)
    
    # Get set of kept filenames (basename only)
    kept_filenames = set([os.path.basename(p) for p in keep_list])
    
    # Calculate Removed files
    # Logic: All Files - Kept Files
    removed_filenames = all_filenames_set - kept_filenames
    removed_count = len(removed_filenames)
    
    # --- Metrics Calculation ---
    # TP (True Positive): Duplicate in GT AND Removed -> Success
    tp = 0
    for f in removed_filenames:
        if f in gt_duplicates:
            tp += 1
            
    # FP (False Positive): Unique file (not in GT) BUT Removed -> Mistake
    fp = removed_count - tp
    
    # Precision = Correctly Removed / Total Removed
    precision = (tp / removed_count) if removed_count > 0 else 0.0
    
    # Recall = Correctly Removed / Total Duplicates in GT
    total_gt = len(gt_duplicates)
    recall = (tp / total_gt) if total_gt > 0 else 0.0
    
    # Dedup Rate
    dedup_rate = (removed_count / total_count) * 100
    
    return {
        "Method": name,
        "Dedup%": f"{dedup_rate:.2f}%",
        "Prec.": f"{precision*100:.2f}%",
        "Recall": f"{recall*100:.2f}%",
        "Kept": len(keep_list),
        "Removed": removed_count
    }

if __name__ == "__main__":
    # 1. Prepare Base Data
    gt_set = load_ground_truth()
    if not gt_set:
        print("[ERROR] Failed to load Ground Truth. Exiting.")
        exit()

    total_files, all_files_set = get_total_files()
    
    print(f"Total Files in Source: {total_files}")
    print(f"Total Ground Truth Duplicates: {len(gt_set)}")
    print("-" * 80)
    print(f"{'Method':<15} | {'Dedup%':<10} | {'Prec.':<10} | {'Recall':<10} | {'Kept'}")
    print("-" * 80)
    
    # 2. Calculate for each method
    for name, path in JSON_FILES.items():
        res = calculate(name, path, gt_set, total_files, all_files_set)
        if res:
            print(f"{res['Method']:<15} | {res['Dedup%']:<10} | {res['Prec.']:<10} | {res['Recall']:<10} | {res['Kept']}")
    
    print("-" * 80)
    print("[DONE] Please copy the data above to your paper table.")