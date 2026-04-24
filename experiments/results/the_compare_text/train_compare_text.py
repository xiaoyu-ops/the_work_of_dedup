import os
import glob
import time
import hashlib
from typing import Iterable, Tuple, List, Set

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
RAW_DATA_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_text"
DEDUP_DIR = r"D:\Deduplication_framework_text_deduped"
TEXT_COLUMN = "content"
LABEL_COLUMN = "label"
TITLE_COLUMN = "title"

CHUNK_SIZE = 20000
EPOCHS = 3 
VAL_RATIO = 0.2
SPLIT_SEED = 42
OUTPUT_CSV = r"D:\Deduplication_framework\2026_new_experiment\result_text\text_training_results.csv"

# 强制限制文件数量（必须与去重阶段保持一致！）
# 设为 10 以匹配你之前的去重脚本
MAX_RAW_FILES = 10 

# 如果明确知道类别，比如 [0, 1]，填在这里可以跳过扫描
KNOWN_CLASSES = None 

METHODS = [
    ("No Dedup", None),
    ("MD5", "dedup_md5.csv"),
    ("SimHash", "dedup_simhash.csv"),
    ("MinHash (Std)", "dedup_minhash_lsh.csv"),
    ("Ours (Multi)", "dedup_ours_lsh.csv"),
]
# ===========================================

def iter_raw_files() -> List[str]:
    # 【关键修改】这里增加了切片 [:MAX_RAW_FILES]
    # 确保只读取前 10 个文件，与去重阶段的数据源保持绝对一致
    all_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "part_*.csv")))
    if MAX_RAW_FILES and len(all_files) > MAX_RAW_FILES:
        print(f"[Info] Limiting raw files to first {MAX_RAW_FILES} (out of {len(all_files)})")
        return all_files[:MAX_RAW_FILES]
    return all_files

def get_file_iterator(file_path: str, chunk_size=CHUNK_SIZE) -> Iterable[pd.DataFrame]:
    """通用的文件读取生成器"""
    try:
        reader = pd.read_csv(
            file_path,
            usecols=[LABEL_COLUMN, TITLE_COLUMN, TEXT_COLUMN],
            on_bad_lines="skip",
            encoding="utf-8",
            chunksize=chunk_size,
        )
        for chunk in reader:
            yield chunk
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}")

def iter_dataset(files: List[str] | str) -> Iterable[pd.DataFrame]:
    """统一接口：处理单个文件路径或文件列表"""
    if isinstance(files, str):
        files = [files]
    
    for path in files:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            continue
        yield from get_file_iterator(path)

def is_validation_sample(text: str) -> bool:
    """
    确定性划分：基于内容的Hash。
    """
    if not isinstance(text, str):
        return False
    h = hashlib.md5((str(text) + str(SPLIT_SEED)).encode("utf-8")).hexdigest()
    # 前8位16进制转整数，模100，小于20则为验证集
    return int(h[:8], 16) % 100 < int(VAL_RATIO * 100)

def collect_label_set(files: List[str]) -> List[int]:
    """扫描所有数据以获取完整标签集"""
    if KNOWN_CLASSES is not None:
        return KNOWN_CLASSES
        
    print("Scannng for labels (limit 10 chunks)...")
    labels = set()
    chunks_scanned = 0
    # 仅扫描原始数据即可
    for chunk in iter_dataset(files):
        labels.update(chunk[LABEL_COLUMN].dropna().unique().tolist())
        chunks_scanned += 1
        if chunks_scanned > 10: 
            break
            
    try:
        sorted_labels = sorted([int(x) for x in labels if str(x).replace('.','',1).isdigit()])
        print(f"Labels found: {sorted_labels}")
        return sorted_labels
    except:
        print(f"[WARN] Non-numeric labels found: {labels}")
        return list(labels)

def evaluate_on_fixed_val_set(clf, vectorizer, raw_files):
    """
    始终在 Raw Data 的 Validation Split 上进行评估。
    """
    y_true = []
    y_pred = []
    
    print("  [Eval] Starting evaluation loop...", end="\r")
    chunk_count = 0
    
    # 这里我们只读取 Raw Data (也是被限制过的 10 个文件)
    for chunk in iter_dataset(raw_files):
        chunk = chunk.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        if chunk.empty: continue
        
        # 计算掩码：找出验证集数据
        mask = chunk[TEXT_COLUMN].apply(is_validation_sample)
        val_df = chunk[mask] # 只取验证集部分
        
        if val_df.empty: continue

        X_val = vectorizer.transform(val_df[TEXT_COLUMN].astype(str))
        y_val = val_df[LABEL_COLUMN].astype(int)
        
        y_pred_chunk = clf.predict(X_val)
        
        y_true.extend(y_val.tolist())
        y_pred.extend(y_pred_chunk.tolist())
        
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  [Eval] Processed {chunk_count} raw chunks...", end="\r")
            
    print(f"  [Eval] Finished. Total eval samples: {len(y_true)}        ")
    return accuracy_score(y_true, y_pred) if y_true else 0.0, len(y_true)

def train_routine(method_name: str, train_source: str | List[str], raw_files: List[str], classes: list):
    vectorizer = HashingVectorizer(n_features=2**20, alternate_sign=False, ngram_range=(1, 2))
    clf = SGDClassifier(loss="log_loss", random_state=SPLIT_SEED, n_jobs=-1) 

    print(f"--- Training {method_name} ---")
    start_time = time.time()
    train_count = 0
    
    # 1. Training Loop
    for epoch in range(EPOCHS):
        print(f"  Epoch {epoch+1}/{EPOCHS} started...")
        data_iter = iter_dataset(train_source)
        
        # 进度监控变量
        chunk_idx = 0
        epoch_start_time = time.time()

        for chunk in data_iter:
            chunk = chunk.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
            if chunk.empty: continue

            # 防止数据泄露
            is_val = chunk[TEXT_COLUMN].apply(is_validation_sample)
            train_df = chunk[~is_val] # 取【非】验证集部分

            if train_df.empty: continue

            X_train = vectorizer.transform(train_df[TEXT_COLUMN].astype(str))
            y_train = train_df[LABEL_COLUMN].astype(int)
            
            clf.partial_fit(X_train, y_train, classes=classes)
            train_count += len(train_df)
            
            # 打印进度
            chunk_idx += 1
            if chunk_idx % 5 == 0: # 每5个chunk打印一次
                elapsed = time.time() - epoch_start_time
                speed = train_count / elapsed if elapsed > 0 else 0
                print(f"    [Epoch {epoch+1}] Chunk {chunk_idx}: Processed {train_count} total samples... ({int(speed)} rows/sec)")
    
    train_time = time.time() - start_time
    
    # 2. Evaluation Loop (Fixed on Raw Data)
    print(f"--- Evaluating {method_name} on FIXED Raw Validation Set ---")
    val_acc, val_count = evaluate_on_fixed_val_set(clf, vectorizer, raw_files)
    
    return val_acc, train_time, train_count, val_count

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # 获取原始文件列表（已被截断为前10个）
    raw_files = iter_raw_files()
    
    if not raw_files:
        print("No raw files found.")
        return

    classes = collect_label_set(raw_files)
    
    results = []
    
    for name, dedup_filename in METHODS:
        if dedup_filename is None:
            train_source = raw_files 
        else:
            train_source = os.path.join(DEDUP_DIR, dedup_filename)
            
        val_acc, elapsed, train_cnt, val_cnt = train_routine(name, train_source, raw_files, classes)
        
        print(f"Result: {name} -> Acc: {val_acc:.4f}, Train Samples: {train_cnt}, Time: {elapsed:.1f}s")
        
        results.append({
            "Method": name,
            "Train Samples": train_cnt,
            "Val Samples (Fixed)": val_cnt,
            "Time(s)": f"{elapsed:.1f}",
            "Val Acc": f"{val_acc*100:.2f}%",
        })

        # 【新增：实时保存】
        # 每跑完一个方法就存一次，防止程序崩溃导致前面的结果丢失
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"  [Saved] Intermediate results saved to {OUTPUT_CSV}")

    print(f"\n[Done] All results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()