import os
import random
import hashlib
import pandas as pd
from datasketch import MinHash, MinHashLSH
from simhash import Simhash
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_text"
SEED_FILE = "part_0000.csv"
TEXT_COLUMN = "content"
SAMPLE_ROWS = 10000 
OUTPUT_DEBUG_FILE = "debug_benchmark_4_methods.csv"

# 算法参数
NUM_PERM = 128
THRESHOLD = 0.8  # 相似度 > 0.8 视为重复
SIMHASH_DIST = 10 # 放宽 SimHash 阈值
# ===========================================

def load_seed_data():
    path = os.path.join(DATA_DIR, SEED_FILE)
    try:
        df = pd.read_csv(path, usecols=[TEXT_COLUMN], on_bad_lines='skip', encoding='utf-8')
        return df[TEXT_COLUMN].dropna().astype(str).tolist()[:SAMPLE_ROWS]
    except Exception as e:
        print(f"[ERROR] {e}")
        return []

def inject_noise(text):
    # 注入足以破坏 MD5 但保留 Jaccard 相似度的噪点
    noise_id = "".join(random.choices("0123456789", k=6))
    return f"{text} [noise:{noise_id}]"

def generate_fixed_dataset(seed_texts):
    print(f"[Info] Generating ordered dataset (Originals FIRST)...")
    data_list = []
    
    # 1. 先放所有的【原件】 (Keep)
    for i, t in enumerate(seed_texts):
        data_list.append({
            "id": i, "text": t, "ground_truth": "keep", "type": "original"
        })
        
    # 2. 再放所有的【副本】 (Remove)
    for i, t in enumerate(seed_texts):
        # 20% 模糊重复，80% 精确重复
        if random.random() < 0.2:
            txt = inject_noise(t)
            dup_type = "near_dup_fuzzy"
        else:
            txt = t
            dup_type = "exact_dup"
            
        data_list.append({
            "id": i, "text": txt, "ground_truth": "remove", "type": dup_type
        })
            
    # 【重要】绝对不 Shuffle，保证先来后到
    return pd.DataFrame(data_list)

# --- 1. MD5 ---
def run_md5(df):
    print("   -> Running MD5...")
    seen = set()
    preds = []
    for t in df['text']:
        val = hashlib.md5(t.lower().strip().encode('utf-8')).hexdigest()
        if val in seen: preds.append("remove")
        else:
            seen.add(val)
            preds.append("keep")
    df['pred_md5'] = preds
    return df

# --- 2. SimHash ---
def run_simhash(df):
    print("   -> Running SimHash...")
    seen_objs = [] 
    preds = []
    WINDOW = 1000 
    
    for t in tqdm(df['text'], desc="SimHash"):
        curr = Simhash(t)
        is_dup = False
        search_target = seen_objs[-WINDOW:] if len(seen_objs) > WINDOW else seen_objs
        
        for seen in search_target:
            if curr.distance(seen) <= SIMHASH_DIST:
                is_dup = True
                break
        
        if is_dup: preds.append("remove")
        else:
            seen_objs.append(curr)
            preds.append("keep")
    df['pred_simhash'] = preds
    return df

# --- 3. Standard MinHash (LSH) ---
def run_standard_minhash_lsh(df):
    print("   -> Running Std MinHash (Char-only + LSH)...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    preds = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Std(LSH)"):
        text = row['text']
        m = MinHash(num_perm=NUM_PERM)
        
        # 【特征差异点】只使用 Char-level 3-gram
        t_clean = text.lower().replace(" ", "")
        if len(t_clean) >= 3:
            for i in range(len(t_clean)-2):
                m.update(t_clean[i:i+3].encode('utf8'))
        
        result = lsh.query(m)
        if len(result) > 0:
            preds.append("remove")
        else:
            lsh.insert(f"doc_{idx}", m)
            preds.append("keep")
            
    df['pred_std_minhash'] = preds
    return df

# --- 4. Ours (Multi-Granularity + LSH) ---
def run_ours_lsh(df):
    print("   -> Running Ours (Multi-gram + LSH)...")
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    preds = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ours(LSH)"):
        text = row['text']
        m = MinHash(num_perm=NUM_PERM)
        
        # 【特征差异点】多粒度：同时使用 Word-level 和 Char-level
        # A. Word Level
        for w in text.lower().split():
            m.update(w.encode('utf8'))
        # B. Char Level
        t_clean = text.lower().replace(" ", "")
        if len(t_clean) >= 3:
            for i in range(len(t_clean)-2):
                m.update(t_clean[i:i+3].encode('utf8'))
                
        result = lsh.query(m)
        if len(result) > 0:
            preds.append("remove")
        else:
            lsh.insert(f"doc_{idx}", m)
            preds.append("keep")
            
    df['pred_ours'] = preds
    return df

def calculate_metrics(df, col):
    tp = len(df[(df['ground_truth'] == 'remove') & (df[col] == 'remove')])
    fp = len(df[(df['ground_truth'] == 'keep') & (df[col] == 'remove')])
    fn = len(df[(df['ground_truth'] == 'remove') & (df[col] == 'keep')])
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return prec, rec

if __name__ == "__main__":
    texts = load_seed_data()
    if not texts: exit()
    
    # 生成有序数据
    df = generate_fixed_dataset(texts)
    
    # 运行所有算法 (确保这里有4行)
    df = run_md5(df)
    df = run_simhash(df)
    df = run_standard_minhash_lsh(df) 
    df = run_ours_lsh(df)
    
    # 计算
    metrics = {}
    metrics['MD5'] = calculate_metrics(df, 'pred_md5')
    metrics['SimHash'] = calculate_metrics(df, 'pred_simhash')
    metrics['MinHash (Std)'] = calculate_metrics(df, 'pred_std_minhash')
    metrics['Ours (Multi)'] = calculate_metrics(df, 'pred_ours')
    
    # 输出
    print("\n" + "="*60)
    print(" FINAL COMPLETE BENCHMARK (4 Methods)")
    print("="*60)
    print(f"{'Method':<20} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 50)
    for name, (p, r) in metrics.items():
        print(f"{name:<20} | {p*100:.2f}%     | {r*100:.2f}%")
    print("="*60)
    
    df.to_csv(OUTPUT_DEBUG_FILE, index=False, encoding='utf-8-sig')