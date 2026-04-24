import os
import json
import librosa
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 配置区域 =================
AUDIO_DIR = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\digital_swamp_audio"
RESULT_DIR = r"D:\Deduplication_framework\2026_new_experiment\result"
# ===========================================

def get_files():
    files = []
    print("扫描音频文件...")
    for r, d, f in os.walk(AUDIO_DIR):
        for file in f:
            if file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

# --- Worker 1: Ours (Spectrogram + Hash) 单个文件处理逻辑 ---
def worker_ours(file_path):
    try:
        # 只读 4 秒，极速模式
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return file_path # 坏文件保留
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_S = librosa.power_to_db(S, ref=np.max)
        min_v, max_v = log_S.min(), log_S.max()
        
        if max_v - min_v > 0:
            img = Image.fromarray((255 * (log_S - min_v) / (max_v - min_v)).astype(np.uint8))
            h = str(imagehash.phash(img))
            return (file_path, h) # 返回 (路径, 哈希值)
        else:
            return file_path # 异常保留
    except:
        return file_path # 报错保留

# --- Worker 2: MFCC 单个文件处理逻辑 ---
def worker_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=4)
        if len(y) == 0: return None
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        feat = np.mean(mfcc.T, axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return (file_path, feat) # 返回 (路径, 特征向量)
    except:
        return None

# ================= 主程序 =================
if __name__ == "__main__":
    # Windows 下多进程必须放在 if __name__ == "__main__" 下
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_files = get_files()
    
    if not all_files:
        print("错误：没找到文件！")
        exit()

    # 获取 CPU 核心数 (留 2 个核心给系统，其他的全用上)
    num_cores = max(1, cpu_count() - 16)
    print(f"火力全开！正在使用 {num_cores} 个 CPU 核心并行处理...")

    # ----------------------------------------------------
    # 任务 1: 生成 Ours 列表
    # ----------------------------------------------------
    ours_json_path = os.path.join(RESULT_DIR, "audio_ours_keep_list.json")
    if not os.path.exists(ours_json_path):
        print(f"\n>>> [1/2] 正在生成 Ours 列表 ({len(all_files)} files)...")
        
        keep_ours = []
        seen_hashes = set()
        
        # 启动进程池
        with Pool(processes=num_cores) as pool:
            # imap_unordered 可以在结果出来时立即返回，让 tqdm 动起来
            results = list(tqdm(pool.imap(worker_ours, all_files), total=len(all_files), desc="Ours Multiprocessing"))
            
        # 汇总结果 (这步很快，单线程即可)
        for res in results:
            if isinstance(res, tuple):
                f_path, h = res
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    keep_ours.append(f_path)
            else:
                # 是单个路径（异常文件），直接保留
                keep_ours.append(res)
                
        with open(ours_json_path, 'w') as f: json.dump(keep_ours, f)
        print(f"Ours 完成！保留: {len(keep_ours)}")
    else:
        print("Ours 列表已存在，跳过。")

    # ----------------------------------------------------
    # 任务 2: 生成 MFCC 列表
    # ----------------------------------------------------
    mfcc_json_path = os.path.join(RESULT_DIR, "audio_mfcc_keep_list.json")
    if not os.path.exists(mfcc_json_path):
        print(f"\n>>> [2/2] 正在生成 MFCC 列表...")
        
        feats = []
        valid_files = []
        
        with Pool(processes=num_cores) as pool:
            # MFCC 只需要提取特征，不需要在这里做 hash 判断
            results = list(tqdm(pool.imap(worker_mfcc, all_files), total=len(all_files), desc="MFCC Extracting"))
            
        # 整理特征
        for res in results:
            if res is not None:
                valid_files.append(res[0])
                feats.append(res[1])
                
        # 计算相似度 (矩阵运算是 numpy 的强项，本身就是并行的，无需多进程)
        if feats:
            print("   正在计算矩阵 (Matrix Calculation)...")
            feats_arr = np.array(feats)
            # 简单的分块或全量矩阵
            sim_mat = np.dot(feats_arr, feats_arr.T)
            np.fill_diagonal(sim_mat, 0)
            
            to_remove = set()
            n = len(valid_files)
            for i in tqdm(range(n), desc="Filtering"):
                if i in to_remove: continue
                dups = np.where(sim_mat[i] > 0.95)[0]
                for j in dups:
                    if j > i: to_remove.add(j)
                    
            keep_mfcc = [valid_files[i] for i in range(n) if i not in to_remove]
            
            # 补回坏文件
            processed_set = set(valid_files)
            for f in all_files:
                if f not in processed_set: keep_mfcc.append(f)
                
            with open(mfcc_json_path, 'w') as f: json.dump(keep_mfcc, f)
            print(f"MFCC 完成！保留: {len(keep_mfcc)}")
    else:
        print("MFCC 列表已存在，跳过。")