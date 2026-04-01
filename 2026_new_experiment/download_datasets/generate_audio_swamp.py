import os
import random
import json
import time
import numpy as np
import soundfile as sf
import requests
import zipfile
import io
from pathlib import Path

# ================= 配置路径 =================
BASE_DIR = r"D:\桌面\Deduplication_framework\2026_new_experiment\datasets"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_swamp_data", "digital_swamp_audio")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth", "audio_ground_truth.json")
TEMP_DIR = os.path.join(BASE_DIR, "temp_esc50") # 临时解压目录

# 目标设定：只扩大 10 倍 (约 6GB)
UPSCALE_FACTOR = 10    
NEAR_DUP_RATIO = 0.2   

# ESC-50 官方下载直链 (只有 600MB，非常快)
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

ground_truth_log = {
    "exact_duplicates": [],
    "near_duplicates": [],
    "dataset_source": "ESC-50 (Direct Download)"
}

def download_and_extract_esc50():
    """直接从 GitHub 下载并解压，避开 HuggingFace 的依赖坑"""
    print(f"[阶段三] 正在从官方源下载 ESC-50 ({ESC50_URL})...")
    try:
        r = requests.get(ESC50_URL, stream=True)
        if r.status_code == 200:
            print("   下载成功，正在解压...")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(TEMP_DIR)
            print("   [成功] 解压完成！")
            
            # 找到音频文件夹路径
            # 解压后通常在: temp_esc50/ESC-50-master/audio/
            audio_dir = list(Path(TEMP_DIR).rglob("audio"))[0]
            return list(audio_dir.glob("*.wav"))
        else:
            print(f"   [失败] 下载失败，状态码: {r.status_code}")
            exit(1)
    except Exception as e:
        print(f"   [错误] 下载或解压出错: {e}")
        print("   提示: 如果下载太慢，你可以手动下载 ESC-50 master.zip 解压到 temp_esc50 文件夹。")
        exit(1)

def process_audio_variant(src_path, mode, save_path):
    try:
        # 使用 soundfile 直接读取 wav，无需复杂解码器
        data, sr = sf.read(src_path)
        
        # 确保 float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            
        if mode == "exact":
            sf.write(save_path, data, sr)
            return "exact_duplicates"
            
        elif mode == "noise_attack":
            # 加噪
            noise_amp = 0.005 * np.random.uniform() * np.max(np.abs(data))
            noise = noise_amp * np.random.normal(size=data.shape)
            audio_noisy = data + noise
            sf.write(save_path, audio_noisy, sr)
            return "near_duplicates"
            
    except Exception as e:
        print(f"处理出错 {src_path}: {e}")
        return None
    return None

# --- 主流程 ---

print(f"[开始] 开始构建音频沼泽 (直连版)...")

# 1. 获取种子文件列表
wav_files = download_and_extract_esc50()
print(f"   [就绪] 种子池就绪: 找到 {len(wav_files)} 个 .wav 文件")

start_time = time.time()
current_size = 0
file_count = 0

print(f"   [执行] 开始 {UPSCALE_FACTOR} 倍裂变...")

# 2. 扩容循环
for round_idx in range(UPSCALE_FACTOR):
    print(f"   正在执行第 {round_idx + 1} / {UPSCALE_FACTOR} 轮复制...")
    
    for wav_path in wav_files:
        filename = wav_path.name
        
        # 构造新文件名
        base_name = f"r{round_idx+1:02d}_{filename}" 
        save_path = os.path.join(OUTPUT_DIR, base_name)
        
        # 20% 概率加噪
        if random.random() < NEAR_DUP_RATIO:
            mode = "noise_attack"
        else:
            mode = "exact"
            
        res = process_audio_variant(wav_path, mode, save_path)
        
        if res:
            ground_truth_log[res].append(base_name)
            if os.path.exists(save_path):
                current_size += os.path.getsize(save_path)
            file_count += 1
            
    # 每轮汇报一次
    gb_size = current_size / (1024**3)
    print(f"   >>> 第 {round_idx + 1} 轮结束。当前总大小: {gb_size:.2f} GB")

# 保存记录
with open(GROUND_TRUTH_PATH, "w") as f:
    json.dump(ground_truth_log, f, indent=4)

end_time = time.time()
print(f"\n[完成] 音频模态构建完毕！")
print(f"数据位置: {OUTPUT_DIR}")
print(f"总文件数: {file_count}")
print(f"总大小: {current_size / (1024**3):.2f} GB")
print(f"耗时: {(end_time - start_time)/60:.2f} 分钟")

# 清理临时文件 (可选)
# import shutil
# shutil.rmtree(TEMP_DIR)