import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil
import time
import multiprocessing
from PIL import Image  # 使用 Pillow 来进行安全的保存

# ================= 配置区域 =================
# 1. 干净数据的输入路径
SOURCE_DIR = Path(r"D:\桌面\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_pytorch\train")

# 2. 脏数据的输出路径
TARGET_DIR = Path(r"D:\桌面\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated\train")

# 3. 设置为 False 全量运行
DRY_RUN = False

# 4. 并行进程数 (进一步降低并发，防止内存溢出导致的进程死亡)
cpu_count = multiprocessing.cpu_count()
NUM_WORKERS = max(1, cpu_count - 6) # 留更多核心给系统
# ===========================================

def cv2_imread(file_path):
    """读取图片：保持用 OpenCV 读取，因为这里目前很稳定"""
    try:
        stream = np.fromfile(str(file_path), dtype=np.uint8)
        if stream.size == 0: return None
        cv_img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return cv_img
    except Exception:
        return None

def safe_save_pil(file_path, cv_img, quality=None):
    """
    [核心修复] 使用 Pillow 保存图片
    Pillow 不会像 OpenCV 那样发生 C++ 级别的崩溃，且原生支持中文路径。
    """
    try:
        # 1. 基础检查
        if cv_img is None or cv_img.size == 0 or cv_img.shape[0] == 0 or cv_img.shape[1] == 0:
            return False

        # 2. 颜色空间转换 (OpenCV BGR -> Pillow RGB)
        # 如果不转，保存出来的图片颜色会变蓝
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # 3. 转为 Pillow 对象
        pil_img = Image.fromarray(img_rgb)
        
        # 4. 保存
        # optimized=True 和 quality 参数可以模拟 JPEG 压缩
        save_kwargs = {}
        if str(file_path).lower().endswith(('.jpg', '.jpeg')):
            save_kwargs['quality'] = quality if quality else 75
        
        # Pillow 支持中文路径，直接传字符串即可
        pil_img.save(str(file_path), **save_kwargs)
        return True
    except Exception:
        # Pillow 如果报错会在这里被捕获，不会炸掉进程
        return False

def safe_read_image(img_path):
    try:
        img = cv2_imread(img_path)
        if img is None: return None
        
        # 强制转 3 通道
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    except Exception:
        return None

def add_noise(image):
    try:
        row, col, ch = image.shape
        mean = 0
        var = 50 
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        # 确保数据类型安全转换
        return np.clip(noisy, 0, 255).astype(np.uint8)
    except Exception:
        return image

def process_single_class(class_dir):
    try:
        class_name = class_dir.name
        target_class_path = TARGET_DIR / class_name
        
        # [断点续传] 如果文件夹已存在且有内容，跳过
        if target_class_path.exists():
             # 简单检查：如果里面文件数量超过 0，就当做完成了
             # 这样比遍历所有文件快得多
             if any(target_class_path.iterdir()):
                 return 0 

        target_class_path.mkdir(parents=True, exist_ok=True)
        
        images = list(class_dir.glob("*.[jJ][pP][gG]")) + list(class_dir.glob("*.[jJ][pP][eE][gG]"))
        processed_count = 0
        
        for img_path in images:
            try:
                img = safe_read_image(img_path)
                if img is None: continue
                
                stem = img_path.stem
                
                # 1. 保存原图 (Copy 最稳)
                shutil.copy2(img_path, target_class_path / img_path.name)
                
                # 2. 污染 A: 缩放
                h, w = img.shape[:2]
                new_h, new_w = int(h*0.8), int(w*0.8)
                if new_h > 0 and new_w > 0:
                    img_resized = cv2.resize(img, (new_w, new_h))
                    # 使用 Pillow 保存 .png
                    safe_save_pil(target_class_path / f"{stem}_aug_scale.png", img_resized)
                
                # 3. 污染 B: 加噪声
                img_noise = add_noise(img)
                # 使用 Pillow 保存 .jpg (quality=50 模拟低画质)
                safe_save_pil(target_class_path / f"{stem}_aug_noise.jpg", img_noise, quality=50)
                
                processed_count += 1
                
            except Exception:
                continue

        return processed_count
    except Exception:
        return 0

def main():
    print(">>> [V5 Pillow 稳健版] 正在扫描进度...")
    
    # 扫描任务
    all_class_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    
    # 快速扫描已完成的文件夹
    existing_classes = set()
    if TARGET_DIR.exists():
        existing_classes = {d.name for d in TARGET_DIR.iterdir() if d.is_dir()}
    
    # 过滤任务
    todo_dirs = []
    skipped_count = 0
    
    # 这里做一个稍微细致的检查：如果目标文件夹虽然存在但为空，还是得重跑
    # 为了速度，我们只检查是否存在。如果之前崩了，建议手动删掉最后一个文件夹
    for d in all_class_dirs:
        if d.name in existing_classes:
            # 如果想更严谨，可以检查目标文件夹是不是空的
            target_subdir = TARGET_DIR / d.name
            if any(target_subdir.iterdir()):
                skipped_count += 1
                continue
        todo_dirs.append(d)
    
    print(f"    总类别数: {len(all_class_dirs)}")
    print(f"    已完成:   {skipped_count} (直接跳过)")
    print(f"    待处理:   {len(todo_dirs)}")

    if len(todo_dirs) == 0:
        print("所有任务已完成！")
        return

    print("\n>>> 开始运行 (使用 Pillow 保存，拒绝崩溃)...")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_class, todo_dirs), total=len(todo_dirs), unit="class"))

    print("\n全部完成！")

if __name__ == "__main__":
    main()