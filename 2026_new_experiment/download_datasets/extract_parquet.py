import os
import pandas as pd
import io
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# 1. Parquet 文件所在的目录 (你截图里的那个路径)
SOURCE_DIR = Path(r"D:\桌面\Deduplication_framework\2026_new_experiment\datasets\image")

# 2. 图片输出目录 (PyTorch 读取这个)
TARGET_DIR = Path(r"D:\桌面\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_pytorch")
# ===========================================

def extract_from_parquet(file_path, split_name):
    """
    读取单个 Parquet 文件并将图片保存到对应的类别文件夹中
    """
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取失败 {file_path.name}: {e}")
        return

    # 确定输出根目录 (train 或 val)
    split_dir = TARGET_DIR / split_name
    
    # 遍历每一行 (每一行是一张图片)
    # 这一步可能会花点时间，因为要逐个保存文件
    for idx, row in df.iterrows():
        try:
            # 1. 获取图片二进制数据
            # HF Parquet 通常结构是: {'bytes': b'...', 'path': '...'}
            image_data = row['image']
            if isinstance(image_data, dict) and 'bytes' in image_data:
                img_bytes = image_data['bytes']
            else:
                # 兼容其他可能的存储格式
                img_bytes = image_data

            # 2. 获取标签 (Label)
            label = row['label']
            
            # 3. 创建类别文件夹 (使用数字索引 0-999)
            # 注意：这里我们直接用数字作为文件夹名，训练效果是一样的
            class_dir = split_dir / str(label)
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. 保存图片
            # 使用文件名或简单的计数命名
            img = Image.open(io.BytesIO(img_bytes))
            
            # 转换成 RGB 防止某些灰度图/RGBA图报错
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            save_name = f"{file_path.stem}_{idx}.jpg"
            img.save(class_dir / save_name)

        except Exception as e:
            # 遇到坏图跳过，不中断程序
            continue

def main():
    print(f"源路径: {SOURCE_DIR}")
    print(f"目标路径: {TARGET_DIR}")
    
    # 1. 扫描所有 Parquet 文件
    all_files = list(SOURCE_DIR.rglob("*.parquet"))
    print(f"找到 {len(all_files)} 个 Parquet 文件")

    # 分离 train 和 validation 文件
    train_files = [f for f in all_files if 'train' in f.name]
    val_files = [f for f in all_files if 'validation' in f.name]
    
    print(f"训练集文件: {len(train_files)} 个")
    print(f"验证集文件: {len(val_files)} 个")

    # 2. 处理训练集
    if train_files:
        print("\n>>> 开始提取训练集 (Train)...")
        # 使用 tqdm 显示总进度
        for p_file in tqdm(train_files):
            extract_from_parquet(p_file, "train")

    # 3. 处理验证集
    if val_files:
        print("\n>>> 开始提取验证集 (Val)...")
        for p_file in tqdm(val_files):
            extract_from_parquet(p_file, "val")

    print("\n>>> 全部完成！")

if __name__ == "__main__":
    main()