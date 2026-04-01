import os
from datasets import load_dataset
import pandas as pd

SEED_DIR = "D:/桌面/Deduplication_framework/2026_new_experiment/datasets/text_seed_data"
os.makedirs(SEED_DIR, exist_ok=True)

print("下载文本数据集的种子数据")
dataset = load_dataset("amazon_polarity", split="train")

seed_file_path = os.path.join(SEED_DIR, "amazon_reviews_seed.csv")
print(f"正在将数据写入 {seed_file_path} ...")

df = dataset.to_pandas()
df.to_csv(seed_file_path, index=False)

print("数据集下载并保存完成。")