import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
# 数据集根目录
DATASET_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_deduped_datasets"

# 要测试的三个数据集文件夹名称
TARGET_DIRS = [
    "audio_md5_deduped",
    "audio_ours_deduped", 
    "audio_mfcc_deduped"
]

# 训练参数
BATCH_SIZE = 32
EPOCHS = 10           # 5 epochs is enough to see the trend
LEARNING_RATE = 0.001
# ===============================================

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # 1. 尝试检测是否有子文件夹 (标准的 ImageFolder 结构)
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        if len(subdirs) > 0:
            # 这里的 subdirs 就是类别名
            self.classes = sorted(subdirs)
            class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for cls_name in subdirs:
                cls_folder = os.path.join(root_dir, cls_name)
                for f in os.listdir(cls_folder):
                    if f.endswith('.wav'):
                        self.samples.append((os.path.join(cls_folder, f), class_to_idx[cls_name]))
        else:
            # 2. 扁平结构，尝试从 ESC-50 文件名解析类别
            # 格式: fs-id-src-class.wav (例如: 1-100032-A-0.wav, 最后一个 0 是类别)
            # 我们的去重文件可能长这样: copy_0_1-137-A-32.wav
            files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]
            
            class_ids = set()
            temp_samples = []
            
            for f in files:
                try:
                    # 去掉 .wav
                    name_no_ext = os.path.splitext(f)[0]
                    # 用 '-' 分割，取最后一部分作为类别 ID
                    parts = name_no_ext.split('-')
                    label_str = parts[-1] 
                    
                    # 确保是数字
                    if label_str.isdigit():
                        label = int(label_str)
                        class_ids.add(label)
                        temp_samples.append((os.path.join(root_dir, f), label))
                except:
                    pass
            
            self.classes = sorted(list(class_ids))
            # 将真实的 label ID 映射到 0..N 的索引
            real_label_map = {lbl: i for i, lbl in enumerate(self.classes)}
            self.samples = [(p, real_label_map[l]) for p, l in temp_samples]

        print(f"[INFO] Loaded: {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # 加载音频 -> 声谱图
            # 限制时长 4秒
            y, sr = librosa.load(path, sr=16000, duration=4)
            if len(y) < 16000*4: # Pad
                y = np.pad(y, (0, 16000*4 - len(y)))
            else:
                y = y[:16000*4]
                
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            
            # 归一化到 0-255 并转 RGB
            min_v, max_v = log_S.min(), log_S.max()
            if max_v - min_v > 0:
                img_arr = (255 * (log_S - min_v) / (max_v - min_v)).astype(np.uint8)
            else:
                img_arr = np.zeros((128, int(16000*4/512)+1), dtype=np.uint8)

            img = Image.fromarray(img_arr).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        except Exception as e:
            # 遇到坏数据返回全黑图，防止训练中断
            return torch.zeros(3, 224, 224), label

def train_one_dataset(folder_name):
    full_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.exists(full_path):
        print(f"[ERROR] Folder not found: {folder_name}, skipping.")
        return "N/A"

    print(f"\n[START] Training on [{folder_name}] ...")
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 构建数据集
    dataset = AudioSpectrogramDataset(full_path, transform=transform)
    
    # 检查数据量
    if len(dataset) < 50:
        print("[WARNING] Dataset too small (<50), skipping training.")
        return "0.00%"
    
    # 检查类别数
    if len(dataset.classes) < 2:
        print("[WARNING] Less than 2 classes found, cannot perform classification.")
        return "0.00%"

    # 划分 80% 训练 / 20% 测试
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 加载模型 (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # 修改全连接层适配当前数据集的实际类别数
    num_ftrs = model.fc.in_features
    actual_num_classes = len(dataset.classes)
    model.fc = nn.Linear(num_ftrs, actual_num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # 测试循环
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total > 0:
        acc = 100 * correct / total
    else:
        acc = 0.0
        
    print(f"[DONE] [{folder_name}] Final Test Acc: {acc:.2f}%")
    return f"{acc:.2f}%"

if __name__ == "__main__":
    results = {}
    print(f"[Init] Starting Downstream Training (Device: {device})")
    print("="*60)
    
    for folder in TARGET_DIRS:
        acc = train_one_dataset(folder)
        results[folder] = acc
        
    print("\n" + "="*60)
    print("=== FINAL RESULTS ===")
    print("="*60)
    print(f"{'Dataset':<25} | {'Test Acc'}")
    print("-" * 40)
    for name, acc in results.items():
        print(f"{name:<25} | {acc}")
    print("="*60)