import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

# ================= Configuration =================
# 1. Dataset Root Directories
DEDUPED_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_deduped_image_datasets"
RAW_ROOT = r"D:\Deduplication_framework\2026_new_experiment\datasets\final_swamp_data\imagenet_bloated"

# 2. Target Tasks (Must match the folder names created in the previous step)
TARGET_DIRS = [
    ("No Dedup",       RAW_ROOT),  # Baseline
    ("MD5 Hash",       os.path.join(DEDUPED_ROOT, "image_md5_deduped")),
    ("pHash",          os.path.join(DEDUPED_ROOT, "image_phash_deduped")),
    ("SimCLR",         os.path.join(DEDUPED_ROOT, "image_simclr_deduped")),
    ("SemDeDup",       os.path.join(DEDUPED_ROOT, "image_semdedup_deduped")),
    ("Ours (CLIP)",    os.path.join(DEDUPED_ROOT, "image_ours_deduped")),
]

# 3. Training Parameters
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================

def train_one_session(name, path):
    print(f"\n[START] Training on: {name}")
    if not os.path.exists(path):
        print(f"[ERROR] Path not found: {path}")
        return "N/A"

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Use ImageFolder to automatically load images (assumes root/class/xxx.jpg structure)
        dataset = ImageFolder(root=path, transform=transform)
    except:
        print("[ERROR] Failed to load ImageFolder. Check directory structure.")
        return "Error"

    print(f"   Samples: {len(dataset)} | Classes: {len(dataset.classes)}")
    
    if len(dataset) < 100:
        return "Too Small"

    # 80/20 Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model: ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Testing Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"[DONE] [{name}] Result: {acc:.2f}%")
    return f"{acc:.2f}%"

if __name__ == "__main__":
    results = {}
    print(f"[INIT] Start Image Downstream Training (Device: {DEVICE})")
    
    for name, path in TARGET_DIRS:
        acc = train_one_session(name, path)
        results[name] = acc
        
    print("\n" + "="*50)
    print(f"{'Method':<20} | {'Test Acc'}")
    print("-" * 35)
    for name, acc in results.items():
        print(f"{name:<20} | {acc}")
    print("="*50)