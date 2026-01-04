import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from model import CUB200CNN

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, max_samples=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        print(f"Loading {split} data from {self.root_dir}...")
        
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
                
            img_files = list(cls_dir.glob("*.jpg"))
            if max_samples:
                img_files = img_files[:max_samples]
                
            for img_path in img_files:
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[cls_name])
                
        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 128, 128), label

def train(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{total_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    # ==================== 配置参数 ====================
    BATCH_SIZE = 32          # 批次大小
    EPOCHS = 50              # 训练轮数（增加到50）
    LEARNING_RATE = 0.001    # 初始学习率
    IMAGE_SIZE = (128, 128)  # 图像尺寸（增加到128x128）
    NUM_WORKERS = 4          # 数据加载线程数
    WEIGHT_DECAY = 1e-4      # L2正则化
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ==================== 数据增强 ====================
    train_transform = transforms.Compose([
        transforms.Resize((int(IMAGE_SIZE[0] * 1.1), int(IMAGE_SIZE[1] * 1.1))),  # 稍微放大
        transforms.RandomCrop(IMAGE_SIZE),      # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        transforms.RandomRotation(15),          # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ==================== 加载数据 ====================
    data_root = Path(__file__).parent / "data"
    
    train_dataset = CUB200Dataset(data_root, split='train', transform=train_transform)
    val_dataset = CUB200Dataset(data_root, split='val', transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # ==================== 初始化模型 ====================
    model = CUB200CNN(num_classes=200, input_size=IMAGE_SIZE[0]).to(DEVICE)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== 损失函数和优化器 ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器：每15个epoch降低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # ==================== 训练循环 ====================
    print("=" * 80)
    print("Start training...")
    print("=" * 80)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} (lr={current_lr:.6f})")
        print("-" * 40)
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, "cub200_cnn_best.pth")
            print(f"  *** Saved best model (Val Acc: {best_acc:.2f}%) ***")
        
        print("-" * 80)
    
    # ==================== 最终评估 ====================
    print("\n" + "=" * 80)
    print("Training finished!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("=" * 80)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load("cub200_cnn_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, DEVICE)
    print(f"\nFinal evaluation with best model:")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.2f}%")

if __name__ == '__main__':
    main()