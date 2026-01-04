import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import math
from model import CUB200Resnet18

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True 

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
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
            return torch.zeros(3, 224, 224), label


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing损失函数"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        
        # 创建平滑标签
        smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_labels * log_probs).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, optimizer, device, epoch, total_epochs, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if use_mixup and np.random.random() > 0.5:  # 50%概率使用mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (i + 1) % 50 == 0:
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr=1e-6):
    """余弦退火学习率调度器，带warmup"""
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Warmup阶段：线性增加
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        # 余弦退火阶段
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    # ==================== 配置参数 ====================
    BATCH_SIZE = 32          # 批次大小
    EPOCHS = 100             # 训练轮数
    LEARNING_RATE = 0.001    # 初始学习率
    IMAGE_SIZE = (224, 224)  # 图像尺寸
    NUM_WORKERS = 4
    WEIGHT_DECAY = 5e-4      # L2正则化
    WARMUP_EPOCHS = 5        # warmup轮数
    LABEL_SMOOTHING = 0.1    # 标签平滑
    USE_MIXUP = True         # 是否使用Mixup
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ==================== 数据增强（更强） ====================
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),                    # 先放大
        transforms.RandomCrop(IMAGE_SIZE),                # 随机裁剪到224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 随机擦除
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
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
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # # ==================== 初始化模型 ====================
    model = CUB200Resnet18(num_classes=200, input_size=IMAGE_SIZE[0]).to(DEVICE)
    
    # # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # # ==================== 损失函数和优化器 ====================
    criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    val_criterion = nn.CrossEntropyLoss()  # 验证用标准交叉熵
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 余弦退火学习率调度器
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)
    
    # ==================== 训练循环 ====================
    print("=" * 80)
    print("Start training with ResNet architecture...")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Image Size: {IMAGE_SIZE}, Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Mixup: {USE_MIXUP}, Weight Decay: {WEIGHT_DECAY}")
    print("=" * 80)
    
    best_acc = 0.0
    patience = 0
    max_patience = 20  # 早停
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} (lr={current_lr:.6f})")
        print("-" * 40)
        
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, DEVICE, 
            epoch, EPOCHS, use_mixup=USE_MIXUP
        )
        val_loss, val_acc = validate(model, val_loader, val_criterion, DEVICE)
        
        # 更新学习率
        scheduler.step()
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, "cub200_resnet18_best.pth")
            print(f"  *** Saved best model (Val Acc: {best_acc:.2f}%) ***")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("-" * 80)
    
    # ==================== 最终评估 ====================
    print("\n" + "=" * 80)
    print("Training finished!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print("=" * 80)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load("cub200_resnet18_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_loss, final_val_acc = validate(model, val_loader, val_criterion, DEVICE)
    print(f"\nFinal evaluation with best model:")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.2f}%")

if __name__ == '__main__':
    main()
