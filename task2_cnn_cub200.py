"""
任务2: 使用深度神经网络进行分类
数据集: CUB-200 鸟类数据集 (200个类别)
要求: 从头训练,不使用预训练权重
"""

import numpy as np
import os
from pathlib import Path
from PIL import Image
from ConvModulePack import *
from optimization import *
from model import Model


def load_cub200_data(data_root, image_size=(64, 64), max_samples_per_class=None):
    """
    加载CUB-200数据集
    
    参数:
        data_root: 数据根目录
        image_size: 调整后的图像尺寸 (height, width)
        max_samples_per_class: 每个类别最大样本数(用于调试)
    
    返回:
        X_train, y_train, X_val, y_val, class_names
    """
    train_dir = Path(data_root) / 'train'
    val_dir = Path(data_root) / 'val'
    
    print("=" * 80)
    print("加载CUB-200鸟类数据集")
    print("=" * 80)
    
    # 获取所有类别
    class_folders = sorted([f for f in train_dir.iterdir() if f.is_dir()])
    class_names = [f.name for f in class_folders]
    num_classes = len(class_names)
    
    print(f"类别数量: {num_classes}")
    print(f"图像尺寸: {image_size}")
    print("-" * 80)
    
    def load_images_from_split(split_dir, class_folders):
        """加载指定分割的图像"""
        images_list = []
        labels_list = []
        
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            class_path = split_dir / class_name
            
            if not class_path.exists():
                continue
            
            # 获取所有jpg图片
            image_files = list(class_path.glob("*.jpg"))
            
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            for img_file in image_files:
                try:
                    # 读取图像
                    img = Image.open(img_file).convert('RGB')
                    
                    # 调整大小
                    img = img.resize(image_size)
                    
                    # 转换为numpy数组并归一化到[0, 1]
                    img_array = np.array(img).astype('float32') / 255.0
                    
                    images_list.append(img_array)
                    labels_list.append(class_idx)
                    
                except Exception as e:
                    print(f"警告: 无法加载图像 {img_file}: {e}")
                    continue
            
            if (class_idx + 1) % 50 == 0:
                print(f"  已加载 {class_idx + 1}/{num_classes} 个类别...")
        
        images = np.array(images_list)
        labels = np.array(labels_list)
        
        return images, labels
    
    # 加载训练集
    print("加载训练集...")
    X_train, y_train = load_images_from_split(train_dir, class_folders)
    print(f"训练集: {X_train.shape[0]} 个样本, 形状: {X_train.shape}")
    
    # 加载验证集
    print("\n加载验证集...")
    X_val, y_val = load_images_from_split(val_dir, class_folders)
    print(f"验证集: {X_val.shape[0]} 个样本, 形状: {X_val.shape}")
    
    # One-hot编码
    y_train_onehot = np.eye(num_classes)[y_train]
    y_val_onehot = np.eye(num_classes)[y_val]
    
    print("\n" + "=" * 80)
    print("数据加载完成!")
    print("=" * 80)
    
    return X_train, y_train_onehot, X_val, y_val_onehot, class_names


def build_cnn_model(num_classes=200, image_size=(64, 64)):
    """
    构建卷积神经网络模型
    
    架构:
        Conv2D(3->32) -> ReLU -> MaxPool
        Conv2D(32->64) -> ReLU -> MaxPool
        Conv2D(64->128) -> ReLU -> MaxPool
        Flatten
        Dense(->512) -> ReLU
        Dense(->256) -> ReLU
        Dense(->num_classes)
    """
    model = Model()
    
    # 第一个卷积块: 64x64x3 -> 64x64x32 -> 32x32x32
    model.add(Conv2D(in_channels=3, out_channels=32, kernel_size=3))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, stride=2))
    
    # 第二个卷积块: 32x32x32 -> 32x32x64 -> 16x16x64
    model.add(Conv2D(in_channels=32, out_channels=64, kernel_size=3))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, stride=2))
    
    # 第三个卷积块: 16x16x64 -> 16x16x128 -> 8x8x128
    model.add(Conv2D(in_channels=64, out_channels=128, kernel_size=3))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, stride=2))
    
    # 展平层
    model.add(Flatten())  # 8*8*128 = 8192
    
    # 全连接层
    model.add(Dense(input_dim=8 * 8 * 128, output_dim=512))
    model.add(ReLU())
    
    model.add(Dense(input_dim=512, output_dim=256))
    model.add(ReLU())
    
    # 输出层
    model.add(Dense(input_dim=256, output_dim=num_classes))
    
    return model


def build_smaller_cnn_model(num_classes=200, image_size=(32, 32)):
    """
    构建较小的CNN模型(用于快速训练/调试)
    """
    model = Model()
    
    # 第一个卷积块: 32x32x3 -> 32x32x16 -> 16x16x16
    model.add(Conv2D(in_channels=3, out_channels=16, kernel_size=3))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, stride=2))
    
    # 第二个卷积块: 16x16x16 -> 16x16x32 -> 8x8x32
    model.add(Conv2D(in_channels=16, out_channels=32, kernel_size=3))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=2, stride=2))
    
    # 展平层
    model.add(Flatten())  # 8*8*32 = 2048
    
    # 全连接层
    model.add(Dense(input_dim=8 * 8 * 32, output_dim=256))
    model.add(ReLU())
    
    # 输出层
    model.add(Dense(input_dim=256, output_dim=num_classes))
    
    return model


def main():
    """
    主函数
    """
    # 数据路径
    data_root = Path(__file__).parent / "data" #/ "data"
    
    # 模型选择
    # 选项1: 较小模型,用于快速训练和调试
    USE_SMALL_MODEL = True
    IMAGE_SIZE = (32, 32) if USE_SMALL_MODEL else (64, 64)
    
    # 超参数
    NUM_CLASSES = 200
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # 可选: 限制每个类别的样本数(用于快速测试)
    # 设为None使用所有样本
    MAX_SAMPLES_PER_CLASS = 30  # 调试时可以设置为较小值,如10
    
    print("=" * 80)
    print("任务2: 使用深度神经网络进行CUB-200鸟类分类")
    print("=" * 80)
    print(f"模型类型: {'小型CNN' if USE_SMALL_MODEL else '标准CNN'}")
    print(f"图像尺寸: {IMAGE_SIZE}")
    print(f"类别数: {NUM_CLASSES}")
    print(f"训练轮数: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 80)
    print()
    
    # 1. 加载数据
    X_train, y_train, X_val, y_val, class_names = load_cub200_data(
        data_root, 
        image_size=IMAGE_SIZE,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    # 2. 构建模型
    print("\n构建模型...")
    if USE_SMALL_MODEL:
        model = build_smaller_cnn_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    else:
        model = build_cnn_model(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    
    print("模型架构:")
    for i, layer in enumerate(model.layers):
        print(f"  层 {i}: {layer.__class__.__name__}")
    print()
    
    # 3. 训练模型
    print("=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        print_every=10,
        save_dir="cub200_cnn_model.npz"
    )
    
    # 4. 最终评估
    print("\n" + "=" * 80)
    print("最终评估")
    print("=" * 80)
    
    # 训练集准确率
    train_pred = model.predict(X_train)
    train_pred_labels = np.argmax(train_pred, axis=1)
    train_true_labels = np.argmax(y_train, axis=1)
    train_acc = np.mean(train_pred_labels == train_true_labels)
    print(f"训练集准确率: {train_acc * 100:.2f}%")
    
    # 测试集准确率
    val_pred = model.predict(X_val)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_true_labels = np.argmax(y_val, axis=1)
    val_acc = np.mean(val_pred_labels == val_true_labels)
    print(f"测试集准确率: {val_acc * 100:.2f}%")
    
    # 5. 展示一些预测示例
    print("\n" + "-" * 80)
    print("预测示例 (前10个测试样本):")
    print("-" * 80)
    
    for i in range(min(10, len(val_pred_labels))):
        true_class = class_names[val_true_labels[i]]
        pred_class = class_names[val_pred_labels[i]]
        is_correct = "✓" if val_true_labels[i] == val_pred_labels[i] else "✗"
        print(f"样本 {i}: 真实={true_class}, 预测={pred_class} {is_correct}")
    
    print("\n" + "=" * 80)
    print("任务2完成! ✓")
    print("=" * 80)


if __name__ == '__main__':
    main()
