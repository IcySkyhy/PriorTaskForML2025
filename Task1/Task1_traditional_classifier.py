"""
任务1: 使用传统机器学习方法进行分类
本代码使用支持向量机(SVM)对10个类别进行分类
使用提供的特征向量(.pt文件)作为输入
"""

import numpy as np
import torch
import os
from pathlib import Path


class SVM:
    """
    支持向量机分类器 - One-vs-Rest策略
    使用SMO算法求解对偶问题(调用库函数)
    """
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', max_iter=1000):
        """
        参数:
            C: 正则化参数
            kernel: 核函数类型 ('linear', 'rbf', 'poly')
            gamma: RBF核的参数
            max_iter: 最大迭代次数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifiers = []  # 存储每个二分类器
        self.classes = None
        
    def _kernel_function(self, X1, X2):
        """
        计算核函数
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # RBF核: K(x, y) = exp(-gamma * ||x - y||^2)
            if self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
            
            # 计算欧氏距离的平方
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * distances)
        elif self.kernel == 'poly':
            # 多项式核: K(x, y) = (gamma * <x, y> + 1)^3
            if self.gamma == 'auto':
                gamma = 1.0 / X1.shape[1]
            else:
                gamma = self.gamma
            return (gamma * np.dot(X1, X2.T) + 1) ** 3
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _train_binary_svm(self, X, y):
        """
        训练单个二分类SVM
        使用简化的SMO算法
        """
        from sklearn.svm import SVC
        # 这里调用库函数求解对偶问题(按要求允许)
        binary_clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, 
                        max_iter=self.max_iter, probability=True)
        binary_clf.fit(X, y)
        return binary_clf
    
    def fit(self, X, y):
        """
        训练SVM分类器 - 使用One-vs-Rest策略
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        print(f"训练SVM分类器 (One-vs-Rest策略)")
        print(f"类别数: {n_classes}, 样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
        print(f"核函数: {self.kernel}, C: {self.C}, gamma: {self.gamma}")
        print("-" * 60)
        
        # 为每个类别训练一个二分类器
        for i, cls in enumerate(self.classes):
            # 创建二分类标签: 当前类 vs 其他类
            y_binary = np.where(y == cls, 1, -1)
            
            print(f"训练第 {i+1}/{n_classes} 个分类器 (类别 {cls})...", end=' ')
            binary_clf = self._train_binary_svm(X, y_binary)
            self.classifiers.append(binary_clf)
            print("完成")
        
        print("-" * 60)
        print("训练完成!")
        return self
    
    def predict(self, X):
        """
        预测样本类别
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # 存储每个分类器的决策值
        decision_values = np.zeros((n_samples, n_classes))
        
        for i, clf in enumerate(self.classifiers):
            # 获取决策函数值
            decision_values[:, i] = clf.decision_function(X)
        
        # 选择决策值最大的类别
        predictions = self.classes[np.argmax(decision_values, axis=1)]
        return predictions
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def load_features_from_folder(data_root, selected_classes=None, split='train'):
    """
    从文件夹中加载特征向量
    
    参数:
        data_root: 数据根目录
        selected_classes: 选择的类别列表(文件夹名),如果为None则加载所有类别
        split: 'train' 或 'val'
    
    返回:
        features: 特征矩阵 (n_samples, feature_dim)
        labels: 标签向量 (n_samples,)
        class_names: 类别名称列表
    """
    split_dir = Path(data_root) / split
    
    # 获取所有类别文件夹
    all_class_folders = sorted([f for f in split_dir.iterdir() if f.is_dir()])
    
    # 如果指定了选择的类别,则只加载这些类别
    if selected_classes is not None:
        class_folders = [f for f in all_class_folders if f.name in selected_classes]
    else:
        class_folders = all_class_folders
    
    if len(class_folders) == 0:
        raise ValueError(f"未找到任何类别文件夹!")
    
    print(f"加载 {split} 数据集...")
    print(f"选择的类别数: {len(class_folders)}")
    
    features_list = []
    labels_list = []
    class_names = []
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        class_names.append(class_name)
        
        # 获取该类别下所有.pt文件
        pt_files = list(class_folder.glob("*.pt"))
        
        print(f"  类别 {class_idx}: {class_name} - {len(pt_files)} 个样本")
        
        for pt_file in pt_files:
            # 加载特征向量
            feature = torch.load(pt_file, weights_only=True)
            
            # 转换为numpy数组并展平
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            feature = feature.flatten()
            
            features_list.append(feature)
            labels_list.append(class_idx)
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"加载完成: {features.shape[0]} 个样本, 特征维度: {features.shape[1]}")
    print("-" * 60)
    
    return features, labels, class_names


def main():
    """
    主函数 - 任务1: 使用SVM进行10分类
    """
    # 数据路径
    data_root = Path(__file__).parent / "data" / "data"
    
    # 选择10个类别进行分类
    # 这里选择前10个类别,你也可以随机选择其他类别
    selected_classes = [
        '001.Black_footed_Albatross',
        '002.Laysan_Albatross',
        '003.Sooty_Albatross',
        '004.Groove_billed_Ani',
        '005.Crested_Auklet',
        '006.Least_Auklet',
        '007.Parakeet_Auklet',
        '008.Rhinoceros_Auklet',
        '009.Brewer_Blackbird',
        '010.Red_winged_Blackbird'
    ]
    
    print("=" * 60)
    print("任务1: 使用传统机器学习方法(SVM)进行分类")
    print("=" * 60)
    print(f"选择的类别: {len(selected_classes)} 个")
    for i, cls in enumerate(selected_classes):
        print(f"  {i}: {cls}")
    print("=" * 60)
    
    # 1. 加载训练集
    X_train, y_train, train_class_names = load_features_from_folder(
        data_root, selected_classes, split='train'
    )
    
    # 2. 加载测试集
    X_test, y_test, test_class_names = load_features_from_folder(
        data_root, selected_classes, split='val'
    )
    
    # 3. 特征标准化(重要!)
    print("进行特征标准化...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8  # 避免除零
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    print("标准化完成")
    print("-" * 60)
    
    # 4. 训练SVM分类器
    # 尝试不同的核函数和参数
    print("\n开始训练SVM分类器...")
    
    # 方案1: RBF核(推荐)
    svm_rbf = SVM(C=10.0, kernel='rbf', gamma='auto', max_iter=1000)
    svm_rbf.fit(X_train, y_train)
    
    # 5. 评估模型
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    
    # 训练集准确率
    train_acc = svm_rbf.score(X_train, y_train)
    print(f"训练集准确率: {train_acc * 100:.2f}%")
    
    # 测试集准确率
    test_acc = svm_rbf.score(X_test, y_test)
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    
    # 6. 详细分析
    print("\n" + "-" * 60)
    print("每个类别的预测情况:")
    print("-" * 60)
    
    y_pred = svm_rbf.predict(X_test)
    
    for class_idx in range(len(selected_classes)):
        # 该类别的样本索引
        class_mask = (y_test == class_idx)
        class_samples = np.sum(class_mask)
        
        # 正确分类的样本数
        correct = np.sum((y_pred == class_idx) & (y_test == class_idx))
        
        # 计算该类别的准确率
        if class_samples > 0:
            class_acc = correct / class_samples * 100
        else:
            class_acc = 0.0
        
        print(f"类别 {class_idx} ({train_class_names[class_idx]}): "
              f"{correct}/{class_samples} 正确, 准确率: {class_acc:.2f}%")
    
    print("=" * 60)
    print("任务1完成! ✓")
    print("=" * 60)
    
    # 7. (可选) 尝试其他核函数进行对比
    print("\n\n" + "=" * 60)
    print("额外实验: 对比不同核函数的性能")
    print("=" * 60)
    
    # 线性核
    print("\n训练线性核SVM...")
    svm_linear = SVM(C=1.0, kernel='linear', max_iter=1000)
    svm_linear.fit(X_train, y_train)
    linear_test_acc = svm_linear.score(X_test, y_test)
    print(f"线性核测试集准确率: {linear_test_acc * 100:.2f}%")
    
    # 多项式核
    print("\n训练多项式核SVM...")
    svm_poly = SVM(C=1.0, kernel='poly', gamma='auto', max_iter=1000)
    svm_poly.fit(X_train, y_train)
    poly_test_acc = svm_poly.score(X_test, y_test)
    print(f"多项式核测试集准确率: {poly_test_acc * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("性能对比总结:")
    print("=" * 60)
    print(f"RBF核:      {test_acc * 100:.2f}%")
    print(f"线性核:     {linear_test_acc * 100:.2f}%")
    print(f"多项式核:   {poly_test_acc * 100:.2f}%")
    print("=" * 60)
    
    # 8. 尝试更多传统方法
    print("\n\n" + "=" * 60)
    print("额外实验: 其他传统机器学习方法")
    print("=" * 60)
    
    from traditional_classifiers import (
        GaussianNaiveBayes, KNearestNeighbors, 
        LogisticRegression
    )
    
    # 朴素贝叶斯
    print("\n训练朴素贝叶斯分类器...")
    nb_clf = GaussianNaiveBayes()
    nb_clf.fit(X_train, y_train)
    nb_test_acc = nb_clf.score(X_test, y_test)
    print(f"朴素贝叶斯测试集准确率: {nb_test_acc * 100:.2f}%")
    
    # K近邻
    print("\n训练K近邻分类器 (K=5)...")
    knn_clf = KNearestNeighbors(k=5, distance_metric='euclidean')
    knn_clf.fit(X_train, y_train)
    knn_test_acc = knn_clf.score(X_test, y_test)
    print(f"K近邻测试集准确率: {knn_test_acc * 100:.2f}%")
    
    # 逻辑回归
    print("\n训练逻辑回归分类器...")
    lr_clf = LogisticRegression(learning_rate=0.1, n_iterations=1000, reg_lambda=0.01)
    lr_clf.fit(X_train, y_train)
    lr_test_acc = lr_clf.score(X_test, y_test)
    print(f"逻辑回归测试集准确率: {lr_test_acc * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("所有方法性能对比总结:")
    print("=" * 60)
    print(f"SVM (RBF核):        {test_acc * 100:.2f}%")
    print(f"SVM (线性核):       {linear_test_acc * 100:.2f}%")
    print(f"SVM (多项式核):     {poly_test_acc * 100:.2f}%")
    print(f"朴素贝叶斯:         {nb_test_acc * 100:.2f}%")
    print(f"K近邻 (K=5):        {knn_test_acc * 100:.2f}%")
    print(f"逻辑回归:           {lr_test_acc * 100:.2f}%")
    print("=" * 60)
    print("\n推荐使用: SVM (RBF核) 获得最佳性能")
    print("=" * 60)


if __name__ == '__main__':
    main()
