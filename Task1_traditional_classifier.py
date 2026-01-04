"""
任务1: 使用传统机器学习方法进行10分类
使用提供的特征向量(.pt文件)作为输入

包含的分类器:
- SVM (支持向量机)
- GaussianNaiveBayes (高斯朴素贝叶斯)
- KNearestNeighbors (K近邻)
- LogisticRegression (逻辑回归)
"""

import numpy as np
import torch
from pathlib import Path


# =============================================================================
# 传统分类器实现
# =============================================================================

class GaussianNaiveBayes:
    """高斯朴素贝叶斯分类器"""
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.vars = {}
        
    def fit(self, X, y):
        """训练分类器"""
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / n_samples
            self.means[cls] = np.mean(X_cls, axis=0)
            self.vars[cls] = np.var(X_cls, axis=0) + 1e-6
        
        return self
    
    def _calculate_likelihood(self, x, cls):
        """计算对数似然 log P(x|C)"""
        mean = self.means[cls]
        var = self.vars[cls]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum((x - mean) ** 2 / var)
        return log_likelihood
    
    def predict(self, X):
        """预测类别"""
        predictions = []
        for x in X:
            posteriors = {}
            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = self._calculate_likelihood(x, cls)
                posteriors[cls] = log_prior + log_likelihood
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return np.array(predictions)
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


class KNearestNeighbors:
    """K近邻分类器"""
    
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _calculate_distance(self, x1, x2):
        """计算距离"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        elif self.distance_metric == 'cosine':
            dot_product = np.sum(x1 * x2, axis=1)
            norm1 = np.sqrt(np.sum(x1 ** 2))
            norm2 = np.sqrt(np.sum(x2 ** 2, axis=1))
            return 1 - dot_product / (norm1 * norm2 + 1e-8)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict(self, X):
        """预测类别"""
        predictions = []
        for x in X:
            distances = self._calculate_distance(x, self.X_train)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(unique_labels[np.argmax(counts)])
        return np.array(predictions)
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


class LogisticRegression:
    """多分类逻辑回归 (Softmax回归)"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.classes = None
    
    def _softmax(self, z):
        """Softmax函数"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """训练分类器"""
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros((1, n_classes))
        y_one_hot = np.eye(n_classes)[y]
        
        for iteration in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(z)
            
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))
            loss += self.reg_lambda * np.sum(self.weights ** 2)
            
            dz = (y_pred - y_one_hot) / n_samples
            dw = np.dot(X.T, dz) + 2 * self.reg_lambda * self.weights
            db = np.sum(dz, axis=0, keepdims=True)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (iteration + 1) % 200 == 0:
                print(f"  迭代 {iteration + 1}/{self.n_iterations}, 损失: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """预测类别"""
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(z)
        return self.classes[np.argmax(y_pred, axis=1)]
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


class SVM:
    """支持向量机分类器 (One-vs-Rest策略, 调用sklearn求解对偶问题)"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='auto', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifiers = []
        self.classes = None
    
    def _train_binary_svm(self, X, y):
        """训练单个二分类SVM"""
        from sklearn.svm import SVC
        binary_clf = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, 
                        max_iter=self.max_iter, probability=True)
        binary_clf.fit(X, y)
        return binary_clf
    
    def fit(self, X, y):
        """训练分类器"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        print(f"训练SVM分类器 (One-vs-Rest策略)")
        print(f"类别数: {n_classes}, 样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")
        print(f"核函数: {self.kernel}, C: {self.C}, gamma: {self.gamma}")
        print("-" * 60)
        
        for i, cls in enumerate(self.classes):
            y_binary = np.where(y == cls, 1, -1)
            print(f"训练第 {i+1}/{n_classes} 个分类器 (类别 {cls})...", end=' ')
            binary_clf = self._train_binary_svm(X, y_binary)
            self.classifiers.append(binary_clf)
            print("完成")
        
        print("-" * 60)
        print("训练完成!")
        return self
    
    def predict(self, X):
        """预测类别"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        decision_values = np.zeros((n_samples, n_classes))
        
        for i, clf in enumerate(self.classifiers):
            decision_values[:, i] = clf.decision_function(X)
        
        return self.classes[np.argmax(decision_values, axis=1)]
    
    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


# =============================================================================
# 数据加载函数
# =============================================================================

def load_features_from_folder(data_root, selected_classes=None, split='train'):
    """从文件夹中加载特征向量"""
    split_dir = Path(data_root) / split
    all_class_folders = sorted([f for f in split_dir.iterdir() if f.is_dir()])
    
    if selected_classes is not None:
        class_folders = [f for f in all_class_folders if f.name in selected_classes]
    else:
        class_folders = all_class_folders
    
    if len(class_folders) == 0:
        raise ValueError("未找到任何类别文件夹!")
    
    print(f"加载 {split} 数据集...")
    print(f"选择的类别数: {len(class_folders)}")
    
    features_list = []
    labels_list = []
    class_names = []
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        class_names.append(class_name)
        pt_files = list(class_folder.glob("*.pt"))
        print(f"  类别 {class_idx}: {class_name} - {len(pt_files)} 个样本")
        
        for pt_file in pt_files:
            feature = torch.load(pt_file, weights_only=True)
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            features_list.append(feature.flatten())
            labels_list.append(class_idx)
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    print(f"加载完成: {features.shape[0]} 个样本, 特征维度: {features.shape[1]}")
    print("-" * 60)
    
    return features, labels, class_names


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数 - 任务1: 使用传统方法进行10分类"""
    data_root = Path(__file__).parent / "data" / "data"
    
    # 选择10个类别
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
    
    # 1. 加载数据
    X_train, y_train, train_class_names = load_features_from_folder(
        data_root, selected_classes, split='train'
    )
    X_test, y_test, test_class_names = load_features_from_folder(
        data_root, selected_classes, split='val'
    )
    
    # 2. 特征标准化
    print("进行特征标准化...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    print("标准化完成")
    print("-" * 60)
    
    # 3. 训练SVM分类器 (RBF核)
    print("\n开始训练SVM分类器...")
    svm_rbf = SVM(C=10.0, kernel='rbf', gamma='auto', max_iter=1000)
    svm_rbf.fit(X_train, y_train)
    
    # 4. 评估模型
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    train_acc = svm_rbf.score(X_train, y_train)
    print(f"训练集准确率: {train_acc * 100:.2f}%")
    test_acc = svm_rbf.score(X_test, y_test)
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    
    # 5. 每个类别的预测情况
    print("\n" + "-" * 60)
    print("每个类别的预测情况:")
    print("-" * 60)
    y_pred = svm_rbf.predict(X_test)
    for class_idx in range(len(selected_classes)):
        class_mask = (y_test == class_idx)
        class_samples = np.sum(class_mask)
        correct = np.sum((y_pred == class_idx) & (y_test == class_idx))
        class_acc = correct / class_samples * 100 if class_samples > 0 else 0.0
        print(f"类别 {class_idx} ({train_class_names[class_idx]}): "
              f"{correct}/{class_samples} 正确, 准确率: {class_acc:.2f}%")
    
    print("=" * 60)
    print("任务1完成! ✓")
    print("=" * 60)
    
    # 6. 对比不同核函数
    print("\n\n" + "=" * 60)
    print("额外实验: 对比不同核函数的性能")
    print("=" * 60)
    
    print("\n训练线性核SVM...")
    svm_linear = SVM(C=1.0, kernel='linear', max_iter=1000)
    svm_linear.fit(X_train, y_train)
    linear_test_acc = svm_linear.score(X_test, y_test)
    print(f"线性核测试集准确率: {linear_test_acc * 100:.2f}%")
    
    print("\n训练多项式核SVM...")
    svm_poly = SVM(C=1.0, kernel='poly', gamma='auto', max_iter=1000)
    svm_poly.fit(X_train, y_train)
    poly_test_acc = svm_poly.score(X_test, y_test)
    print(f"多项式核测试集准确率: {poly_test_acc * 100:.2f}%")
    
    # 7. 其他传统方法
    print("\n\n" + "=" * 60)
    print("额外实验: 其他传统机器学习方法")
    print("=" * 60)
    
    print("\n训练朴素贝叶斯分类器...")
    nb_clf = GaussianNaiveBayes()
    nb_clf.fit(X_train, y_train)
    nb_test_acc = nb_clf.score(X_test, y_test)
    print(f"朴素贝叶斯测试集准确率: {nb_test_acc * 100:.2f}%")
    
    print("\n训练K近邻分类器 (K=5)...")
    knn_clf = KNearestNeighbors(k=5, distance_metric='euclidean')
    knn_clf.fit(X_train, y_train)
    knn_test_acc = knn_clf.score(X_test, y_test)
    print(f"K近邻测试集准确率: {knn_test_acc * 100:.2f}%")
    
    print("\n训练逻辑回归分类器...")
    lr_clf = LogisticRegression(learning_rate=0.1, n_iterations=1000, reg_lambda=0.01)
    lr_clf.fit(X_train, y_train)
    lr_test_acc = lr_clf.score(X_test, y_test)
    print(f"逻辑回归测试集准确率: {lr_test_acc * 100:.2f}%")
    
    # 8. 总结
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


if __name__ == '__main__':
    main()
