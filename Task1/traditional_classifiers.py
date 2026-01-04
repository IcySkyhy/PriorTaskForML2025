"""
任务1的补充: 额外实现的传统分类方法
包括朴素贝叶斯、K近邻、逻辑回归等方法
用于对比不同传统方法的性能
"""

import numpy as np
from scipy.stats import multivariate_normal


class GaussianNaiveBayes:
    """
    高斯朴素贝叶斯分类器
    假设每个特征在给定类别下服从高斯分布
    """
    def __init__(self):
        self.classes = None
        self.class_priors = {}  # 先验概率 P(C)
        self.means = {}  # 每个类别每个特征的均值
        self.vars = {}   # 每个类别每个特征的方差
        
    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器
        """
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        for cls in self.classes:
            # 获取该类别的所有样本
            X_cls = X[y == cls]
            
            # 计算先验概率 P(C)
            self.class_priors[cls] = len(X_cls) / n_samples
            
            # 计算每个特征的均值和方差
            self.means[cls] = np.mean(X_cls, axis=0)
            self.vars[cls] = np.var(X_cls, axis=0) + 1e-6  # 加小值避免方差为0
        
        return self
    
    def _calculate_likelihood(self, x, cls):
        """
        计算似然概率 P(x|C)
        在朴素贝叶斯假设下,各特征独立:
        P(x|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)
        """
        mean = self.means[cls]
        var = self.vars[cls]
        
        # 计算对数似然(避免数值下溢)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum((x - mean) ** 2 / var)
        
        return log_likelihood
    
    def predict(self, X):
        """
        预测样本类别
        """
        predictions = []
        
        for x in X:
            # 计算后验概率 P(C|x) ∝ P(x|C) * P(C)
            posteriors = {}
            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = self._calculate_likelihood(x, cls)
                posteriors[cls] = log_prior + log_likelihood
            
            # 选择后验概率最大的类别
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class KNearestNeighbors:
    """
    K近邻分类器
    基于K个最近邻居的投票进行分类
    """
    def __init__(self, k=5, distance_metric='euclidean'):
        """
        参数:
            k: 邻居数量
            distance_metric: 距离度量 ('euclidean', 'manhattan', 'cosine')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        训练KNN(实际上只是存储训练数据)
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def _calculate_distance(self, x1, x2):
        """
        计算两个样本之间的距离
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        elif self.distance_metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            dot_product = np.sum(x1 * x2, axis=1)
            norm1 = np.sqrt(np.sum(x1 ** 2))
            norm2 = np.sqrt(np.sum(x2 ** 2, axis=1))
            return 1 - dot_product / (norm1 * norm2 + 1e-8)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def predict(self, X):
        """
        预测样本类别
        """
        predictions = []
        
        for x in X:
            # 计算到所有训练样本的距离
            distances = self._calculate_distance(x, self.X_train)
            
            # 找到K个最近邻居的索引
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # 投票:选择出现次数最多的类别
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_class = unique_labels[np.argmax(counts)]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class LogisticRegression:
    """
    多分类逻辑回归 (Softmax回归)
    使用梯度下降优化
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.01):
        """
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
            reg_lambda: L2正则化系数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.classes = None
    
    def _softmax(self, z):
        """
        Softmax函数
        """
        # 减去最大值以提高数值稳定性
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """
        训练逻辑回归分类器
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 初始化权重和偏置
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros((1, n_classes))
        
        # 将标签转换为one-hot编码
        y_one_hot = np.eye(n_classes)[y]
        
        # 梯度下降
        for iteration in range(self.n_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(z)
            
            # 计算损失(交叉熵 + L2正则化)
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred + 1e-8), axis=1))
            loss += self.reg_lambda * np.sum(self.weights ** 2)
            
            # 反向传播
            dz = (y_pred - y_one_hot) / n_samples
            dw = np.dot(X.T, dz) + 2 * self.reg_lambda * self.weights
            db = np.sum(dz, axis=0, keepdims=True)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 每100次迭代打印一次损失
            if (iteration + 1) % 200 == 0:
                print(f"  迭代 {iteration + 1}/{self.n_iterations}, 损失: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测样本类别
        """
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(z)
        return self.classes[np.argmax(y_pred, axis=1)]
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DecisionStump:
    """
    决策树桩(单层决策树)
    用于AdaBoost集成学习
    """
    def __init__(self):
        self.feature_idx = None  # 选择的特征索引
        self.threshold = None    # 分割阈值
        self.polarity = 1        # 决策方向
        self.alpha = None        # 分类器权重
    
    def fit(self, X, y, sample_weights):
        """
        训练决策树桩
        """
        n_samples, n_features = X.shape
        best_error = float('inf')
        
        # 遍历所有特征
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 尝试不同的阈值
            for threshold in unique_values:
                # 尝试两个方向
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values > threshold] = -1
                    
                    # 计算加权错误率
                    error = np.sum(sample_weights[predictions != y])
                    
                    if error < best_error:
                        best_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity
        
        return best_error
    
    def predict(self, X):
        """
        预测
        """
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] > self.threshold] = -1
        
        return predictions


class AdaBoost:
    """
    AdaBoost集成学习分类器
    使用决策树桩作为弱分类器
    """
    def __init__(self, n_estimators=50):
        """
        参数:
            n_estimators: 弱分类器数量
        """
        self.n_estimators = n_estimators
        self.stumps = []
        self.classes = None
        self.binary_classifiers = {}  # 每个类别的二分类器
    
    def _fit_binary(self, X, y):
        """
        训练二分类AdaBoost
        """
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        stumps = []
        
        for i in range(self.n_estimators):
            stump = DecisionStump()
            error = stump.fit(X, y, sample_weights)
            
            # 避免除零
            error = max(error, 1e-10)
            
            # 计算分类器权重
            stump.alpha = 0.5 * np.log((1 - error) / error)
            
            # 更新样本权重
            predictions = stump.predict(X)
            sample_weights *= np.exp(-stump.alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)
            
            stumps.append(stump)
        
        return stumps
    
    def fit(self, X, y):
        """
        训练多分类AdaBoost (One-vs-Rest)
        """
        self.classes = np.unique(y)
        
        # 为每个类别训练一个二分类器
        for cls in self.classes:
            print(f"  训练类别 {cls} 的AdaBoost分类器...")
            y_binary = np.where(y == cls, 1, -1)
            self.binary_classifiers[cls] = self._fit_binary(X, y_binary)
        
        return self
    
    def _predict_binary(self, X, stumps):
        """
        二分类预测
        """
        predictions = np.zeros(X.shape[0])
        for stump in stumps:
            predictions += stump.alpha * stump.predict(X)
        return np.sign(predictions)
    
    def predict(self, X):
        """
        多分类预测
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            stumps = self.binary_classifiers[cls]
            for stump in stumps:
                scores[:, i] += stump.alpha * stump.predict(X)
        
        return self.classes[np.argmax(scores, axis=1)]
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
