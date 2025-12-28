import numpy as np
from ConvModulePack import *

# --- 损失函数 ---
class SoftmaxCrossEntropy:
    def forward(self, x, y):
        # Softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Cross-Entropy Loss
        self.y = y
        N = x.shape[0]
        log_likelihood = -np.log(self.probs[range(N), np.argmax(y, axis=1)])
        loss = np.sum(log_likelihood) / N
        return loss

    def backward(self):
        N = self.probs.shape[0]
        dx = self.probs.copy()
        dx[range(N), np.argmax(self.y, axis=1)] -= 1
        dx /= N
        return dx

# --- 优化器 ---
class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layers):
        for layer in layers:
            if isinstance(layer, (Dense, Conv2D)):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db