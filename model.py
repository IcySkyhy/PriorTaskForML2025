import numpy as np
from ConvModulePack import *
from optimization import *

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y, loss_fn):
        scores = self.predict(x)
        return loss_fn.forward(scores, y)

    def train_step(self, x_batch, y_batch, loss_fn, optimizer):
        # 1. 前向传播
        out = x_batch
        for layer in self.layers:
            out = layer.forward(out)
        
        # 2. 计算损失
        loss = loss_fn.forward(out, y_batch)
        
        # 3. 反向传播
        dout = loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            
        # 4. 更新权重
        optimizer.update(self.layers)
        
        return loss
        
    def save_weights(self, filename):
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Dense, Conv2D)):
                weights[f'layer_{i}_W'] = layer.W
                weights[f'layer_{i}_b'] = layer.b
        np.savez(filename, **weights)
        print(f"Weights saved to {filename}")

    def load_weights(self, filename):
        data = np.load(filename)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Dense, Conv2D)):
                layer.W = data[f'layer_{i}_W']
                layer.b = data[f'layer_{i}_b']
        print(f"Weights loaded from {filename}")

    def fit(
            self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            epochs=5,
            batch_size=64,
            lr=0.01,
            print_every=100,
            save_dir="cifar10_pretrained.npz"
        ):
        loss_fn = SoftmaxCrossEntropy()
        optimizer = SGD(learning_rate=lr)

        num_batches = X_train.shape[0] // batch_size
        
        print(f"Starting training: {epochs} epochs, {num_batches} batches per epoch")
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0] if X_val is not None else 0}")
        print("-" * 80)

        for epoch in range(epochs):
            epoch_loss = 0
            correct_train = 0
            total_train = 0
            
            for i in range(num_batches):
                
                start = i * batch_size
                end = start + batch_size
                x_batch = X_train[start:end]
                y_batch = y_train[start:end]

                loss = self.train_step(x_batch, y_batch, loss_fn, optimizer)
                epoch_loss += loss
                
                # 计算训练batch准确率
                scores = self.predict(x_batch)
                predictions = np.argmax(scores, axis=1)
                true_labels = np.argmax(y_batch, axis=1)
                correct_train += np.sum(predictions == true_labels)
                total_train += y_batch.shape[0]

                if i % print_every == 0:
                    batch_acc = np.mean(predictions == true_labels) * 100
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{num_batches}], Loss: {loss:.4f}, Batch Acc: {batch_acc:.2f}%")
            
            # Epoch统计
            avg_loss = epoch_loss / num_batches
            train_acc = (correct_train / total_train) * 100
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Training Accuracy: {train_acc:.2f}%")
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_scores = self.predict(X_val)
                val_predictions = np.argmax(val_scores, axis=1)
                val_true = np.argmax(y_val, axis=1)
                val_accuracy = np.mean(val_predictions == val_true) * 100
                
                # 计算验证集loss
                val_loss = loss_fn.forward(val_scores, y_val)
                
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {val_accuracy:.2f}%")
            
            print(f"{'='*80}\n")

        # 检查保存路径，如果是directory则添加默认文件名，如果是文件名则直接保存
        save_dir = save_dir if save_dir.endswith('.npz') else save_dir + "/cifar10_pretrained.npz"

        self.save_weights(save_dir)

        # 最终测试
        if X_val is not None and y_val is not None:
            print("\nFinal Evaluation on Validation Set:")
            val_scores = self.predict(X_val)
            val_predictions = np.argmax(val_scores, axis=1)
            val_true = np.argmax(y_val, axis=1)
            accuracy = np.mean(val_predictions == val_true) * 100
            print(f"Final Validation Accuracy: {accuracy:.2f}%")
            