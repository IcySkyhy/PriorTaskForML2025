import numpy as np

class ReLU:
    '''ReLU 激活函数'''
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Flatten:
    '''展平层'''
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


class Dense:
    '''全连接层'''
    def __init__(self, input_dim, output_dim):
        # Xavier 初始化
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

# --- 卷积层 (简化版，stride=1, padding='same') ---
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Xavier 初始化
        self.W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) / np.sqrt(in_channels * kernel_size * kernel_size)
        self.b = np.zeros(out_channels)
        self.pad = kernel_size // 2

    def forward(self, x):
        self.x = np.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant')
        N, H, W, C = x.shape
        out_h = H
        out_w = W
        
        out = np.zeros((N, out_h, out_w, self.out_channels))
        
        for n in range(N):
            for f in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = self.x[n, i:i+self.kernel_size, j:j+self.kernel_size, :]
                        out[n, i, j, f] = np.sum(region * self.W[:, :, :, f]) + self.b[f]
        return out

    def backward(self, dout):
        N, H_out, W_out, F = dout.shape
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dx_padded = np.zeros_like(self.x)

        for n in range(N):
            for f in range(F):
                self.db[f] += np.sum(dout[n, :, :, f])
                for i in range(H_out):
                    for j in range(W_out):
                        region = self.x[n, i:i+self.kernel_size, j:j+self.kernel_size, :]
                        self.dW[:, :, :, f] += region * dout[n, i, j, f]
                        dx_padded[n, i:i+self.kernel_size, j:j+self.kernel_size, :] += self.W[:, :, :, f] * dout[n, i, j, f]
        
        # 去掉 padding
        dx = dx_padded[:, self.pad:-self.pad, self.pad:-self.pad, :]
        return dx
        
# --- 池化层 ---
class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, H, W, C = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        out = np.zeros((N, out_h, out_w, C))

        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * self.stride, j * self.stride
                        region = x[n, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c]
                        out[n, i, j, c] = np.max(region)
        return out

    def backward(self, dout):
        N, H_out, W_out, C = dout.shape
        dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * self.stride, j * self.stride
                        region = self.x[n, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c]
                        max_val = np.max(region)
                        mask = (region == max_val)
                        dx[n, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size, c] += mask * dout[n, i, j, c]
        return dx