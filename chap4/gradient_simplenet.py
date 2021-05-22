import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)   #ガウス(正規)分布で初期化
    
    def predict(self, x):       #予測するためのメソッド
        return np.dot(x, self.W)
    
    def loss(self, x, t):       #t:正解ラベル   #損失関数の値を求めるメソッド
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss


net = simpleNet()
print(net.W)        #重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p)    #最大値のインデックス

t = np.array([0, 0, 1])     #正解ラベル
ex = net.loss(x, t)
print(ex)
print()

#勾配を求める
def f(W):       #ダミー
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)