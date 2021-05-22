#p104　偏微分の勾配
import numpy as np
import matplotlib.pylab as plt

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)     #xと同じ形状の配列を生成
    
    for idx in range(x.size):
        tmp_val= x[idx]
        #f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        #f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 =f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val    #値を元に戻す
    return grad


a1 = numerical_gradient(function_2, np.array([0.0, 2.0]))
print(a1)
