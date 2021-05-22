#p107 勾配降下法
import numpy as np

def gradient_decent(f, init_x, lr=0.01, step_num=100):      #lrは学習率 learning rate   #step_num:勾配法による繰り返しの回数
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)     #数値微分
        x -= lr * grad
    
    return x

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

def function_2(x):
    return np.sum(x**2)


#勾配法を使用して最小値の探索を行う
init_x = np.array([-3.0, 4.0])
ans = gradient_decent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(ans)

#学習率は大きすぎても、小さすぎてもいい結果にならない
#学習率あ大きすぎる例
#大きな値に発散してしまう
init_x = np.array([-3.0, 4.0])
ans2 = gradient_decent(function_2, init_x=init_x, lr=10.0, step_num=100)
print(ans2)

#学習率が小さすぎる例
#ほとんど更新されずに終わってしまう
init_x = np.array([-3.0, 4.0])
ans3 = gradient_decent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(ans3)

#適切な学習率を設定することが重要な問題になる

#ハイパーパラメータ:学習率のように、人の手によって設定されるパラメータ