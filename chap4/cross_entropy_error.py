#p91 交差エントロピー誤差
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7    #マイナス無限大対策     1x10の-7乗
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
a1 = cross_entropy_error(np.array(y1), np.array(t))
print(a1)

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
a2 =cross_entropy_error(np.array(y2), np.array(t))
print(a2)


#出力される値が小さいほど誤差が少なく、より正解に近い