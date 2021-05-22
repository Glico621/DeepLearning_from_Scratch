#p89 2乗和誤差
import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

#正解を２とする
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

#確率   例:2の確率が最も高い場合    →誤差は小さい
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
a1 = sum_squared_error(np.array(y1), np.array(t))
print(a1)

#例:７の確率が最も高い場合
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
a2 = sum_squared_error(np.array(y2), np.array(t))
print(a2)