#偏微分計算問題
import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_temp2(x1):
    return 3.0**2.0 + x1*x1

a = numerical_diff(function_tmp1, 3.0)
b = numerical_diff(function_temp2, 4.0)

print(a)
print(b)