#p98 微分
#numerical differentiation 数値微分
#悪い例
def bad_numerical_diff(f, x):
    h = 1e-50   #小さすぎて丸め誤差問題が発生する
    return (f(x+h) - f(x)) / h

