#p138 リンゴの買い物
import sys, os
sys.path.append(os.pardir)
from chap5.layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

#backward  各変数に関する微分を求める
#dprice = 1
#dapple_price, dtax = mul_tax_layer.backward(dprice)
#dapple, dapple_num = mul_apple_layer.bakward(dapple_price)

#print(dapple, dapple_num, dtax)