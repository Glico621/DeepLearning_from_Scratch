#p140 加算，乗算レイヤを用いた誤差逆伝播
import sys, os
sys.path.append(os.pardir)
from chap5.layer_naive import MulLayer
from chap5.layer_naive_add import AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#layer
#それぞれのレイヤー〇を定義する
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward 順伝播
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

#backward 逆伝播
dprice = 1

dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_num, dapple, dorange, dorange_num, dtax)
