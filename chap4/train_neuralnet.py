#p118
#TwoLayerNetクラスを対象に、ミニバッチで学習を行う
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from chap4.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#ハイパーパラメータ
items_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(items_num):
    #ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch) 誤差逆伝播の高速版
    
    #パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    #学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)