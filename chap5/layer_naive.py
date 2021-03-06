#p137 乗算レイヤの実装

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    #順伝播
    def forward(self, x, y): 
        self.x = x
        self.y = y
        out = x * y
        return out
    
    #逆伝播
    def backward(self, dout):
        dx = dout * self.y      #xとyをひっくり返す
        dy = dout * self.x
        return dx, dy
