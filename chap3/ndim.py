import numpy as np

A = np.array([1,2,3,4])

print(A)
print(np.ndim(A))    #Aの次元
print(A.shape)       #Aの配列構成
print(A.shape[0])

B = np.array([[1,2],[3,4],[5,6]])

print(B)
print(np.ndim(B))
print(B.shape)      #行×列
