import numpy as np

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])

print(np.dot(A,B))

C = np.array([[1,2,3],[4,5,6]])
D = np.array([[1,2], [3,4], [5,6]])

print(np.dot(C,D))
print(np.dot(D,C))

E = np.array([[1,2],[3,4],[5,6]])
F = np.array([7,8])

print(np.dot(E,F))
print()

X = np.array([1,2])
W = np.array([[1,2,3],[4,5,6]])
print(X.shape)
print(W.shape)

Y = np.dot(X,W)
print(Y)