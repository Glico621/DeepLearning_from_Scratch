#102 偏微分
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def function_2(x):
    return x[0]**2 + x[1]**2
    #または return np.sum(x**2)


x0 = np.arange(-3, 3, 0.1)
x1 = np.arange(-3, 3, 0.1)
X0, X1 = np.meshgrid(x0, x1)

y = function_2(np.array([X0, X1]))
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("f(x0, x1)")

ax.plot_wireframe(X0, X1, y)
plt.show()