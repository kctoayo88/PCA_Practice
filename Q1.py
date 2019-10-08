# ##Import required module
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ## Load data
data_homework = np.loadtxt(open("./homework 1_2.csv", "rb"), delimiter=",", skiprows=0)
data_T = data_homework.T
plt.plot(data_T)
plt.show()

# ## PCA
pca = PCA(n_components=3)
dimension_reduction = pca.fit_transform(data_T)
print(dimension_reduction.shape)
# ## Plot result
plt.plot(dimension_reduction)
plt.savefig("./homework_Q1.png", dpi=300, format="png")
plt.show()
