import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

num_samples = 100

#x, y = make_classification(n_features=2, n_samples=num_samples, n_redundant=0, n_informative=1, n_clusters_per_class=1)
x, y = make_blobs(n_samples=num_samples, n_features=2, centers=2)

plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=0)

model = SVC(kernel="linear", C=1)
model.fit(x, y)

predictions = model.predict(x)
print(confusion_matrix(y, predictions))
print(classification_report(y, predictions))


#plot taken from scikit documentation:

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plot support vectors
sv = model.support_vectors_
print(sv)
num_sv = len(sv)
print(num_sv)
ax.scatter(sv[:, 0], sv[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.show()

lagrange_multipliers = np.abs(model.dual_coef_)
print(lagrange_multipliers)

print(model.fit_status_)