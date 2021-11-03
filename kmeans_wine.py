from sklearn.datasets import load_wine
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine_data = load_wine()

X = wine_data["data"]
y = wine_data["target"]

X_s = preprocessing.scale(X)

from sklearn.decomposition import PCA
pca_wine = PCA(n_components=2)
pca_wine_fit = pca_wine.fit(X_s).transform(X_s)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], ['1','2','3']):
    plt.scatter(
        pca_wine_fit[y == i, 0], pca_wine_fit[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of Wine dataset")

plt.show()

from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(pca_wine_fit)

# plot the 3 clusters
plt.scatter(
    pca_wine_fit[y_km == 0, 0], pca_wine_fit[y_km == 0, 1],
    s=50, c='red',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    pca_wine_fit[y_km == 1, 0], pca_wine_fit[y_km == 1, 1],
    s=50, c='blue',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    pca_wine_fit[y_km == 2, 0], pca_wine_fit[y_km == 2, 1],
    s=50, c='green',
    marker='v', edgecolor='black',
    label='cluster 3'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='+',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


km2 = KMeans(
    n_clusters=5, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km2 = km2.fit_predict(pca_wine_fit)

# plot the 5 clusters
plt.scatter(
    pca_wine_fit[y_km2 == 0, 0], pca_wine_fit[y_km2 == 0, 1],
    s=50, c='red',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    pca_wine_fit[y_km2 == 1, 0], pca_wine_fit[y_km2 == 1, 1],
    s=50, c='blue',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    pca_wine_fit[y_km2 == 2, 0], pca_wine_fit[y_km2 == 2, 1],
    s=50, c='green',
    marker='v', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    pca_wine_fit[y_km2 == 3, 0], pca_wine_fit[y_km2 == 3, 1],
    s=50, c='navy',
    marker='v', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    pca_wine_fit[y_km2 == 4, 0], pca_wine_fit[y_km2 == 4, 1],
    s=50, c='orange',
    marker='v', edgecolor='black',
    label='cluster 5'
)

# plot the centroids
plt.scatter(
    km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1],
    s=250, marker='+',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()