from sklearn.datasets import load_wine
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Load dataset and seperate data and target value from from it
wine_data = load_wine()

X = wine_data["data"]
y = wine_data["target"]
# Data prepocessin using standarzation 
X_s = preprocessing.scale(X)

# Extract 2 Principal components
pca_wine = PCA(n_components=2)
pca_wine_fit = pca_wine.fit(X_s).transform(X_s)

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca_wine.explained_variance_ratio_)
)

#Plot Classes
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], ['Class1','Class2','Class3']):
    plt.scatter(
        pca_wine_fit[y == i, 0], pca_wine_fit[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of Wine dataset")

plt.show()

