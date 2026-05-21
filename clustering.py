import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def generate_cluster_plot(features):

    # create fake dataset from features
    data = np.tile(features, (10,1))

    # add small noise
    noise = np.random.normal(0,0.01,data.shape)
    data = data + noise

    # PCA
    pca = PCA(n_components=2)
    pts = pca.fit_transform(data)

    # plot
    plt.figure(figsize=(5,5))
    plt.scatter(pts[:,0], pts[:,1], c="blue")

    plt.title("Feature Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    path = "static/uploads/cluster.png"
    plt.savefig(path)
    plt.close()

    return path