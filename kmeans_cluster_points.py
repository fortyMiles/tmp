from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def cluster_points(start, points, kernel=3, draw=False):
    estimator = KMeans(n_clusters=kernel)
    X = np.array([[p.x, p.y] for p in points])
    estimator.fit(X)

    clusted = {}

    for i in range(estimator.n_clusters):
        clusted[i] = [points[index] for index in np.where(estimator.labels_ == i)[0]]

    if draw:
        for i in set(estimator.labels_):
            index = estimator.labels_ == i
            plt.plot(X[index, 0], X[index, 1], 'o')
        plt.show()

    return clusted

