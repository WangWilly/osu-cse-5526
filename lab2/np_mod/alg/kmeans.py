import numpy as np

from .centroids_const import CENTROIDS_INIT_FUNC

################################################################################


def kmeans(
    X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform k-means clustering on the given data.
    X: input data of shape (n_samples, n_features)
    k: number of clusters
    max_iters: maximum number of iterations
    tol: tolerance for convergence
    """

    # initialize centroids
    indices = np.random.choice(
        X.shape[0], k, replace=False
    )  # randomly choose k indices from the data. shape: (k,)
    centroids = X[
        indices
    ]  # initialize centroids as the data points at the chosen indices. shape: (k, n_features)

    clusters = np.zeros(
        X.shape[0], dtype=int
    )  # initialize clusters. shape: (n_samples,)

    for _ in range(max_iters):
        # assign clusters
        diff = (
            X[:, np.newaxis] - centroids
        )  # add a new axis to X to allow broadcasting.
        # X w/ (n_samples, 1, n_features) - centroids w/ (k, n_features) -> diff w/ (n_samples, k, n_features) means the difference between each point and each centroid
        sq_diff = diff**2
        distances = np.linalg.norm(
            sq_diff, axis=2
        )  # compute the distance between each point and each centroid. shape: (n_samples, k)
        curr_clusters = np.argmin(
            distances, axis=1
        )  # assign each point to the closest centroid. shape: (n_samples,)

        # update centroids
        new_centroids = np.array(
            [X[curr_clusters == i].mean(axis=0) for i in range(k)]
        )  # compute the new centroids
        # `X[curr_clusters == i]` selects the points that are assigned to the i-th cluster. shape: (n_i_samples, n_features)
        if np.all(np.abs(new_centroids - centroids) < tol):
            # if the centroids do not change much, break
            break
        centroids = new_centroids
        clusters = curr_clusters

    return centroids, clusters


def kmeans_centroids(
    k: int, max_iters: int = 100, tol: float = 1e-4
) -> CENTROIDS_INIT_FUNC:
    def f(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centroids, _ = kmeans(X, k, max_iters, tol)

        # spread is varience of the distance between centroids
        d_max = np.max(
            [np.linalg.norm(c1 - c2) for c1 in centroids for c2 in centroids]
        )
        spreads = np.repeat(d_max / np.sqrt(2 * len(centroids)), len(centroids))

        return centroids, spreads

    return f


################################################################################

if __name__ == "__main__":
    # Generate some data
    X = np.random.randn(100, 2)

    # Perform k-means clustering
    k = 3
    centroids, clusters = kmeans(X, k)

    # Plot the data
    import matplotlib.pyplot as plt

    colors = ["red", "green", "blue"]
    for i in range(k):
        plt.scatter(X[clusters == i, 0], X[clusters == i, 1], color=colors[i])
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="x")
    plt.show()
