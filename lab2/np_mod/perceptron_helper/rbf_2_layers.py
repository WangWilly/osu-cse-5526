import numpy as np
from tqdm import tqdm

from lab1.np_mod.cost.const import Cost
from lab2.np_mod.alg.centroids_const import CENTROIDS_INIT_FUNC

################################################################################

NEURON_LOWER_BOUND = -1.0
NEURON_UPPER_BOUND = 1.0

################################################################################


# RBF layer
class Perceptron:
    def __init__(
        self, k: int, output_size: int, c_init_func: CENTROIDS_INIT_FUNC, cost: Cost
    ) -> None:
        self.weights = np.random.uniform(
            NEURON_LOWER_BOUND,
            NEURON_UPPER_BOUND,
            (k + 1, output_size),
        )  # Add bias

        self.c_init_func = c_init_func

        self.cost = cost
        self.cost_hist: list[float] = []

    ############################################################################

    def init_centroids(self, X: np.ndarray) -> None:
        self.centroids, self.spreads = self.c_init_func(X)

    ############################################################################

    def rbf(self, x: np.ndarray, c: np.ndarray, s: float) -> np.ndarray:
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s**2))

    def rbf_forward(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "centroids") or not hasattr(self, "spreads"):
            raise ValueError(
                "Centroids are not initialized. Run `init_centroids` method first."
            )

        # <=> np.array([[self.rbf(x, c, s) for c, s in zip(self.centroids, self.spreads)] for x in X])
        phi = np.zeros((X.shape[0], len(self.centroids)))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centroids):
                phi[i, j] = self.rbf(x, c, self.spreads[j])
        return phi

    ############################################################################

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phi = self.rbf_forward(X)
        phi_bias = np.c_[phi, np.ones(X.shape[0])]
        Y_out = np.dot(phi_bias, self.weights)
        return phi, Y_out

    def backward(
        self, X: np.ndarray, Y: np.ndarray, phi: np.ndarray, Y_out: np.ndarray
    ) -> tuple[np.ndarray]:
        m = X.shape[0]
        dY_out = self.cost.backward(Y, Y_out)
        phi_bias: np.ndarray = np.c_[phi, np.ones(m)]
        dW = np.dot(phi_bias.T, dY_out) / m
        return dW

    def update_parameters(self, dW: np.ndarray, learning_rate: float) -> None:
        self.weights += learning_rate * dW

    ############################################################################

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float,
        epochs: int,
    ) -> None:
        self.init_centroids(X)
        epBar = tqdm(range(epochs))

        for _ in epBar:
            phi, Y_out = self.forward(X)
            cost = self.cost.forward(Y, Y_out)
            self.cost_hist.append(cost)
            epBar.set_postfix_str("Cost: %.16f" % cost)

            dW = self.backward(X, Y, phi, Y_out)
            self.update_parameters(dW, learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, Y_out = self.forward(X)
        return Y_out

    def __str__(self) -> str:
        return f"Perceptron\nWeights:\n{self.weights}\nCentroids:\n{self.centroids}\nSpreads:\n{self.spreads}"


################################################################################
