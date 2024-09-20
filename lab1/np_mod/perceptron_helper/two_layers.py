import numpy as np
from tqdm import tqdm

from lab1.np_mod.activation_funcs.const import SIGMOID_FUNC
from lab1.np_mod.cost.mse import COST_FUNC

################################################################################

RANDOM_SEED = 8787
NEURON_LOWER_BOUND = -1.0
NEURON_UPPER_BOUND = 1.0

################################################################################


class Perceptron:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        acti_func_pair: tuple[SIGMOID_FUNC, SIGMOID_FUNC],
        cost_func_pair: tuple[COST_FUNC, COST_FUNC],
    ) -> None:
        np.random.seed(RANDOM_SEED)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.uniform(
            NEURON_LOWER_BOUND,
            NEURON_UPPER_BOUND,
            (self.input_size + 1, self.hidden_size),
        )  # Add bias
        self.W2 = np.random.uniform(
            NEURON_LOWER_BOUND,
            NEURON_UPPER_BOUND,
            (self.hidden_size + 1, self.output_size),
        )  # Add bias

        self.acti_func = acti_func_pair[0]
        self.acti_func_derivative = acti_func_pair[1]

        self.cost_func = cost_func_pair[0]
        self.cost_func_derivative = cost_func_pair[1]

        self.cost_hist: list[float] = []

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_bias = np.c_[X, np.ones(X.shape[0])]
        V1 = np.dot(X_bias, self.W1)
        Y1 = self.acti_func(V1)

        Y1_bias = np.c_[Y1, np.ones(Y1.shape[0])]
        V2 = np.dot(Y1_bias, self.W2)
        Y2 = self.acti_func(V2)

        return V1, Y1, V2, Y2

    def backward(
        self, m: int, X: np.ndarray, Y: np.ndarray, Y1: np.ndarray, Y2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        delta2 = self.cost_func_derivative(Y, Y2) * self.acti_func_derivative(Y2)
        Y1_bias: np.ndarray = np.c_[Y1, np.ones(Y1.shape[0])]
        dW2 = np.dot(Y1_bias.T, delta2) / m

        sumPropFrom2 = np.dot(delta2, self.W2[:-1].T)
        delta1 = sumPropFrom2 * self.acti_func_derivative(Y1)
        X_bias: np.ndarray = np.c_[X, np.ones(X.shape[0])]
        dW1 = np.dot(X_bias.T, delta1) / m

        return dW1, dW2

    def update_parameters(
        self,
        dW1: np.ndarray,
        dW2: np.ndarray,
        learning_rate: float,
        momentum_alpha: float,
    ) -> None:
        self.W1 += learning_rate / (1 - momentum_alpha) * dW1
        self.W2 += learning_rate / (1 - momentum_alpha) * dW2

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float,
        epochs: int,
        momentum_alpha: float = 0.0,
    ) -> None:
        m = X.shape[0]
        epBar = tqdm(range(epochs))

        for _ in epBar:
            _, Y1, _, Y2 = self.forward(X)
            cost = self.cost_func(Y, Y2)
            self.cost_hist.append(cost)
            epBar.set_postfix_str("Cost: %.16f" % cost)

            dW1, dW2 = self.backward(m, X, Y, Y1, Y2)
            self.update_parameters(dW1, dW2, learning_rate, momentum_alpha)

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, _, Y_hat = self.forward(X)
        return Y_hat

    def print_param(self) -> None:
        print("W1: \n", self.W1)
        print("W2: \n", self.W2)
