import torch
from tqdm import tqdm

from lab1.torch_mod.activation_funcs.const import SIGMOID_FUNC
from lab1.torch_mod.cost.mse import COST_FUNC

################################################################################

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

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

         # Add bias
        self.W1 = torch.rand(self.input_size + 1, self.hidden_size) * 2.0 - 1.0
        # Add bias
        self.W2 = torch.rand(self.hidden_size + 1, self.output_size) * 2.0 - 1.0

        self.acti_func = acti_func_pair[0]
        self.acti_func_derivative = acti_func_pair[1]

        self.cost_func = cost_func_pair[0]
        self.cost_func_derivative = cost_func_pair[1]

        self.cost_hist: list[float] = []

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Arrange
        device = self.W1.device

        # Resolve
        X_bias = torch.cat((X, torch.ones(X.shape[0], 1, device=device)), dim=1)
        V1 = torch.matmul(X_bias, self.W1)
        Y1 = self.acti_func(V1)

        Y1_bias = torch.cat((Y1, torch.ones(Y1.shape[0], 1, device=device)), dim=1)
        V2 = torch.matmul(Y1_bias, self.W2)
        Y2 = self.acti_func(V2)

        return V1, Y1, V2, Y2

    def backward(
        self, m: int, X: torch.Tensor, Y: torch.Tensor, Y1: torch.Tensor, Y2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Arrange
        device = self.W1.device

        # Resolve
        delta2 = self.cost_func_derivative(Y, Y2) * self.acti_func_derivative(Y2)
        Y1_bias: torch.Tensor = torch.cat((Y1, torch.ones(Y1.shape[0], 1, device=device)), dim=1)
        dW2 = torch.matmul(Y1_bias.T, delta2) / m

        sumPropFrom2 = torch.matmul(delta2, self.W2[:-1].T)
        delta1 = sumPropFrom2 * self.acti_func_derivative(Y1)
        X_bias: torch.Tensor = torch.cat((X, torch.ones(X.shape[0], 1, device=device)), dim=1)
        dW1 = torch.matmul(X_bias.T, delta1) / m

        return dW1, dW2

    def update_parameters(
        self, dW1: torch.Tensor, dW2: torch.Tensor, learning_rate: float
    ) -> None:
        self.W1 += learning_rate * dW1
        self.W2 += learning_rate * dW2

    def train(
        self, X: torch.Tensor, Y: torch.Tensor, learning_rate: float, epochs: int
    ) -> None:
        # Arrange
        m = X.shape[0]
        X = X.to(self.W1.device)
        Y = Y.to(self.W1.device)

        # Training
        epBar = tqdm(range(epochs))
        for _ in epBar:
            _, Y1, _, Y2 = self.forward(X)
            cost = self.cost_func(Y, Y2)
            self.cost_hist.append(cost)
            epBar.set_postfix_str("Cost: %.16f" % cost)

            dW1, dW2 = self.backward(m, X, Y, Y1, Y2)
            self.update_parameters(dW1, dW2, learning_rate)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        _, _, _, Y_hat = self.forward(X)
        return Y_hat

    def to(self, device: torch.device) -> None:
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)

    def print_param(self) -> None:
        print("W1: \n", self.W1)
        print("W2: \n", self.W2)
