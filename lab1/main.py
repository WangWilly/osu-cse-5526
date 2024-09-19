import argparse

from np_mod.activation_funcs.sigmoid import acti_pair
from np_mod.cost.mse import cost_pair
from np_mod.dataset.is_odd import X, Y
from np_mod.perceptron_helper.two_layers import Perceptron

################################################################################s


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lab1")
    parser.add_argument(
        "--learning_rate", type=float, default=0.5, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=1000000, help="Number of epochs")
    return parser.parse_args()


def main():
    # Setups
    args = argument_parser()

    input_size = 4
    hidden_size = 4
    output_size = 1

    # Print size setups
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Output size: {output_size}")

    ############################################################################

    # Print hyperparameters
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")

    # Training
    perceptron = Perceptron(input_size, hidden_size, output_size, acti_pair, cost_pair)
    perceptron.train(X, Y, args.learning_rate, args.epochs)

    ############################################################################

    # Testing
    for x, y in zip(X, Y):
        x = x.reshape(1, -1)
        _, _, _, y_hat = perceptron.forward(x)
        print(f"Prediction: {y_hat}, Actual: {y}")

    ############################################################################

    # Print parameters
    perceptron.print_param()


################################################################################

if __name__ == "__main__":
    main()
