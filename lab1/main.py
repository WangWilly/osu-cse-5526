import argparse
from perceptron_helper.two_layers import Perceptron
from cost.mse import cost
from activation_funcs.sigmoid import sigmoid_pair
from dataset.is_odd import X, Y

################################################################################s

def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lab1")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000000, help="Number of epochs")
    return parser.parse_args()

def main():
    # Setups
    args = argument_parser()

    input_size = 4
    hidden_size = 4
    output_size = 1

    ############################################################################

    # Training
    perceptron = Perceptron(input_size, hidden_size, output_size, sigmoid_pair, cost)
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
