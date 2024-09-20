import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(this_dir, "..")))

################################################################################

import argparse
import os

import matplotlib.pyplot as plt

from lab1.np_mod.activation_funcs.sigmoid import Sigmoid
from lab1.np_mod.cost.mse import MSE
from lab1.np_mod.dataset.is_odd import X, Y
from lab1.np_mod.perceptron_helper.two_layers import Perceptron

# TODO: torch is slower than numpy ???
# import torch


################################################################################

this_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(this_dir, "output")

################################################################################s


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lab1")
    parser.add_argument(
        "--learning_rate", type=float, default=0.5, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=1000000, help="Number of epochs")
    parser.add_argument("--momentum", type=float, default=0.0, help="Momentum")
    return parser.parse_args()


def main():
    # Setups
    # TODO: torch is slower than numpy ???
    # mps_device = torch.device("mps")

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
    print(f"Momentum: {args.momentum}")

    # Training
    perceptron = Perceptron(input_size, hidden_size, output_size, Sigmoid(), MSE())
    # TODO: torch is slower than numpy ???
    # perceptron.to(mps_device)
    perceptron.train(X, Y, args.learning_rate, args.epochs, args.momentum)

    ############################################################################

    # Testing
    for x, y in zip(X, Y):
        x = x.reshape(1, -1)
        _, _, _, y_hat = perceptron.forward(x)
        print(f"Prediction: {y_hat}, Actual: {y}")

    ############################################################################

    # Print parameters
    perceptron.print_param()

    # Plot cost history
    plt.plot(perceptron.cost_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.title(
        "lr = "
        + str(args.learning_rate)
        + ", epochs = "
        + str(args.epochs)
        + ", momentum = "
        + str(args.momentum)
    )

    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = (
        "lr_"
        + str(args.learning_rate)
        + "_epochs_"
        + str(args.epochs)
        + "_momentum_"
        + str(args.momentum)
        + ".jpg"
    )
    img_path = os.path.join(output_dir, img_name)
    plt.savefig(img_path, format="jpeg")

    # Show the plot
    plt.show()


################################################################################

if __name__ == "__main__":
    main()
