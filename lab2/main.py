import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(this_dir, "..")))

################################################################################

import argparse

import matplotlib.pyplot as plt

from lab1.np_mod.cost.mse import MSE
from lab2.np_mod.alg.kmeans import kmeans_centroids
from lab2.np_mod.gen_data import lab2_noised_target_output, lab2_random_input
from lab2.np_mod.perceptron_helper.rbf_2_layers import Perceptron

################################################################################

output_dir = os.path.join(this_dir, "output")

################################################################################s


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lab2")
    parser.add_argument("--rbf_bases", type=int, default=2, help="Number of RBF bases")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()


def main():
    args = argument_parser()

    ############################################################################
    # Data

    X = lab2_random_input()
    Y = lab2_noised_target_output(X)

    ############################################################################

    output_size = 1

    # Print size setups
    print(f"Output size: {output_size}")

    ############################################################################

    # Print hyperparameters
    print(f"RBF bases: {args.rbf_bases}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")

    # Training
    c_init_func = kmeans_centroids(args.rbf_bases)
    perceptron = Perceptron(args.rbf_bases, output_size, c_init_func, MSE())
    perceptron.train(X, Y, args.learning_rate, args.epochs)

    ############################################################################

    # Testing
    Y_hat = perceptron.predict(X)
    for y, y_hat in zip(Y, Y_hat):
        print(f"Prediction: {y_hat}, Actual: {y}")

    # Plot the data
    plt.scatter(X, Y, color="blue")
    plt.scatter(X, Y_hat, color="green")
    plt.title(
        "Predicted vs Actual: "
        + "lr = "
        + str(args.learning_rate)
        + ", epochs = "
        + str(args.epochs)
        + ", RBF bases = "
        + str(args.rbf_bases)
    )

    # Save plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pred_img_name = (
        "predVsAct"
        + "_lr_"
        + str(args.learning_rate)
        + "_epochs_"
        + str(args.epochs)
        + "_rbf_"
        + str(args.rbf_bases)
        + ".jpg"
    )

    pred_img_path = os.path.join(output_dir, pred_img_name)
    plt.savefig(pred_img_path, format="jpeg")

    # plt.show()
    # refresh plot
    plt.clf()

    ############################################################################

    # Print parameters
    print(perceptron)

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
        + ", RBF bases = "
        + str(args.rbf_bases)
    )

    img_name = (
        "lr_"
        + str(args.learning_rate)
        + "_epochs_"
        + str(args.epochs)
        + "_rbf_"
        + str(args.rbf_bases)
        + ".jpg"
    )
    img_path = os.path.join(output_dir, img_name)
    plt.savefig(img_path, format="jpeg")

    # Show the plot
    # plt.show()


################################################################################

if __name__ == "__main__":
    main()
