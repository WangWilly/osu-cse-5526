from perceptron_helper.two_layers import Perceptron
from cost.mse import cost
from activation_funcs.sigmoid import sigmoid, sigmoid_derivative
from dataset.is_odd import X, Y

################################################################################

def main():
    input_size = 4
    hidden_size = 4
    output_size = 1

    perceptron = Perceptron(input_size, hidden_size, output_size, (sigmoid, sigmoid_derivative), cost)
    perceptron.train(X, Y, 0.05, 100000)

    for x, y in zip(X, Y):
        x = x.reshape(1, -1)
        _, _, _, y_hat = perceptron.forward(x)
        print(f"Prediction: {y_hat}, Actual: {y}")

    perceptron.print_param()

if __name__ == "__main__":
    main()
