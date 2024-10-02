import matplotlib.pyplot as plt
import numpy as np

################################################################################


def target_function(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.4 * np.sin(2 * np.pi * x)


def noised_target_function(
    x: np.ndarray, noise_lo: float, noise_hi: float
) -> np.ndarray:
    y = target_function(x)
    noise = np.random.uniform(noise_lo, noise_hi, y.shape)
    return y + noise


################################################################################

if __name__ == "__main__":
    x = np.random.uniform(0.0, 1.0, 100)
    y = target_function(x)
    y_noised = noised_target_function(x, -0.1, 0.1)

    # Show the data
    plt.scatter(x, y, color="blue")
    plt.scatter(x, y_noised, color="green")
    plt.show()
