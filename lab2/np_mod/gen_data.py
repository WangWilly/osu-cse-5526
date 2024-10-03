import numpy as np

################################################################################

LAB2_X_SIZE = 75
LAB2_X_LO = 0.0
LAB2_X_HI = 1.0
LAB2_NOISE_LO = -0.1
LAB2_NOISE_HI = 0.1

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


def lab2_random_input() -> np.ndarray:
    return np.random.uniform(LAB2_X_LO, LAB2_X_HI, (LAB2_X_SIZE, 1))


def lab2_noised_target_output(x: np.ndarray) -> np.ndarray:
    return noised_target_function(x, LAB2_NOISE_LO, LAB2_NOISE_HI)


################################################################################

if __name__ == "__main__":
    x = np.random.uniform(0.0, 1.0, 100)
    y = target_function(x)
    y_noised = noised_target_function(x, -0.1, 0.1)

    # Show the data
    import matplotlib.pyplot as plt

    plt.scatter(x, y, color="blue")
    plt.scatter(x, y_noised, color="green")
    plt.show()
