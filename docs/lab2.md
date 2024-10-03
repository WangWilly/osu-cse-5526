# Lab 2: Radial Basis Function Network

## Implementation of Radial Basis Function Network (RBFN) in Python using NumPy

### 1. **Import Required Libraries**
Here we use NumPy for numerical computations:

```python
import numpy as np
```

### 2. **Define the RBF Function**
The Radial Basis Function (RBF) is typically a Gaussian function:

```python
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x-c)**2 / (2 * s**2))
```

`np.linalg` is used to calculate the Euclidean distance between the input `x` and the center `c`. The spread `s` controls the width of the Gaussian function.

### 3. **Initialize Network Parameters**
Define the centers and spreads for the RBF neurons:

```python
def initialize_centers(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    centers = X[indices]
    return centers

def initialize_spreads(centers):
    d_max = np.max([np.linalg.norm(c1-c2) for c1 in centers for c2 in centers])
    spreads = np.repeat(d_max / np.sqrt(2*len(centers)), len(centers))
    return spreads
```

### 4. **Compute the RBF Layer Output**
Calculate the output of the RBF layer:

```python
def compute_rbf_layer(X, centers, spreads):
    G = np.zeros((X.shape[0], len(centers)))
    for i, x in enumerate(X):
        for j, c in enumerate(centers):
            G[i, j] = rbf(x, c, spreads[j])
    return G
```

`G` is a matrix where each row corresponds to an input sample and each column corresponds to an RBF neuron. The value at position `(i, j)` is the output of the `j`-th RBF neuron for the `i`-th input sample.

### 5. **Train the Network**
Train the network using linear regression on the RBF layer output:

```python
def train_rbf_network(X, y, k):
    centers = initialize_centers(X, k)
    spreads = initialize_spreads(centers)
    G = compute_rbf_layer(X, centers, spreads)
    weights = np.linalg.pinv(G).dot(y)
    return centers, spreads, weights
```

The `np.linalg.pinv` function calculates the pseudo-inverse of the matrix `G` to solve the linear regression problem.

### 6. **Make Predictions**
Use the trained network to make predictions:

```python
def predict(X, centers, spreads, weights):
    G = compute_rbf_layer(X, centers, spreads)
    return G.dot(weights)
```

### Example Usage
Here's how you can use the above functions to train and test an RBF network:

```python
# Example data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]) # input data with shape (4, 2), 4 samples with 2 features
y_train = np.array([1, 2, 3, 4]) # target data with shape (4,), 4 samples

# Train the network
k = 2  # Number of RBF neurons
centers, spreads, weights = train_rbf_network(X_train, y_train, k)

# Make predictions
X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
predictions = predict(X_test, centers, spreads, weights)
print(predictions)
```

This example demonstrates a simple RBF network using only NumPy. You can expand and modify it to suit more complex datasets and tasks. If you have any specific questions or need further assistance, feel free to ask!

### Sources

1.  Radial Basis Function Networks (RBFNs) with Python 3: A Comprehensive .... https://innovationyourself.com/radial-basis-function-networks-rbfn/.
2. What are radial basis function neural networks? - GeeksforGeeks. https://www.geeksforgeeks.org/what-are-radial-basis-function-neural-networks/.
3. Radial Basis Function Neural Network implementation for Python. https://github.com/eugeniashurko/rbfnnpy.
4. MachineLearningTutorial/Numpy/RBF-Network-with-Kmeans ... - GitHub. https://github.com/Yangyangii/MachineLearningTutorial/blob/master/Numpy/RBF-Network-with-Kmeans-clustering.ipynb.
5. GitHub - mrthetkhine/RBFNeuralNetwork: RBF(Radial Basis Function .... https://github.com/mrthetkhine/RBFNeuralNetwork.

## Implementation of Radial Basis Function Network (RBFN) with K-Means Clustering

Incorporating k-means clustering to determine the centers of the RBF neurons can improve the performance of the RBF network. Here's how you can revise the program to use k-means clustering:

### 1. **Import Required Libraries**
First, import NumPy and the k-means algorithm from SciPy:

```python
import numpy as np
from scipy.cluster.vq import kmeans2
```

### 2. **Define the RBF Function**
The Radial Basis Function (RBF) remains the same:

```python
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x-c)**2 / (2 * s**2))
```

### 3. **Initialize Network Parameters Using k-means**
Use k-means to initialize the centers and spreads:

```python
def initialize_centers_kmeans(X, k):
    centers, _ = kmeans2(X, k, minit='points')
    return centers

def initialize_spreads_kmeans(centers):
    d_max = np.max([np.linalg.norm(c1-c2) for c1 in centers for c2 in centers])
    spreads = np.repeat(d_max / np.sqrt(2*len(centers)), len(centers))
    return spreads
```

### 4. **Compute the RBF Layer Output**
Calculate the output of the RBF layer:

```python
def compute_rbf_layer(X, centers, spreads):
    G = np.zeros((X.shape[0], len(centers)))
    for i, x in enumerate(X):
        for j, c in enumerate(centers):
            G[i, j] = rbf(x, c, spreads[j])
    return G
```

### 5. **Train the Network**
Train the network using linear regression on the RBF layer output:

```python
def train_rbf_network_kmeans(X, y, k):
    centers = initialize_centers_kmeans(X, k)
    spreads = initialize_spreads_kmeans(centers)
    G = compute_rbf_layer(X, centers, spreads)
    weights = np.linalg.pinv(G).dot(y)
    return centers, spreads, weights
```

### 6. **Make Predictions**
Use the trained network to make predictions:

```python
def predict(X, centers, spreads, weights):
    G = compute_rbf_layer(X, centers, spreads)
    return G.dot(weights)
```

### Example Usage
Here's how you can use the revised functions to train and test an RBF network with k-means clustering:

```python
# Example data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([1, 2, 3, 4])

# Train the network
k = 2  # Number of RBF neurons
centers, spreads, weights = train_rbf_network_kmeans(X_train, y_train, k)

# Make predictions
X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
predictions = predict(X_test, centers, spreads, weights)
print(predictions)
```

### Sources

1. GitHub - ShyamalShah3/KMEANSRBF: This is an implementation of the RBF .... https://github.com/ShyamalShah3/KMEANSRBF.
2. Research on the application of RBF neural network based on K-means .... https://ieeexplore.ieee.org/document/6159984/.
3. An RBF Neural Network Clustering Algorithm Based on K‐Nearest Neighbor. https://onlinelibrary.wiley.com/doi/pdf/10.1155/2022/1083961.
4. RBF-Network-with-Kmeans-clustering.ipynb - GitHub. https://github.com/Yangyangii/MachineLearningTutorial/blob/master/Numpy/RBF-Network-with-Kmeans-clustering.ipynb.
5. Radial Basis Function Network (RBFN) Tutorial · Chris McCormick. https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/.

## Implementation of K-Means Clustering Algorithm in Python using NumPy

Sure! Writing the k-means algorithm from scratch in Python using NumPy involves several steps: initializing the centroids, assigning each data point to the nearest centroid, updating the centroids, and repeating the process until convergence. Here's a step-by-step implementation:

### 1. **Import Required Libraries**
First, import NumPy:

```python
import numpy as np
```

### 2. **Initialize Centroids**
Randomly initialize the centroids by selecting `k` random points from the dataset:

```python
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    return centroids
```

### 3. **Assign Clusters**
Assign each data point to the nearest centroid:

```python
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters
```

### 4. **Update Centroids**
Update the centroids by calculating the mean of all points assigned to each centroid:

```python
def update_centroids(X, clusters, k):
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return new_centroids
```

### 5. **K-Means Algorithm**
Combine the above steps into the k-means algorithm:

```python
def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, clusters
```

### Example Usage
Here's how you can use the k-means algorithm to cluster a dataset:

```python
# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])

# Number of clusters
k = 2

# Run k-means
centroids, clusters = kmeans(X, k)

print("Centroids:\n", centroids)
print("Clusters:\n", clusters)
```

This implementation initializes the centroids, assigns clusters, updates the centroids, and repeats the process until the centroids converge or the maximum number of iterations is reached. You can adjust the number of clusters `k` and the tolerance `tol` as needed.

### Sources

1. K-Means Clustering in Python: Step-by-Step Example - Statology. https://www.statology.org/k-means-clustering-in-python/.
2. K-Means Clustering in Python: A Practical Guide – Real Python. https://realpython.com/k-means-clustering-python/.
3. Coding K-Means Clustering using Python and NumPy. https://dev.to/sajal2692/coding-k-means-clustering-using-python-and-numpy-fg1.
4. Introduction to k-Means Clustering with scikit-learn in Python. https://www.datacamp.com/tutorial/k-means-clustering-python.

## Training an RBF Neural Network with Gradient Descent in Python

Updating the weights in an RBF neural network typically involves using a method like gradient descent or a closed-form solution such as the pseudo-inverse. Here's how you can implement weight updates using gradient descent:

### 1. **Define the Loss Function**
First, define the Mean Squared Error (MSE) loss function:

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### 2. **Compute the Gradient**
Calculate the gradient of the loss function with respect to the weights:

```python
def compute_gradient(G, y_true, y_pred):
    return -2 * G.T.dot(y_true - y_pred) / y_true.size
```

### 3. **Update the Weights**
Update the weights using the gradient descent rule:

```python
def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient
```

### 4. **Train the Network with Gradient Descent**
Combine these steps to train the RBF network using gradient descent:

```python
def train_rbf_network_gd(X, y, k, learning_rate=0.01, max_iters=1000, tol=1e-6):
    centers = initialize_centers_kmeans(X, k)
    spreads = initialize_spreads_kmeans(centers)
    G = compute_rbf_layer(X, centers, spreads)
    weights = np.random.randn(G.shape[1])
    
    for i in range(max_iters):
        y_pred = G.dot(weights)
        loss = mse_loss(y, y_pred)
        gradient = compute_gradient(G, y, y_pred)
        new_weights = update_weights(weights, gradient, learning_rate)
        
        if np.linalg.norm(new_weights - weights) < tol:
            break
        
        weights = new_weights
    
    return centers, spreads, weights
```

### Example Usage
Here's how you can use the gradient descent method to train and test the RBF network:

```python
# Example data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([1, 2, 3, 4])

# Train the network
k = 2  # Number of RBF neurons
learning_rate = 0.01
centers, spreads, weights = train_rbf_network_gd(X_train, y_train, k, learning_rate)

# Make predictions
X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
G_test = compute_rbf_layer(X_test, centers, spreads)
predictions = G_test.dot(weights)
print(predictions)
```

This implementation uses gradient descent to update the weights of the RBF network. The `train_rbf_network_gd` function initializes the centers using k-means, computes the RBF layer output, and iteratively updates the weights until convergence.

### Sources
1. What are radial basis function neural networks? - GeeksforGeeks. https://www.geeksforgeeks.org/what-are-radial-basis-function-neural-networks/.
2. How to Train a Machine Learning Radial Basis Function Network Using C#. https://visualstudiomagazine.com/articles/2020/03/19/train-radial-basis-function.aspx.
3. Introduction to Radial Bias Function Networks. https://www.aiplusinfo.com/blog/introduction-to-radial-bias-function-networks/.
4. An online adjusting RBF neural network for nonlinear system modeling. https://link.springer.com/article/10.1007/s10489-021-03106-7.
5. Training Radial Basis Function Networks - Computational Intelligence. https://www.computational-intelligence.eu/cibook_media/Downloads/NN/NN_11_RBF_Training.pdf.
