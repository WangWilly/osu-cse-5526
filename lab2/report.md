# Report of the Lab 2 (RBFLayer implementation)

## My observations

- Although the center and the width of the RBF layer are driven by the KMeans algorithm, the weights of the RBF layer are learned by the backpropagation algorithm.
- The number of RBF units is a hyperparameter that needs to be tuned. The more RBF units, the more complex the model, but it can lead to overfitting. (For now, 16 RBF units of the hidden layer are fine for learning the sine function.)
- The given iteration number (100) is not enough to learn the sine function. The cost function is still high after 100 iterations. (The cost function is the mean squared error (MSE) between the predicted value and the actual value.)
- For now this is a approximation of the sine function. However, it shows that the model can learn classfications and regressions.


## Result (cost function: MSE)

| LR 0.01 | LR 0.02 |
| ------------- | ------------- |
| <img src="output/lr_0.01_epochs_100_rbf_2.jpg" alt="lr_0.01_epochs_100_rbf_2" width="200"/>   | <img src="output/lr_0.02_epochs_100_rbf_2.jpg" alt="lr_0.02_epochs_100_rbf_2" width="200"/> |
| <img src="output/lr_0.01_epochs_100_rbf_4.jpg" alt="lr_0.01_epochs_100_rbf_4" width="200"/>   | <img src="output/lr_0.02_epochs_100_rbf_4.jpg" alt="lr_0.02_epochs_100_rbf_4" width="200"/> |
| <img src="output/lr_0.01_epochs_100_rbf_7.jpg" alt="lr_0.01_epochs_100_rbf_7" width="200"/>   | <img src="output/lr_0.02_epochs_100_rbf_7.jpg" alt="lr_0.02_epochs_100_rbf_7" width="200"/> |
| <img src="output/lr_0.01_epochs_100_rbf_11.jpg" alt="lr_0.01_epochs_100_rbf_11" width="200"/> | <img src="output/lr_0.02_epochs_100_rbf_11.jpg" alt="lr_0.02_epochs_100_rbf_11" width="200"/> |
| <img src="output/lr_0.01_epochs_100_rbf_16.jpg" alt="lr_0.01_epochs_100_rbf_16" width="200"/> | <img src="output/lr_0.02_epochs_100_rbf_16.jpg" alt="lr_0.02_epochs_100_rbf_16" width="200"/> |

## Result (Predicted vs Actual)

| LR 0.01 | LR 0.02 |
| ------------- | ------------- |
| <img src="output/predVsAct_lr_0.01_epochs_100_rbf_2.jpg" alt="lr_0.01_epochs_100_rbf_2" width="200"/>   | <img src="output/predVsAct_lr_0.02_epochs_100_rbf_2.jpg" alt="lr_0.02_epochs_100_rbf_2" width="200"/> |
| <img src="output/predVsAct_lr_0.01_epochs_100_rbf_4.jpg" alt="lr_0.01_epochs_100_rbf_4" width="200"/>   | <img src="output/predVsAct_lr_0.02_epochs_100_rbf_4.jpg" alt="lr_0.02_epochs_100_rbf_4" width="200"/> |
| <img src="output/predVsAct_lr_0.01_epochs_100_rbf_7.jpg" alt="lr_0.01_epochs_100_rbf_7" width="200"/>   | <img src="output/predVsAct_lr_0.02_epochs_100_rbf_7.jpg" alt="lr_0.02_epochs_100_rbf_7" width="200"/> |
| <img src="output/predVsAct_lr_0.01_epochs_100_rbf_11.jpg" alt="lr_0.01_epochs_100_rbf_11" width="200"/> | <img src="output/predVsAct_lr_0.02_epochs_100_rbf_11.jpg" alt="lr_0.02_epochs_100_rbf_11" width="200"/> |
| <img src="output/predVsAct_lr_0.01_epochs_100_rbf_16.jpg" alt="lr_0.01_epochs_100_rbf_16" width="200"/> | <img src="output/predVsAct_lr_0.02_epochs_100_rbf_16.jpg" alt="lr_0.02_epochs_100_rbf_16" width="200"/> |
