# Sub1: Lab 3 - Linear SVMs

```
osu-cse-5526-py3.12(⎈|lab6:group-mitglied) ~/Projects/osu/osu-cse-5526/ [master] ./scripts/lab3-1-choose-c.sh
Training the linear SVM models
Evaluating the linear SVM models
Model C=2^-4, Accuracy: 66.4336%
Model C=2^-3, Accuracy: 66.4336%
Model C=2^-2, Accuracy: 66.4336%
Model C=2^-1, Accuracy: 77.8222%
Model C=2^0, Accuracy: 92.5075%
Model C=2^1, Accuracy: 94.006%
Model C=2^2, Accuracy: 93.7063%
Model C=2^3, Accuracy: 93.8062%
Model C=2^4, Accuracy: 93.8062%
Model C=2^5, Accuracy: 93.8062%
Model C=2^6, Accuracy: 93.8062%
Model C=2^7, Accuracy: 93.8062%
Model C=2^8, Accuracy: 93.8062%
```

# Sub2: Lab 3 - RBF SVMs

```
osu-cse-5526-py3.12(⎈|lab6:group-mitglied) ~/Projects/osu/osu-cse-5526/ [master*] ./scripts/lab3-2-half-sampling.sh
Sampling 1000 rows from ./lab3/ncRNA_s.train.txt
Finding the best C and α values for the RBF kernel SVM model
Correlative Matrix: (C\α)
68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 
68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.6% 68.4% 68.4% 68.4% 
68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 68.4% 70.2% 73.5% 73.2% 71.4% 68.8% 68.4% 
68.4% 68.4% 68.4% 68.4% 68.4% 71.0% 79.8% 83.1% 82.4% 79.0% 76.1% 72.5% 69.1% 
68.4% 68.4% 68.4% 68.6% 82.0% 91.2% 90.8% 90.6% 88.3% 84.9% 81.6% 76.2% 72.3% 
68.4% 68.4% 69.7% 88.8% 94.9% 94.2% 93.4% 93.3% 90.6% 88.3% 84.0% 78.5% 74.0% 
68.4% 69.9% 90.5% 95.3% 95.0% 94.8% 93.4% 93.2% 91.6% 88.6% 84.0% 78.7% 73.9% 
69.9% 91.3% 95.6% 95.6% 95.1% 94.2% 93.7% 93.7% 91.8% 88.2% 84.3% 78.9% 74.0% 
91.7% 95.5% 95.5% 95.6% 94.9% 94.2% 94.5% 93.4% 91.2% 88.1% 84.6% 79.0% 74.0% 
95.4% 95.7% 95.6% 95.1% 94.4% 94.9% 94.7% 92.5% 90.4% 88.3% 84.9% 79.0% 74.0% 
95.7% 95.8% 95.4% 95.4% 94.5% 95.1% 94.1% 91.4% 89.7% 88.1% 84.8% 79.0% 74.0% 
95.8% 95.6% 95.7% 95.1% 95.0% 94.7% 93.1% 91.0% 89.5% 88.0% 85.1% 78.9% 74.1% 
95.8% 95.9% 95.3% 95.4% 94.3% 94.0% 92.1% 90.4% 89.4% 88.5% 85.2% 79.2% 74.3% 
Best C: 256, Best α: 0.125, Accuracy: 95.9%
Evaluating the model on the evaluation set
Accuracy = 94.6054% (947/1001) (classification)
```
