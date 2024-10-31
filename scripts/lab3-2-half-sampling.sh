#!/bin/bash

FOLDER_PATH="./lab3/dist/sub2"

# if it is not exist, create it
if [ ! -d $FOLDER_PATH ]; then
    mkdir -p $FOLDER_PATH
fi

################################################################################

# TRAINING_FILE=$1
# EVALUATION_FILE=$2
TRAINING_FILE="./lab3/ncRNA_s.train.txt"
EVALUATION_FILE="./lab3/ncRNA_s.test.txt"

# sample the random half rows from the input file
ROWS=$(wc -l < $TRAINING_FILE)
HALF_ROWS=$((ROWS / 2))
echo "Sampling $HALF_ROWS rows from $TRAINING_FILE"

# shuffle the input file and get the first half
# make a temporary file to store the shuffled rows
TEMP_FILE=$(mktemp)
shuf --random-source=<(yes 8787) $TRAINING_FILE | head -n $HALF_ROWS > $TEMP_FILE

# echo "Shuffled rows saved to $TEMP_FILE"

################################################################################

# iterate over the C and α values
echo "Finding the best C and α values for the RBF kernel SVM model"

iters=(
    0.0625
    0.125
    0.25
    0.5
    1
    2
    4
    8
    16
    32
    64
    128
    256
)

maxAcc=0
bestC=0
bestG=0

correlativeMatrix=()

# iterate over the C and α values
for i in "${iters[@]}"
do
    row=()
    for j in "${iters[@]}"
    do
        currAcc=$(svm-train -t 2 -c $i -g $j -v 5 $TEMP_FILE | grep "Cross Validation Accuracy" | awk '{print $5}' | sed 's/%//')
        # .1f is used to format the float number to 1 decimal places
        row+="$(printf "%.1f" $currAcc)% "
        if (( $(echo "$currAcc > $maxAcc" | bc -l) )); then
            maxAcc=$currAcc
            bestC=$i
            bestG=$j
        fi
    done
    correlativeMatrix+=("${row[@]}")
done

################################################################################

# print the correlative matrix
echo "Correlative Matrix: (C\α)"
for row in "${correlativeMatrix[@]}"
do
    echo "${row[@]}"
done

echo "Best C: $bestC, Best α: $bestG, Accuracy: $maxAcc%"

################################################################################
# train the model with the best C and α values
svm-train -t 2 -c $bestC -g $bestG -q $TRAINING_FILE ${FOLDER_PATH}/5fold_cross_validation_rbf_svm.model

################################################################################
# evaluate the model
echo "Evaluating the model on the evaluation set"
svm-predict $EVALUATION_FILE ${FOLDER_PATH}/5fold_cross_validation_rbf_svm.model ${FOLDER_PATH}/5fold_cross_validation_rbf_svm.predictions

################################################################################

# remove the temporary file
rm $TEMP_FILE
