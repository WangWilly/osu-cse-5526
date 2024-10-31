#!/bin/bash

FOLDER_PATH="./lab3/dist/sub1"

# if it is not exist, create it
if [ ! -d $FOLDER_PATH ]; then
    mkdir -p $FOLDER_PATH
fi

################################################################################

# TRAINING_FILE=$1
# EVALUATION_FILE=$2
TRAINING_FILE="./lab3/ncRNA_s.train.txt"
EVALUATION_FILE="./lab3/ncRNA_s.test.txt"


# iterate over the C values
echo "Training the linear SVM models"

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

# iterate over the C and α values
for idx in "${!iters[@]}"
do
    label=$((idx-4))
    svm-train -t 0 -c ${iters[idx]} -q $TRAINING_FILE ${FOLDER_PATH}/C_2to${label}_linear_svm.model
done

################################################################################
# evaluate the models
echo "Evaluating the linear SVM models"
for idx in "${!iters[@]}"
do
    label=$((idx-4))
    acc=$(svm-predict $EVALUATION_FILE ${FOLDER_PATH}/C_2to${label}_linear_svm.model ${FOLDER_PATH}/C_2to${label}_linear_svm.predict | grep "Accuracy" | awk '{print $3}')
    echo "Model C=2^$label, Accuracy: $acc"
done
