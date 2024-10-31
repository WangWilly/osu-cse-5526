#!/bin/bash

# This script is used to test OpenMP support on the current machine
# It will compile and run a simple OpenMP program
# clang++ -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include -L$(brew --prefix libomp)/lib -lomp test_openmp/main.cpp -o openmp
clang++ -Xpreprocessor -fopenmp test_openmp/main.cpp -o test_openmp/openmp

# Run the program
./test_openmp/openmp

# Clean up
rm ./test_openmp/openmp
