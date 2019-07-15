#!/bin/bash

# Checks if Rosenblatt_Linux.so exists
if [[ -f ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so ]]; then
    echo "Deleting Rosenblatt_Linux.so..."
    rm ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so
fi

# Checks if Rosenblatt_Linux.o exists
if [[ -f ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o ]]; then
    echo "Deleting Rosenblatt_Linux.o..."
    rm ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o
fi

# Checks if MultiLayerPerceptron_Linux.so exists
if [[ -f ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so ]]; then
    echo "Deleting MultiLayerPerceptron_Linux.so..."
    rm ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so
fi

# Checks if MultiLayerPerceptron_Linux.o exists
if [[ -f ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o ]]; then
    echo "Deleting MultiLayerPerceptron_Linux.o..."
    rm ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o
fi

# Checks if RBF_Linux.so exists
if [[ -f ../../Implementation/RBF/Librairie/Linux/RBF_Linux.so ]]; then
    echo "Deleting RBF_Linux.so..."
    rm ../../Implementation/RBF/Librairie/Linux/RBF_Linux.so
fi

# Checks if RBF_Linux.o exists
if [[ -f ../../Implementation/RBF/Librairie/Linux/RBF_Linux.o ]]; then
    echo "Deleting RBF_Linux.o..."
    rm ../../Implementation/RBF/Librairie/Linux/RBF_Linux.o
fi

echo "Creating Rosenblatt_Linux.o..."
g++ -c -fPIC ../../Implementation/Rosenblatt/Rosenblatt.cpp -o ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o
echo "Creating Rosenblatt_Linux.so..."
g++ -shared -Wl,-soname,../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so -o ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so ../../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o

echo "Creating MultiLayerPerceptron_Linux.o..."
g++ -c -fPIC ../../Implementation/MLP/MultiLayerPerceptron.cpp -o ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o
echo "Creating MultiLayerPerceptron_Linux.so..."
g++ -shared -Wl,-soname,../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so -o ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so ../../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o

echo "Creating RBF_Linux.o..."
g++ -c -fPIC ../../Implementation/RBF/RBF.cpp -o ../../Implementation/RBF/Librairie/Linux/RBF_Linux.o
echo "Creating RBF_Linux.so..."
g++ -shared -Wl,-soname,../../Implementation/RBF/Librairie/Linux/RBF_Linux.so -o ../../Implementation/RBF/Librairie/Linux/RBF_Linux.so ../../Implementation/RBF/Librairie/Linux/RBF_Linux.o