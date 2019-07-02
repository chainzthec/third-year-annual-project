g++ -c -std=c++17 ../Implementation/Rosenblatt/Rosenblatt.cpp -o ../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o
g++ -shared -Wl -o ../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.so ../Implementation/Rosenblatt/Librairie/Linux/Rosenblatt_Linux.o
g++ -c -std=c++17 ../Implementation/MLP/MultiLayerPerceptron.cpp -o ../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o
g++ -shared -Wl -o ../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.so ../Implementation/MLP/Librairie/Linux/MultiLayerPerceptron_Linux.o
g++ -c -std=c++17 ../Implementation/RBF/RBF.cpp -o ../Implementation/RBF/Librairie/Linux/RBF_Linux.o
g++ -shared -Wl -o ../Implementation/RBF/Librairie/Linux/RBF_Linux.so ../Implementation/RBF/Librairie/Linux/RBF_Linux.o