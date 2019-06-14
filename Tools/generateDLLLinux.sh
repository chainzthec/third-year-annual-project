g++ -c -std=c++17 Rosenblatt.cpp -o Librairie/Linux/Rosenblatt_Linux.o
g++ -shared -Wl -o Librairie/Linux/Rosenblatt_Linux.so Librairie/Linux/Rosenblatt_Linux.o
g++ -c -std=c++17 MultiLayerPerceptron.cpp -o Librairie/Linux/MultiLayerPerceptron_Linux.o
g++ -shared -Wl -o Librairie/Linux/MultiLayerPerceptron_Linux.so Librairie/Linux/MultiLayerPerceptron_Linux.o