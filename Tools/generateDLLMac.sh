g++ -c -std=c++17 Rosenblatt.cpp -o Librairie/Mac/Rosenblatt_Mac.o &&
g++ -shared -Wl -o Librairie/Mac/Rosenblatt_Mac.so Librairie/Mac/Rosenblatt_Mac.o
g++ -c -std=c++17 MultiLayerPerceptron.cpp -o Librairie/Mac/MultiLayerPerceptron_Mac.o &&
g++ -shared -Wl -o Librairie/Mac/MultiLayerPerceptron_Mac.so Librairie/Mac/MultiLayerPerceptron_Mac.o