g++ -c Rosenblatt.cpp
g++ -shared -o Rosenblatt.dll Rosenblatt.o -W
g++ -c MultiLayerPerceptron.cpp
g++ -shared -o Librairie/Windows/MultiLayerPerceptron_Windows.dll Librairie/Windows/MultiLayerPerceptron_Windows.o -W