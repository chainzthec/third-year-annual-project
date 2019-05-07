#!/bin/bash
# Author : Théo Huchard
# Last edited : 07/05/2019 16:00

if [[ -f ../Implementation/$1/$1.so ]]; then
    rm ../Implementation/$1/$1.so
fi

if [[ -f ../Implementation/$1/$1.o ]]; then
    rm ../Implementation/$1/$1.o
fi

g++ -c -fPIC ../Implementation/$1/$1.cpp -o ../Implementation/$1/$1.o
g++ -shared -Wl,-soname,../Implementation/$1/$1.so -o ../Implementation/$1/$1.so ../Implementation/$1/$1.o
