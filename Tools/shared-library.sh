#!/bin/bash
if [[ -f libsource.so ]]; then
    rm libsource.so
fi

if [[ -f source.o ]]; then
    rm source.o
fi

g++ -c -fPIC source.cpp -o source.o
g++ -shared -Wl,-soname,libsource.so -o libsource.so source.o
