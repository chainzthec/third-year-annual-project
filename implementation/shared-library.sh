g++ -c -fPIC source.cpp -o source.o
g++ -shared -Wl,-soname,libsource.so -o libsource.so source.o
