FOLDER=mkl-dnn
P1=$(FOLDER)/include
P2=$(FOLDER)/build/include 
P3=$(FOLDER)/examples
CXX=clang++-4.0 -Wall -I$(P1) -I$(P2) -I$(P3)


main: main.cpp
	$(CXX) -std=c++0x -o prog main.cpp -L$(FOLDER)/build/src -lmkldnn -ldnnl -Wl,-rpath,$(FOLDER)/build/src
	