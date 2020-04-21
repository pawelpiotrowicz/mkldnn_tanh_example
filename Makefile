#FOLDER=mkl-dnn_v1.2
FOLDER=mkl-dnn-develop
#FOLDER=mkl-dnn

#FOLDER=mkl-dnn2
P1=$(FOLDER)/include
P2=$(FOLDER)/build/include 
P3=$(FOLDER)/examples
CXX=clang++-4.0 -Wall -g -I$(P1) -I$(P2) -I$(P3)
BENCH=$(FOLDER)/build/tests/benchdnn/benchdnn

main: main.cpp
	$(CXX) -std=c++0x -o prog main.cpp -L$(FOLDER)/build/src -lmkldnn -ldnnl -Wl,-rpath,$(FOLDER)/build/src
	
test: $(BENCH)
	./$(BENCH) --eltwise --dir=FWD_D,BWD_D --dt=f32 --tag=nchw,nChw16c --inplace=true 50x192x55x55
