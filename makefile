#Important:
#see the params.h for the parameters
#need to do a make clean first before executing the program if modifying the parameters file


CUDAOBJECTS = GPU.o kernel.o kmeansCPU.o kmeansUtils.o main.o
CC = nvcc
EXECUTABLE = main


LFLAGS = -Wall -std=c99 -pedantic -fopenmp $(DEBUG)

FLAGS = -O3 -std=c++11 -Xcompiler -fopenmp -arch=compute_60 -code=sm_60 -lcuda -lineinfo
 
CFLAGS = -c 


all: $(EXECUTABLE)





kernel.o: kernel.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kernel.cu 		

GPU.o: GPU.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) GPU.cu

kmeansCPU.o: kmeansCPU.cu kmeansCPU.h 	
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansCPU.cu

kmeansUtils.o: kmeansUtils.cu kmeansUtils.h 
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansUtils.cu

main.o: main.cu params.h kmeansCPU.h kmeansUtils.h GPU.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) main.cu 

$(EXECUTABLE): $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS)
	rm main
