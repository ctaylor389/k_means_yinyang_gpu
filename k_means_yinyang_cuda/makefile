#Important:
#see the params.h for the parameters
#need to do a make clean first before executing the program if modifying the parameters file


CUDAOBJECTS = kmeansKernel.o kmeansGPU.o kmeansCPU.o kmeansUtil.o kmeansTest.o  kmeansMain.o
CC = nvcc
EXECUTABLE = kmeans


LFLAGS = -Wall -std=c99 -pedantic -fopenmp $(DEBUG)

FLAGS = -O3 -std=c++11 -Xcompiler -fopenmp -arch=compute_75 -code=sm_75 -lcuda -lineinfo
 
CFLAGS = -c 


all: $(EXECUTABLE)





kmeansKernel.o: kmeansKernel.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansKernel.cu 		

kmeansGPU.o: kmeansGPU.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansGPU.cu

kmeansCPU.o: kmeansCPU.cu kmeansCPU.h 	
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansCPU.cu

kmeansUtil.o: kmeansUtil.cu kmeansUtil.h 
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansUtil.cu

kmeansTest.o: kmeansTest.cu kmeansTest.h params.h kmeansCPU.h kmeansUtil.h kmeansGPU.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansTest.cu 

kmeansMain.o: kmeansMain.cu params.h kmeansCPU.h kmeansUtil.h kmeansGPU.h kmeansTest.h
	$(CC) $(CFLAGS) $(FLAGS) $(NTHREADS) $(SEARCHMODE) $(PARAMS) kmeansMain.cu 

$(EXECUTABLE): $(CUDAOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CUDAOBJECTS)
	rm kmeans
