#Important:
#see the params.h for the parameters
#need to do a make clean first before executing the program if modifying the parameters file


CODEOBJECTS = kmeansGPU.o kmeansKernel.o kmeansCPU.o kmeansUtil.o kmeansTest.o kmeansMain.o
CC = hipcc
EXECUTABLE = kmeans


LFLAGS = -Wall -std=c99 -pedantic -fopenmp

FLAGS = -O3 -fopenmp --offload-arch=gfx906
 
CFLAGS = -c


all: $(EXECUTABLE)





kmeansKernel.o: kmeansKernel.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) kmeansKernel.cpp 	

kmeansGPU.o: kmeansGPU.cpp params.h
	$(CC) $(CFLAGS) $(FLAGS) kmeansGPU.cpp

kmeansCPU.o: kmeansCPU.cpp kmeansCPU.h 	
	$(CC) $(CFLAGS) $(FLAGS) kmeansCPU.cpp

kmeansUtil.o: kmeansUtil.cpp kmeansUtil.h 
	$(CC) $(CFLAGS) $(FLAGS) kmeansUtil.cpp

kmeansTest.o: kmeansTest.cpp kmeansTest.h params.h kmeansCPU.h kmeansUtil.h kmeansGPU.h
	$(CC) $(CFLAGS) $(FLAGS) kmeansTest.cpp

kmeansMain.o: kmeansMain.cpp params.h kmeansCPU.h kmeansUtil.h kmeansGPU.h kmeansTest.h
	$(CC) $(CFLAGS) $(FLAGS) kmeansMain.cpp

$(EXECUTABLE): $(CODEOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CODEOBJECTS)
	rm $(EXECUTABLE)
