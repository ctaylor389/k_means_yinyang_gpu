#Important:
#see the params.h for the parameters
#need to do a make clean first before executing the program if modifying the parameters file


CODEOBJECTS = GPU.o kernel.o kmeansCPU.o kmeansUtils.o main.o
CC = hipcc
EXECUTABLE = kmeans


LFLAGS = -Wall -std=c99 -pedantic -fopenmp

FLAGS = -O3 -fopenmp --offload-arch=gfx906
 
CFLAGS = -c


all: $(EXECUTABLE)





kernel.o: kernel.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) kernel.cu 	

GPU.o: GPU.cu params.h
	$(CC) $(CFLAGS) $(FLAGS) GPU.cu

kmeansCPU.o: kmeansCPU.cu kmeansCPU.h 	
	$(CC) $(CFLAGS) $(FLAGS) kmeansCPU.cu

kmeansUtils.o: kmeansUtils.cu kmeansUtils.h 
	$(CC) $(CFLAGS) $(FLAGS) kmeansUtils.cu

main.o: main.cu params.h kmeansCPU.h kmeansUtils.h GPU.h
	$(CC) $(CFLAGS) $(FLAGS) main.cu 

$(EXECUTABLE): $(CODEOBJECTS)
	$(CC) $(FLAGS) $^ -o $@




clean:
	rm $(CODEOBJECTS)
	rm $(EXECUTABLE)
