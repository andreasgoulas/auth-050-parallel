CC = gcc
NVCC = nvcc
CFLAGS = -O3 -Wall -Iinc
NVFLAGS =-Xptxas="-v -maxrregcount=62" -O3 -Iinc -dlink
TYPES = sequential gpu_v1 gpu_v2 gpu_v3

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/ising_, $(TYPES)))

lib/%.a: obj/%.o
	mkdir -p lib
	ar rcs $@ $<

obj/ising_sequential.o: src/ising_sequential.c
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

obj/ising_gpu_v1.o: src/ising_gpu_v1.cu
	mkdir -p obj
	$(NVCC) $(NVFLAGS) -o $@ -c $<

obj/ising_gpu_v2.o: src/ising_gpu_v2.cu
	mkdir -p obj
	$(NVCC) $(NVFLAGS) -o $@ -c $<

obj/ising_gpu_v3.o: src/ising_gpu_v3.cu
	mkdir -p obj
	$(NVCC) $(NVFLAGS) -o $@ -c $<

clean:
	rm obj/ising_*.o lib/ising_*.a

