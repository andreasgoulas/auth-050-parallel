CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -Iinc -I${OPENBLAS_ROOT}/include -I${OPENMPI_ROOT}/include
TYPES = sequential mpi_sync mpi

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/knnring_, $(TYPES)))

lib/%.a: obj/%.o
	mkdir -p lib
	ar rcs $@ $<

obj/knnring_sequential.o: src/knnring_sequential.c
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

obj/knnring_mpi_sync.o: src/knnring_mpi_sync.c
	mkdir -p obj
	$(MPICC) $(CFLAGS) -o $@ -c $<

obj/knnring_mpi.o: src/knnring_mpi.c
	mkdir -p obj
	$(MPICC) $(CFLAGS) -o $@ -c $<
clean:
	rm obj/knnring_*.o lib/knnring_*.a

