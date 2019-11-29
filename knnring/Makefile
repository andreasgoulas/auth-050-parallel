CC = gcc
CFLAGS = -O3 -Wall -Iinc -I${OPENBLAS_ROOT}/include -I${OPENMPI_ROOT}/include
TYPES = sequential mpi_sync mpi

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/knnring_, $(TYPES)))

lib/%.a: %.o
	mkdir -p lib
	ar rcs $@ $<

knnring_sequential.o: src/knnring_sequential.c
	$(CC) $(CFLAGS) -o $@ -c $<

knnring_mpi_sync.o: src/knnring_mpi_sync.c
	$(CC) $(CFLAGS) -o $@ -c $<

knnring_mpi.o: src/knnring_mpi.c
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm *.o lib/knnring_*.a

