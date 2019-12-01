CC = gcc
CFLAGS = -O3 -Wall -Iinc -I${OPENBLAS_ROOT}/include -I${OPENMPI_ROOT}/include
TYPES = sequential mpi_sync mpi

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/knnring_, $(TYPES)))

lib/%.a: obj/%.o
	mkdir -p lib
	ar rcs $@ $<

obj/%.o: src/%.c
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm obj/knnring_*.o lib/knnring_*.a

