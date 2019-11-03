CC = gcc-7
CFLAGS = -O3 -Wall -Iinc
TYPES = sequential openmp cilk pthreads

.PHONY: all lib clean

all: lib

lib: $(addsuffix .a, $(addprefix lib/vptree_, $(TYPES)))

lib/%.a: %.o
	ar rcs $@ $<

%.o: src/%.c
	$(CC) $(CFLAGS) -o $@ -c $<

vptree_openmp.o: src/vptree_openmp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ -c $<

vptree_cilk.o: src/vptree_cilk.c
	$(CC) $(CFLAGS) -fcilkplus -o $@ -c $<

clean:
	rm *.o lib/vptree_*.a

