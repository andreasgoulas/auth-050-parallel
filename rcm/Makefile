CC = gcc
CFLAGS = -O3 -Wall -Iinc -fopenmp -g

.PHONY: all lib clean

all: lib test

test: obj/test.o lib/rcm.a
	$(CC) $(CFLAGS) -o $@ $^

lib: lib/rcm.a

lib/rcm.a: obj/rcm.o
	mkdir -p lib
	ar rcs $@ $^

obj/%.o: src/%.c
	mkdir -p obj
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm obj/*.o lib/*.a

