#include "ising.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <sys/time.h>

double kW[] = {
  0.004, 0.016, 0.026, 0.016, 0.004,
  0.016, 0.071, 0.117, 0.071, 0.016,
  0.026, 0.117, 0.000, 0.117, 0.026,
  0.016, 0.071, 0.117, 0.071, 0.016,
  0.004, 0.016, 0.026, 0.016, 0.004
};

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: %s n k\n", argv[0]);
    return -1;
  }

  int n = atoi(argv[1]);
  int k = atoi(argv[2]);

  srand(time(NULL));

  int* g = malloc(n * n * sizeof(int));
  for (int i = 0; i < n * n; ++i) {
    g[i] = (rand() % 2 == 0) ? -1 : 1;
  }

  struct timeval start, stop;
  gettimeofday(&start, NULL);
  ising(g, kW, k, n);
  gettimeofday(&stop, NULL);

  struct timeval time;
  time.tv_sec = stop.tv_sec - start.tv_sec;
  time.tv_usec = stop.tv_usec - start.tv_usec;
  uint64_t dt = (time.tv_sec * (uint64_t)1000) + (time.tv_usec / 1000);

  printf("n=%d, k=%d: %dms\n", n, k, (int)dt);

  free(g);
  return 0;
}
