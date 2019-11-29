#include "knnring.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double drand() {
  return (double)rand() / (double)RAND_MAX;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: %s n d k\n", argv[0]);
    return -1;
  }

  int n = atoi(argv[1]);
  int k = atoi(argv[2]);
  int d = atoi(argv[3]);

  srand(time(NULL));

  double* x = malloc(n * d * sizeof(double));
  for (int i = 0; i < n * d; ++i) {
    x[i] = drand();
  }

  knnresult result = kNN(x, x, n, n, d, k);

  free(x);
  free(result.nidx);
  free(result.ndist);
  return 0;
}
