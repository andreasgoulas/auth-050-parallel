#include "vptree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <sys/time.h>

#define SAMPLES 10
#define NTESTS 5
int kN[NTESTS] = { 100000, 100000, 100000, 100000, 100000 };
int kD[NTESTS] = { 10, 100, 500, 1000, 5000 };

double drand() {
  return (double)rand() / (double)RAND_MAX;
}

void vpfree(vptree* t) {
  if (t == NULL) {
    return;
  }

  vpfree(t->inner);
  vpfree(t->outer);
  free(t);
}

uint64_t test(int n, int d) {
  double* x = malloc(sizeof(double) * n * d);
  if (x == NULL) {
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      x[j + d*i] = drand();
    }
  }

  struct timeval start, stop;
  gettimeofday(&start, NULL);
  vptree* t = buildvp(x, n, d);
  gettimeofday(&stop, NULL);

  struct timeval time;
  time.tv_sec = stop.tv_sec - start.tv_sec;
  time.tv_usec = stop.tv_usec - start.tv_usec;
  uint64_t dt = (time.tv_sec * (uint64_t)1000) + (time.tv_usec / 1000);

  vpfree(t);
  free(x);
  return dt;
}

int main() {
  for (int i = 0; i < NTESTS; ++i) {
    uint64_t sum = 0;
    for (int j = 0; j < SAMPLES; ++j) {
      sum += test(kN[i], kD[i]);
    }
    sum /= SAMPLES;
    printf("n: %d d: %d\n", kN[i], kD[i]);
    printf("\tdt: %dms\n", sum);
  }

  return 0;
}
