#include "ising.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 517

double kW[] = {
  0.004, 0.016, 0.026, 0.016, 0.004,
  0.016, 0.071, 0.117, 0.071, 0.016,
  0.026, 0.117, 0.000, 0.117, 0.026,
  0.016, 0.071, 0.117, 0.071, 0.016,
  0.004, 0.016, 0.026, 0.016, 0.004
};

int* load_bin(char* path) {
  int* data = malloc(N * N * sizeof(int));
  if (data == NULL) {
    return NULL;
  }

  FILE* file = fopen(path, "rb");
  if (file == NULL) {
    free(data);
    return NULL;
  }

  fread(data, sizeof(int), N * N, file);
  fclose(file);
  return data;
}

void check(int* init, int* expected, int k) {
  int* data = malloc(N * N * sizeof(int));
  if (data == NULL) {
    return;
  }

  memcpy(data, init, N * N * sizeof(int));
  ising(data, kW, k, N);
  if (memcmp(data, expected, N * N * sizeof(int)) == 0) {
    printf("k %d: correct\n", k);
  } else {
    printf("k %d: wrong\n", k);
  }

  free(data);
}

int main() {
  int* c0 = load_bin("conf-init.bin");
  int* c1 = load_bin("conf-1.bin");
  int* c4 = load_bin("conf-4.bin");
  int* c11 = load_bin("conf-11.bin");
  if (c0 == NULL || c1 == NULL || c4 == NULL || c11 == NULL) {
    exit(-1);
  }

  check(c0, c1, 1);
  check(c0, c4, 4);
  check(c0, c11, 11);

  free(c0);
  free(c1);
  free(c4);
  free(c11);
  return 0;
}
