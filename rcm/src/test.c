#include "rcm.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define SAVE
// #define COMPARE

#ifdef COMPARE
#include "depend/rcm.hpp"
#endif

static char* kSavePath = "perm.csv";

static void* my_malloc(size_t size) {
  void* p = malloc(size);
  if (p == NULL) {
    printf("out of memory!\n");
    exit(-1);
  }

  return p;
}

// Convert a list of coordinates to a CSR sparse matrix.
static void convert_csr(
    int n, int nnz, int* rlist, int* clist, int* ridx, int* cidx) {
  // Count nonzeros per row.
  memset(ridx, 0, n * sizeof(int));
  for (int i = 0; i < nnz; ++i) {
    ridx[rlist[i]]++;
  }

  // Prefix sum.
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    int tmp = ridx[i];
    ridx[i] = sum;
    sum += tmp;
  }
  ridx[n] = nnz;

  // Write column indices.
  for (int i = 0; i < nnz; ++i) {
    int row = rlist[i];
    int dst = ridx[row];
    cidx[dst] = clist[i];
    ++ridx[row];
  }

  int last = 0;
  for (int i = 0; i < n; ++i) {
    int tmp = ridx[i];
    ridx[i] = last;
    last = tmp;
  }
}

// Load a sparse matrix from a binary file.
// The format of the file is:
// n [4 bytes]
// nnz [4 bytes]
// rowidx [4*n bytes]
// colidx [4*n bytes]
static void load_bin(
    char* path, int* n_out, int* nnz_out, int** ridx_out, int** cidx_out) {
  FILE* f = fopen(path, "rb");
  if (f == NULL) {
    printf("error: fopen %s\n", path);
    exit(-1);
  }

  // Read header.
  int idx[2];
  fread(idx, sizeof(int), 2, f);
  if (feof(f)) {
    printf("unexpected eof\n");
    exit(-1);
  }

  // Read coordinate list.
  int n = idx[0], nnz = idx[1];
  int* data = my_malloc(2 * nnz * sizeof(int));
  fread(data, sizeof(int), 2 * nnz, f);
  if (feof(f)) {
    printf("unexpected eof\n");
    exit(-1);
  }

  fclose(f);

  int* ridx = my_malloc((n + 1) * sizeof(int));
  int* cidx = my_malloc(nnz * sizeof(int));
  convert_csr(n, nnz, data, data + nnz, ridx, cidx);

  *n_out = n;
  *nnz_out = nnz;
  *ridx_out = ridx;
  *cidx_out = cidx;

  free(data);
}

// Save the permutation to a CSV file.
static void save_perm(char* path, int n, int* perm) {
  FILE* f = fopen(path, "w");
  if (f == NULL) {
    printf("error: fopen %s\n", path);
    return;
  }

  for (int i = 0; i < n; ++i) {
    fprintf(f, "%d", perm[i] + 1);
    if (i != n - 1) {
      fprintf(f, ", ");
    }
  }

  fprintf(f, "\n");
  fclose(f);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("usage: %s filename\n", argv[0]);
    return 0;
  }

  int n, nnz, *ridx, *cidx;
  load_bin(argv[1], &n, &nnz, &ridx, &cidx);
  printf("%s: n=%d nnz=%d\n", argv[1], n, nnz);

  int* perm = my_malloc(n * sizeof(int));

  double time_rcm = -omp_get_wtime();
  rcm(n, ridx, cidx, perm);
  time_rcm += omp_get_wtime();
  printf("rcm: %.3fs\n", time_rcm);

#ifdef SAVE
  save_perm(kSavePath, n, perm);
#endif

#ifdef COMPARE
  // Fortran convention.
  for (int i = 0; i < n + 1; ++i) {
    ++ridx[i];
  }
  for (int i = 0; i < nnz; ++i) {
    ++cidx[i];
  }

  double time_other = -omp_get_wtime();
  genrcm(n, nnz, ridx, cidx, perm);
  time_other += omp_get_wtime();
  printf("other: %.3fs\n", time_other);

  for (int i = 0; i < n; ++i) {
    --perm[i];
  }
#endif

  free(ridx);
  free(cidx);
  free(perm);
  return 0;
}
