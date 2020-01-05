#include "ising.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_SIZE 96   // Points per block
#define BLOCK_SIZE 16  // Threads per block

#define CHECK(x) do {                                \
    cudaError_t err = (x);                           \
    if (err != cudaSuccess) {                        \
      printf("error %s\n", cudaGetErrorString(err)); \
    }                                                \
  } while(0);

__global__
void compute(int* g_prev, int* g_next, double* w, int n) {
  int x0 = TILE_SIZE * blockIdx.x;
  int y0 = TILE_SIZE * blockIdx.y;

  __shared__ double w_cache[5][5];
  if (threadIdx.x < 5 && threadIdx.y < 5) {
    w_cache[threadIdx.x][threadIdx.y] = w[threadIdx.x + 5 * threadIdx.y];
  }

  __shared__ int g_cache[TILE_SIZE + 4][TILE_SIZE + 4];
  for (int y = threadIdx.y; y < TILE_SIZE + 4; y += BLOCK_SIZE) {
    for (int x = threadIdx.x; x < TILE_SIZE + 4; x += BLOCK_SIZE) {
      int xx = (x0 + x - 2 + n) % n;
      int yy = (y0 + y - 2 + n) % n;
      g_cache[x][y] = g_prev[xx + n * yy];
    }
  }

  __syncthreads();

  for (int y = y0 + threadIdx.y; y < y0 + TILE_SIZE; y += BLOCK_SIZE) {
    for (int x = x0 + threadIdx.x; x < x0 + TILE_SIZE; x += BLOCK_SIZE) {
      if (x >= n || y >= n) {
        continue;
      }

      int cache_x = (x % TILE_SIZE) + 2;
      int cache_y = (y % TILE_SIZE) + 2;

      double sum = 0.0;
      for (int dy = 0; dy < 5; ++dy) {
        for (int dx = 0; dx < 5; ++dx) {
          int xx = cache_x + dx - 2;
          int yy = cache_y + dy - 2;
          sum += w_cache[dx][dy] * g_cache[xx][yy];
        }
      }

      int v;
      if (sum > 1e-6) {
        v = 1;
      } else if (sum < -1e-6) {
        v = -1;
      } else {
        v = g_cache[cache_x][cache_y];
      }

      g_next[x + n * y] = v;
    }
  }
}

extern "C"
void ising(int* g, double* w, int k, int n) {
  double* dev_w;
  int* dev_g_prev, *dev_g_next;
  CHECK(cudaMalloc(&dev_w, 5 * 5 * sizeof(double)));
  CHECK(cudaMalloc(&dev_g_prev,n * n * sizeof(int)));
  CHECK(cudaMalloc(&dev_g_next, n * n * sizeof(int)));

  CHECK(cudaMemcpy(dev_w, w, 5 * 5 * sizeof(double), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dev_g_prev, g, n * n * sizeof(int), cudaMemcpyHostToDevice));

  int num_blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
  uint3 dim_block = make_uint3(BLOCK_SIZE, BLOCK_SIZE, 1);
  uint3 dim_grid = make_uint3(num_blocks, num_blocks, 1);

  for (int i = 0; i < k; ++i) {
    compute<<<dim_grid, dim_block>>>(dev_g_prev, dev_g_next, dev_w, n);
    CHECK(cudaGetLastError());

    int* tmp = dev_g_prev;
    dev_g_prev = dev_g_next;
    dev_g_next = tmp;
  }

  CHECK(cudaMemcpy(g, dev_g_prev, n * n * sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(dev_w);
  cudaFree(dev_g_prev);
  cudaFree(dev_g_next);
}
