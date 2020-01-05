#include "ising.h"

#include <stdlib.h>
#include <string.h>

void ising(int* g, double* w, int k, int n) {
  int* tmp = malloc(n * n * sizeof(int));
  if (tmp == NULL) {
    return;
  }

  int* g_prev = g;
  int* g_next = tmp;

  int i;
  for (i = 0; i < k; ++i) {
    int count = 0;
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        double sum = 0.0;
        for (int dy = 0; dy < 5; ++dy) {
          for (int dx = 0; dx < 5; ++dx) {
            int xx = (x + dx - 2 + n) % n;
            int yy = (y + dy - 2 + n) % n;
            sum += w[dx + 5 * dy] * g_prev[xx + n * yy];
          }
        }

        int v, prev = g_prev[x + n * y];
        if (sum > 1e-6) {
          v = 1;
        } else if (sum < -1e-6) {
          v = -1;
        } else {
          v = prev;
        }

        g_next[x + n * y] = v;
        if (v != prev) {
          ++count;
        }
      }
    }

    int* tmp = g_prev;
    g_prev = g_next;
    g_next = tmp;

    if (count == 0) {
      ++i;
      break;
    }
  }

  if (i % 2 == 1) {
    memcpy(g, tmp, n * n * sizeof(int));
  }

  free(tmp);
}
