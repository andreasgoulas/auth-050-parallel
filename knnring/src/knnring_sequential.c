#include "knnring.h"

#include <math.h>
#include <stdlib.h>

#include <cblas.h>

#define SWAP(T, x, y) \
  do { \
    T tmp = x; \
    x = y; \
    y = tmp; \
  } while (0);

typedef struct {
  int idx;
  double dist;
} point;

int point_cmp(const void* p1, const void* p2) {
  point* a = (point*)p1;
  point* b = (point*)p2;
  return a->dist < b->dist ? -1 : 1;
}

int partition(point* arr, int left, int right) {
  double pivot = arr[right].dist;
  int index = left;
  for (int j = left; j <= right - 1; ++j) {
    if (arr[j].dist <= pivot) {
      SWAP(point, arr[index], arr[j]);
      index++;
    }
  }

  SWAP(point, arr[index], arr[right]);
  return index;
}

double qselect(point* arr, int left, int right, int k) {
  if (left == right) {
    return arr[left].dist;
  }

  int index = partition(arr, left, right);
  if (k == index) {
    return arr[index].dist;
  } else if (k < index) {
    return qselect(arr, left, index - 1, k);
  } else {
    return qselect(arr, index + 1, right, k);
  }
}

void sum_square(double* m, double* result, int n, int d) {
  for (int i = 0; i < n; ++i) {
    result[i] = 0.0;
    for (int j = 0; j < d; ++j) {
      double f = m[j + i * d];
      result[i] += f * f;
    }
  }
}

knnresult kNN(double* x, double* y, int n, int m, int d, int k) {
  knnresult result;
  result.nidx = NULL;
  result.ndist = NULL;
  result.m = result.k = 0;

  double* x_sqr = malloc(n * d * sizeof(double));
  double* y_sqr = malloc(m * d * sizeof(double));
  double* d_mat = malloc(n * m * sizeof(double));
  if (x_sqr == NULL || y_sqr == NULL || d_mat == NULL) {
    free(x_sqr);
    free(y_sqr);
    free(d_mat);
    return result;
  }

  sum_square(x, x_sqr, n, d);
  sum_square(y, y_sqr, m, d);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2.0,
      x, d, y, d, 0.0, d_mat, m);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      d_mat[j + i * m] += x_sqr[i] + y_sqr[j];
    }
  }

  free(x_sqr);
  free(y_sqr);

  point* points = malloc(n * sizeof(point));
  result.nidx = malloc(m * k * sizeof(int));
  result.ndist = malloc(m * k * sizeof(double));
  if (points == NULL || result.nidx == NULL || result.ndist == NULL) {
    free(points);
    free(result.nidx);
    free(result.ndist);
    free(d_mat);
    return result;
  }

  result.m = m;
  result.k = k;

  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      points[i].idx = i;
      points[i].dist = d_mat[j + i * m];
    }

    qselect(points, 0, n - 1, k - 1);
    qsort(points, k, sizeof(point), point_cmp);

    for (int i = 0; i < k; ++i) {
      result.nidx[i + j * k] = points[i].idx;

      double dist2 = points[i].dist;
      if (dist2 < 0.0) {
        result.ndist[i + j * k] = 0.0;
      } else {
        result.ndist[i + j * k] = sqrt(dist2);
      }
    }
  }

  free(points);
  free(d_mat);
  return result;
}

knnresult distrAllkNN(double* x, int n, int d, int k) {
  knnresult result;
  result.m = result.k = 0;
  result.nidx = NULL;
  result.ndist = NULL;
  return result;
}
