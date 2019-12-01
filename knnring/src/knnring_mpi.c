#include "knnring.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include <cblas.h>

// REDUCE controls whether to perform global reductions.
#define REDUCE 1

// TIME_COMM controls whether to measure the communication cost.
#define TIME_COMM 1

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

void merge(knnresult* out, knnresult* tmp, int n, int k) {
  for (int qi = 0; qi < n; ++qi) {
    for (int i = 0; i < k; ++i) {
      if (tmp->ndist[qi * k] < out->ndist[i + qi * k]) {
        double dist = tmp->ndist[qi * k];
        int idx = tmp->nidx[qi * k];

        out->ndist[i + qi * k] = dist;
        out->nidx[i + qi * k] = idx;

        int j;
        for (j = 1; j < k && tmp->ndist[j + qi * k] < dist; ++j) {
          tmp->ndist[j - 1 + qi * k] = tmp->ndist[j + qi * k];
          tmp->nidx[j - 1 + qi * k] = tmp->nidx[j + qi * k];
        }

        tmp->ndist[j - 1 + qi * k] = dist;
        tmp->nidx[j - 1 + qi * k] = idx;
      }
    }
  }
}

knnresult distrAllkNN(double* x, int n, int d, int k) {
  knnresult result;
  result.m = result.k = 0;
  result.nidx = NULL;
  result.ndist = NULL;

  double* y = malloc(n * d * sizeof(double));
  double* z = malloc(n * d * sizeof(double));
  if (y == NULL || z == NULL) {
    free(y);
    free(z);
    return result;
  }

  memcpy(y, x, n * d * sizeof(double));

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#if TIME_COMM
  double total_time = 0.0;
#endif

  for (int i = 0; i < size; ++i) {
    MPI_Request send_request, recv_request;
    if (i != size - 1) {
      int next = (rank + 1) % size;
      int prev = (rank - 1) % size;
      MPI_Isend(y, n * d, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &send_request);
      MPI_Irecv(z, n * d, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &recv_request);
    }

    knnresult curr = kNN(y, x, n, n, d, k);
    int src = rank - i - 1;
    if (src < 0) {
      src += size;
    }

    for (int j = 0; j < k * n; ++j) {
      curr.nidx[j] += src * n;
    }

    if (i == 0) {
      result = curr;
    } else {
      merge(&result, &curr, n, k);
      free(curr.nidx);
      free(curr.ndist);
    }

    if (i != size - 1) {
#if TIME_COMM
      double time_start = MPI_Wtime();
#endif
      MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
      MPI_Wait(&send_request, MPI_STATUS_IGNORE);
#if TIME_COMM
      double time_end = MPI_Wtime();
      total_time += time_end - time_start;
#endif

      SWAP(double*, y, z);
    }
  }

  free(y);
  free(z);

#if REDUCE
  double min = DBL_MAX, max = -DBL_MAX;
  for (int i = 0; i < result.m * result.k; ++i) {
    if (result.ndist[i] > 1e-6 && result.ndist[i] < min) {
      min = result.ndist[i];
    }

    if (result.ndist[i] > max) {
      max = result.ndist[i];
    }
  }

  double global_min, global_max;
  MPI_Reduce(&min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Global min: %f\nGlobal max: %f\n", global_min, global_max);
  }
#endif

#if TIME_COMM
  double max_time;
  MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Mean Comm Time: %fs\n", total_time / size);
  }
#endif

  return result;
}
