#include "vptree.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <pthread.h>

#define N_DTHREADS 4
int n_threads = 0, max_threads = 32, threshold = 10000;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

#define SWAP(x, y) \
  point tmp = x; \
  x = y; \
  y = tmp;

typedef struct {
  int idx;
  double dist;
} point;

typedef struct {
  int d;
  double* x;
  point* points;
  int left, right;
} subtree_args;

typedef struct {
  int d;
  double* vp;
  double* x;
  point* points;
  int left, right;
} distance_args;

int partition(point* arr, int left, int right) {
  double pivot = arr[right].dist;
  int index = left;
  for (int j = left; j <= right - 1; ++j) {
    if (arr[j].dist <= pivot) {
      SWAP(arr[index], arr[j]);
      index++;
    }
  }

  SWAP(arr[index], arr[right]);
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

double distance(double* a, double *b, int d) {
  double dist = 0.0;
  for (int i = 0; i < d; ++i) {
    double diff = a[i] - b[i];
    dist += diff * diff;
  }

  return sqrt(dist);
}

void* distances(void* p) {
  distance_args* args = (distance_args*)p;
  int d = args->d, left = args->left, right = args->right;
  for (int i = left; i <= right; ++i) {
    int idx = args->points[i].idx;
    args->points[i].dist = distance(&args->x[idx * d], args->vp, d);
  }

  return NULL;
}

vptree* subtree_seq(subtree_args* args) {
  int d = args->d, left = args->left, right = args->right;
  int n = right - left + 1;
  if (n == 0) {
    return NULL;
  }

  vptree* tree = malloc(sizeof(vptree));
  if (tree == NULL) {
    return NULL;
  }

  tree->idx = args->points[right].idx;
  tree->vp = &args->x[tree->idx * d];

  if (n == 1) {
    tree->md = 0.0;
    tree->inner = NULL;
    tree->outer = NULL;
    return tree;
  }

  for (int i = left; i <= right - 1; ++i) {
    int idx = args->points[i].idx;
    args->points[i].dist = distance(&args->x[idx * d], tree->vp, d);
  }

  int k = (left + right - 1) / 2;
  tree->md = qselect(args->points, left, right - 1, k);

  subtree_args inner_args = *args;
  inner_args.left = left;
  inner_args.right = k;

  subtree_args outer_args = *args;
  outer_args.left = k + 1;
  outer_args.right = right - 1;

  tree->inner = subtree_seq(&inner_args);
  tree->outer = subtree_seq(&outer_args);
  return tree;
}

void* subtree(void* p) {
  subtree_args* args = (subtree_args*)p;
  int d = args->d, left = args->left, right = args->right;
  int n = right - left + 1;
  if (n < threshold) {
    return subtree_seq(args);
  }

  vptree* tree = malloc(sizeof(vptree));
  if (tree == NULL) {
    return NULL;
  }

  tree->idx = args->points[right].idx;
  tree->vp = &args->x[tree->idx * d];

  int parallel = 0;
  pthread_t t[N_DTHREADS];
  distance_args dist_args[N_DTHREADS];
  pthread_mutex_lock(&lock);
  if (n_threads + N_DTHREADS <= max_threads) {
    parallel = 1;
    n_threads += N_DTHREADS;

    int step = (n - 1) / N_DTHREADS;
    for (int i = 0; i < N_DTHREADS; ++i) {
      dist_args[i].d = d;
      dist_args[i].vp = tree->vp;
      dist_args[i].x = args->x;
      dist_args[i].points = args->points;
      dist_args[i].left = left + i * step;
      dist_args[i].right = dist_args[i].left + step - 1;
    }

    dist_args[N_DTHREADS - 1].right = right - 1;
    for (int i = 0; i < N_DTHREADS; ++i) {
      pthread_create(&t[i], NULL, distances, &dist_args[i]);
    }
  }
  pthread_mutex_unlock(&lock);

  if (parallel) {
    for (int i = 0; i < N_DTHREADS; ++i) {
      pthread_join(t[i], NULL);
    }

    pthread_mutex_lock(&lock);
    n_threads -= N_DTHREADS;
    pthread_mutex_unlock(&lock);
  } else {
    for (int i = left; i <= right - 1; ++i) {
      int idx = args->points[i].idx;
      args->points[i].dist = distance(&args->x[idx * d], tree->vp, d);
    }
  }

  int k = (left + right - 1) / 2;
  tree->md = qselect(args->points, left, right - 1, k);

  subtree_args inner_args = *args;
  inner_args.left = left;
  inner_args.right = k;

  subtree_args outer_args = *args;
  outer_args.left = k + 1;
  outer_args.right = right - 1;

  parallel = 0;
  pthread_t tsub;
  pthread_mutex_lock(&lock);
  if (n_threads < max_threads) {
    parallel = 1;
    ++n_threads;
    pthread_create(&tsub, NULL, subtree, &inner_args);
  }
  pthread_mutex_unlock(&lock);

  tree->outer = (vptree*)subtree(&outer_args);

  if (parallel) {
    void* r;
    pthread_join(tsub, &r);
    tree->inner = (vptree*)r;

    pthread_mutex_lock(&lock);
    --n_threads;
    pthread_mutex_unlock(&lock);
  } else {
    tree->inner = (vptree*)subtree(&inner_args);
  }

  return tree;
}

vptree* buildvp(double* x, int n, int d) {
  if (n == 0) {
    return NULL;
  }

  subtree_args args;
  args.d = d;
  args.x = x;
  args.left = 0;
  args.right = n - 1;

  args.points = malloc(sizeof(point) * n);
  if (args.points == NULL) {
    return NULL;
  }

  for (int i = 0; i < n; ++i) {
    args.points[i].idx = i;
  }

  vptree* tree = (vptree*)subtree(&args);
  free(args.points);
  return tree;
}

vptree* getInner(vptree* t) {
  return t->inner;
}

vptree* getOuter(vptree* t) {
  return t->outer;
}

double getMD(vptree* t) {
  return t->md;
}

double* getVP(vptree* t) {
  return t->vp;
}

int getIDX(vptree* t) {
  return t->idx;
}
