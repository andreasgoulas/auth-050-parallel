#include "vptree.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cilk/cilk.h>

int threshold = 10000;

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
  while (k < right - 1 && args->points[k + 1].dist <= tree->md) {
    ++k;
  }

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

vptree* subtree(subtree_args* args) {
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

  cilk_for (int i = left; i <= right - 1; ++i) {
    int idx = args->points[i].idx;
    args->points[i].dist = distance(&args->x[idx * d], tree->vp, d);
  }

  int k = (left + right - 1) / 2;
  tree->md = qselect(args->points, left, right - 1, k);
  while (k < right - 1 && args->points[k + 1].dist <= tree->md) {
    ++k;
  }

  subtree_args inner_args = *args;
  inner_args.left = left;
  inner_args.right = k;

  subtree_args outer_args = *args;
  outer_args.left = k + 1;
  outer_args.right = right - 1;

  tree->inner = cilk_spawn subtree(&inner_args);
  tree->outer = subtree(&outer_args);
  cilk_sync;
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

  vptree* tree = subtree(&args);
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
