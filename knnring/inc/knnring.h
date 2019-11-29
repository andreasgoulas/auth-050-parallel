#ifndef KNNRING_H
#define KNNRING_H

typedef struct knnresult {
  int* nidx;      //!< Indices (0-based) of nearest neighbors [m-by-k]
  double* ndist;  //!< Distance of nearest neighbors          [m-by-k]
  int m;          //!< Number of query points                 [scalar]
  int k;          //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [n-by-d]
//! \param X Corpus data points      [n-by-d]
//! \param Y Query data points       [m-by-d]
//! \param n Number of corpus points [scalar]
//! \param m Number of query points  [scalar]
//! \param d Number of dimensions    [scalar]
//! \param k Number of neighbors     [scalar]
//! \return The kNN result
knnresult kNN(double* x, double* y, int n, int m, int d, int k);

//! Compute distributed all-kNN of points in X
//! \param X Data points           [n-by-d]
//! \param n Number of data points [scalar]
//! \param d Number of dimensions  [scalar]
//! \param k Number of neighbors   [scalar]
//! \retur n The kNN result
knnresult distrAllkNN(double* x, int n, int d, int k);

#endif
