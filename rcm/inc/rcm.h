#ifndef RCM_H
#define RCM_H

#ifdef __cplusplus
extern "C" {
#endif

//! Find the reverse Cuthill-McKee ordering for a sparse matrix.
//! The matrix is stored in CSR format.
//! \param n Number of rows.
//! \param ridx Indices in cidx where the given row starts. [1-by-n]
//! \param cidx Column indices of nonzero entries. [1-by-nnz]
//! \param order New vertex ordering. [1-by-n]
void rcm(int n, int* ridx, int* cidx, int* order);

#ifdef __cplusplus
}
#endif

#endif
