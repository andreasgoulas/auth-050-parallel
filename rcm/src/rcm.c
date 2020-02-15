#include "rcm.h"

#include <assert.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>

#define SWAP(T, x, y)  do { \
    T tmp = x; \
    x = y; \
    y = tmp; \
  } while(0);

#define MAX(x, y) ((x) > (y) ? (x) : (y))

static void* my_malloc(size_t size) {
  void* p = malloc(size);
  if (p == NULL) {
    printf("out of memory!\n");
    exit(-1);
  }

  return p;
}

static int prefix_sum_seq(int* arr, int len) {
  int sum = 0;
  for (int i = 0; i < len; ++i) {
    int tmp = arr[i];
    arr[i] = sum;
    sum += tmp;
  }

  return sum;
}

static int prefix_sum(int* arr, int len, int nthreads) {
  static int kLimit = 1024;
  if (len <= kLimit || nthreads == 1) {
    return prefix_sum_seq(arr, len);
  } else {
    int sum0, sum1, mid = len / 2;
    #pragma omp parallel sections
    {
      #pragma omp section
      sum0 = prefix_sum(arr, mid, nthreads / 2);
      #pragma omp section
      sum1 = prefix_sum(arr + mid, len - mid, nthreads - nthreads / 2);
    }

    #pragma omp parallel for
    for (int i = mid; i < len; ++i) {
      arr[i] += sum0;
    }

    return sum0 + sum1;
  }
}

static int find_components(
    int n, int* ridx, int* cidx, int** sizes_out, int** ord_out) {
  int max_threads = omp_get_max_threads();

  // Initialize the Union-Find algorithm.
  volatile int* p = my_malloc(n * sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    p[i] = i;
  }

  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int k = ridx[i]; k < ridx[i + 1]; ++k) {
      int j = cidx[k];
      int x = i, y = j;
      for (;;) {
        int px = p[x], py = p[y];
        if (px == py) {
          break;  // Same set.
        }

        // Merge the set with the larger index into the other set.
        if (px < py) {
          SWAP(int, x, y);
          SWAP(int, px, py);
        }

        // If x is the root, try to merge.
        if (px == x && __sync_bool_compare_and_swap(&p[x], px, py)) {
          break;
        }

        // Follow the chain up the tree.
        p[x] = py;
        x = px;
      }
    }
  }

  int count = 0;
  int* root_id = my_malloc(n * sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    // Path compression.
    int r = i;
    while (p[r] != r) {
      r = p[r];
    }
    p[i] = r;

    // Mark the component index of each root vertex.
    if (i == r) {
      #pragma omp atomic capture
      { root_id[i] = count; ++count; }
    }
  }

  // Count number of vertices per component.
  int* sizes = my_malloc(count * sizeof(int));
  memset(sizes, 0, count * sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    #pragma omp atomic
    ++sizes[root_id[p[i]]];
  }

  // Prefix sum.
  int* next_idx = my_malloc(count * sizeof(int));
  memcpy(next_idx, sizes, count * sizeof(int));
  prefix_sum(next_idx, count, max_threads);

  // Placement.
  int* ord = my_malloc(n * sizeof(int));
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    int index, comp_id = root_id[p[i]];
    #pragma omp atomic capture
    { index = next_idx[comp_id]; next_idx[comp_id]++; }
    ord[index] = i;
  }

  free((void*)p);
  free(root_id);
  free(next_idx);

  *sizes_out = sizes;
  *ord_out = ord;
  return count;
}

static int bfs(int n, int* ridx, int* cidx, int root,
    int* levels, int* queues[2],
    int* count_out, int* width_out, int* short_circuit) {
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    levels[i] = -1;
  }

  int* queue_prev = queues[0];
  int* queue_next = queues[1];

  int num_levels = 0, end = 0;
  levels[root] = 0;
  queue_next[end++] = root;
  while (end > 0) {
    // If requested, save tree width.
    if (width_out != NULL) {
      *width_out = MAX(end, *width_out);
    }

    // If requested, stop the search when the tree becomes too wide.
    if (short_circuit != NULL) {
      if (end > *short_circuit) {
        num_levels = -1;
        break;
      }
    }

    ++num_levels;
    SWAP(int*, queue_prev, queue_next);

    int count = 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < end; ++i) {
      int parent = queue_prev[i];
      for (int k = ridx[parent]; k < ridx[parent + 1]; ++k) {
        int j = cidx[k];

        // The data race here may cause the vertex to be added multiple times
        // to the queue. This is resolved in the next iteration.
        if (levels[j] == -1) {
          levels[j] = num_levels;

          int head;
          #pragma omp atomic capture
          { head = count; ++count; }

          assert(head < n);
          queue_next[head] = j;
        }
      }
    }

    // If requested, save the number of vertices at the last level.
    if (count_out != NULL) {
      *count_out = end;
    }

    end = count;
  }

  return num_levels;
}

static int find_min(int* vertices, int len, int* ridx) {
  int max_threads = omp_get_max_threads();
  int* local_min = my_malloc(max_threads * sizeof(int));
  int* local_min_idx = my_malloc(max_threads * sizeof(int));

  int idx = -1;
  #pragma omp parallel
  {
    // Find the local minimum.
    int min = INT_MAX, min_idx = -1;
    #pragma omp for
    for (int i = 0; i < len; ++i) {
      int v = vertices[i];
      int deg = ridx[v + 1] - ridx[v];
      if (deg < min) {
        min = deg;
        min_idx = v;
      }
    }

    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    local_min[tid] = min;
    local_min_idx[tid] = min_idx;

    // Find the global minimum.
    #pragma omp barrier
    #pragma omp master
    {
      min = INT_MAX;
      for (int i = 0; i < nthreads; ++i) {
        if (local_min[i] < min) {
          min = local_min[i];
          idx = local_min_idx[i];
        }
      }
    }
  }

  free(local_min);
  free(local_min_idx);
  return idx;
}

static int pseudo_diameter(int* vertices, int len, int* ridx, int* cidx,
    int* levels, int* queues[2]) {
  int s_width = 0, e_width = 0;
  int s = find_min(vertices, len, ridx);
  int e = -1;
  do {
    // Breadth-first search.
    int end;  // Number of candidates.
    int diameter = bfs(len, ridx, cidx, s, levels, queues,
        &end, &s_width, NULL);

    // Sort candidates (insertion sort).
    int* tmp = queues[diameter % 2];
    for (int i = 1; i < end; ++i) {
      int key = tmp[i];
      int j = i - 1;
      while (j >= 0) {
        int jv = tmp[j];
        if (ridx[jv + 1] - ridx[jv] <= ridx[key + 1] - ridx[key]) {
          break;
        }

        tmp[j + 1] = tmp[j];
        --j;
      }

      tmp[j + 1] = key;
    }

    // Select the first five non adjacent nodes.
    int count = 0;
    int cands[5];
    for (int i = 1; i < end && count < 5; ++i) {
      int v = tmp[i], adjacent = 0;
      // Check if the vertex alreay exists.
      for (int k = 0; k < count; ++k) {
        if (cands[k] == v) {
          adjacent = 1;
          break;
        }
      }

      if (adjacent) {
        continue;
      }

      // Check neigbors.
      for (int k = ridx[v]; k < ridx[v + 1]; ++k) {
        int j = cidx[k];
        for (int p = 0; p < count; ++p) {
          if (cands[p] == j) {
            adjacent = 1;
            break;
          }
        }
      }

      if (!adjacent) {
        cands[count++] = v;
      }
    }

    // BFS at each candidate.
    int min_width = INT_MAX;
    for (int i = 0; i < count; ++i) {
      int v = cands[i], width = 0;
      int c_diameter = bfs(len, ridx, cidx, v, levels, queues,
          NULL, &width, &min_width);
      if (c_diameter == -1) {
        continue;  // Short circuit.
      } else if (c_diameter > diameter && width < min_width) {
        s = v;
        e = -1;
        break;
      } else if (width < min_width) {
        min_width = width;
        e = v;
      }
    }

    e_width = min_width;
  } while (e == -1);

  if (e_width < s_width) {
    SWAP(int, e, s);
  }

  return s;
}

static void count_vertices(int* vertices, int len,
    int* counts, int* levels, int num_levels) {
  int max_threads = omp_get_max_threads();
  int* local_counts = my_malloc(max_threads * num_levels * sizeof(int));
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    memset(local_counts + tid * num_levels, 0, num_levels * sizeof(int));

    #pragma omp for
    for (int i = 0; i < len; ++i) {
      int level = levels[vertices[i]];
      ++local_counts[level + tid * num_levels];
    }
  }

  #pragma omp parallel for
  for (int l = 0; l < num_levels; ++l) {
    counts[l] = 0;
    for (int i = 0; i < max_threads; ++i) {
      counts[l] += local_counts[l + i * num_levels];
    }
  }

  free(local_counts);
}

void rcm(int n, int* ridx, int* cidx, int* perm) {
  if (n == 0) {
    return;
  }

  // Find connected components.
  int* csizes, *ord;
  int num_comp = find_components(n, ridx, cidx, &csizes, &ord);

  int max_threads = omp_get_max_threads();

  // Find the maximum degree.
  int max_deg = 0;
  #pragma omp parallel for reduction(max: max_deg)
  for (int i = 0; i < n; ++i) {
    int deg = ridx[i + 1] - ridx[i];
    max_deg = MAX(deg, max_deg);
  }

  // Distance from the root vertex.
  int* levels = my_malloc(n * sizeof(int));

  // Queues used during BFS.
  int* queues[2];
  queues[0] = my_malloc(n * sizeof(int));
  queues[1] = my_malloc(n * sizeof(int));

  // Number of vertices per level.
  int* counts = my_malloc(n * sizeof(int));

  // Thread-local array of children.
  int* local_children = my_malloc(max_deg * max_threads * sizeof(int));

  // Write offset into perm per level.
  volatile int* write_offset = my_malloc(n * sizeof(int));

  int nr = 0;  // Number of vertices in the result set.
  for (int ic = 0; ic < num_comp; ++ic) {
    // Select the root vertex.
    int len = csizes[ic];
    int root = pseudo_diameter(ord + nr, len, ridx, cidx, levels, queues);
    assert(root != -1);

    // Breadth-first search.
    int num_levels = bfs(n, ridx, cidx, root, levels, queues, NULL, NULL, NULL);

    // Count vertices per level.
    count_vertices(ord + nr, len, counts, levels, num_levels);

    // Prefix sum.
    counts[num_levels] = prefix_sum(counts, num_levels, max_threads);

    // Placement.
    perm[n - nr - 1] = root;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      for (int l = tid; l < num_levels; l += nthreads) {
        write_offset[l] = nr + counts[l];
      }

      #pragma omp master
      ++write_offset[0];
      #pragma omp barrier

      int* children = local_children + tid * max_deg;
      for (int l = tid; l < num_levels; l += nthreads) {
        int read_idx = nr + counts[l];
        while (read_idx != nr + counts[l + 1]) {
          while (read_idx == write_offset[l]);  // Spin
          int v = perm[n - read_idx - 1];
          ++read_idx;

          // Gather children.
          int count = 0;
          for (int k = ridx[v]; k < ridx[v + 1]; ++k) {
            int j = cidx[k];
            if (levels[j] == l + 1) {
              levels[j] = -1;
              children[count] = j;
              ++count;
            }
          }

          // Sort children (insertion sort).
          for (int i = 1; i < count; ++i) {
            int key = children[i];
            int j = i - 1;
            while (j >= 0) {
              int jv = children[j];
              if (ridx[jv + 1] - ridx[jv] <= ridx[key + 1] - ridx[key]) {
                break;
              }

              children[j + 1] = children[j];
              --j;
            }

            children[j + 1] = key;
          }

          // Write children.
          for (int i = 0; i < count; ++i) {
            int v = children[i];
            int idx = n - write_offset[l + 1] - 1;
            perm[idx] = v;
            ++write_offset[l + 1];
          }
        }
      }
    }

    nr += len;
  }

  free(csizes);
  free(ord);

  free(levels);
  free(queues[0]);
  free(queues[1]);
  free(counts);
  free(local_children);
  free((void*)write_offset);
}

