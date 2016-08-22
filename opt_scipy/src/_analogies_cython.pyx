from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange


@cython.boundscheck(False)
def calculate_analogies_cython_helper(str w_a, str w_b, str w_c,
                                      double [:] A_memview, double [:] B_memview,
                                      double [:] C_memview, double [:,:] vectors_memview):
    # build the X array for comparison
    cdef double[:] X_memview = np.add(np.subtract(B_memview, A_memview), C_memview)

    # hardcoded variable for dimensions, figure it out dynamically if i have time to
    # come back and change it
    cdef size_t dimensions = 300

    # keep track of the max cosine similarity and the index of its associated w_d
    cdef double max_cosine_similarity
    cdef size_t w_d_idx

    # temp variable for the word vector we're currently comparing
    cdef double[:] d_candidate
    cdef double[:] similarities
    cdef double d_cos_similarity

    # keep track of the number of vectors
    cdef size_t num_vectors = vectors_memview.shape[0]

    # temp variable for iteration, since we can't dynamically generate them
    # in the loop declaration
    cdef size_t i = 0
    with nogil:
        for i in range(num_vectors):
            if(memview_equals(vectors_memview[i], A_memview, dimensions)
               or memview_equals(vectors_memview[i], B_memview, dimensions)
               or memview_equals(vectors_memview[i], C_memview, dimensions)):
                continue
            d_cos_similarity = cosine_sim_cython_nogil(vectors_memview[i], X_memview, dimensions)
            if d_cos_similarity > max_cosine_similarity:
                max_cosine_similarity = d_cos_similarity
                w_d_idx = i

    return w_d_idx

@cython.boundscheck(False)
cdef bint memview_equals(double[:] X, double[:] Y, size_t size) nogil:
    cdef size_t i

    for i in range(size):
        if X[i] != Y[i]:
            return 0
    return 1

@cython.boundscheck(False)
cdef double cosine_sim_cython_nogil(double[:] A, double[:] B, size_t size) nogil:
    cdef size_t i
    cdef double dot_product = 0.0
    cdef double mag_A = 0.0
    cdef double mag_B = 0.0

    for i in prange(size, schedule='guided', num_threads=4):
        dot_product += A[i] * B[i]
        mag_A += A[i] * A[i]
        mag_B += B[i] * B[i]
    mag_A = sqrt(mag_A)
    mag_b = sqrt(mag_B)
    return dot_product / (mag_A * mag_B)
