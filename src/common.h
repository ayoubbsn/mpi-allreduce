#ifndef ALLREDUCE_COMMON_H
#define ALLREDUCE_COMMON_H

#include <stddef.h>
#include <mpi.h>

/* Internal helpers shared by the allreduce implementations. */

enum { AR_SCRATCH_TREE = 0, AR_SCRATCH_RING, AR_SCRATCH_SLOTS };

/* Element size for a supported datatype, MPI_ERR_TYPE otherwise. */
int ar_elem_size(MPI_Datatype dtype, size_t *size);

/* inout[i] += in[i] for i < n. */
void ar_sum(void *inout, const void *in, int n, MPI_Datatype dtype);

/* Scratch buffer of at least `bytes` bytes, kept between calls. NULL on failure. */
void *ar_scratch(int slot, size_t bytes);

#endif
