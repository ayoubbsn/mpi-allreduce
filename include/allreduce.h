#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <mpi.h>

/*
 * Point-to-point implementations of MPI_Allreduce with MPI_SUM.
 *
 * Supported datatypes: MPI_FLOAT, MPI_DOUBLE.
 * sendbuf may be MPI_IN_PLACE (recvbuf then holds the input).
 * Every rank ends with a bit-identical result.
 *
 * Both functions keep a scratch buffer between calls so that the timed loop
 * does not pay for malloc/page faults. They are not thread-safe;
 * call allreduce_free_scratch() before MPI_Finalize to release it.
 *
 * Return MPI_SUCCESS, MPI_ERR_TYPE for an unsupported datatype,
 * MPI_ERR_COUNT for a negative count or MPI_ERR_NO_MEM.
 */

/* Recursive doubling (binomial butterfly) with the usual fold for
 * non-power-of-two process counts. log2(p) steps, each exchanging the whole
 * vector: cost ~ log2(p) * (alpha + n * beta). */
int allreduce_tree(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype dtype, MPI_Comm comm);

/* Ring: reduce-scatter followed by allgather, each over p-1 steps.
 * The incoming chunk of the reduce-scatter phase is received in segments so
 * that the reduction of one segment overlaps the transfer of the next.
 * cost ~ 2(p-1) * alpha + 2 (p-1)/p * n * beta. */
int allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype dtype, MPI_Comm comm);

/* Segment size used by allreduce_ring, in bytes (default 64 KiB). */
void allreduce_ring_set_segment(size_t bytes);

void allreduce_free_scratch(void);

#endif
