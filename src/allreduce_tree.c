/*
 * Recursive doubling allreduce.
 *
 * With p a power of two, at step k rank r exchanges its whole partial sum with
 * rank r ^ 2^k and adds the received vector. After log2(p) steps every rank
 * holds the full sum. Because a + b == b + a in IEEE arithmetic, both partners
 * compute the same bits, so all ranks end with an identical result.
 *
 * For other p, let pof2 be the largest power of two <= p and rem = p - pof2.
 * Among the first 2*rem ranks, even ranks send their data to the next odd
 * rank and sit out; the remaining pof2 ranks run the butterfly; the odd ranks
 * then send the result back to their even neighbour.
 */
#include "allreduce.h"
#include "common.h"

#include <string.h>

#define TAG_FOLD 101
#define TAG_STEP 102
#define TAG_UNFOLD 103

int allreduce_tree(const void *sendbuf, void *recvbuf, int count,
                   MPI_Datatype dtype, MPI_Comm comm)
{
    int rank, size, err;
    size_t esz;

    if (count < 0)
        return MPI_ERR_COUNT;
    if ((err = ar_elem_size(dtype, &esz)) != MPI_SUCCESS)
        return err;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t bytes = (size_t)count * esz;
    if (sendbuf != MPI_IN_PLACE && bytes > 0)
        memcpy(recvbuf, sendbuf, bytes);
    if (size == 1 || count == 0)
        return MPI_SUCCESS;

    void *tmp = ar_scratch(AR_SCRATCH_TREE, bytes);
    if (tmp == NULL)
        return MPI_ERR_NO_MEM;

    int pof2 = 1;
    while (pof2 * 2 <= size)
        pof2 *= 2;
    int rem = size - pof2;

    int newrank;
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            err = MPI_Send(recvbuf, count, dtype, rank + 1, TAG_FOLD, comm);
            newrank = -1;
        } else {
            err = MPI_Recv(tmp, count, dtype, rank - 1, TAG_FOLD, comm,
                           MPI_STATUS_IGNORE);
            ar_sum(recvbuf, tmp, count, dtype);
            newrank = rank / 2;
        }
        if (err != MPI_SUCCESS)
            return err;
    } else {
        newrank = rank - rem;
    }

    if (newrank >= 0) {
        for (int mask = 1; mask < pof2; mask <<= 1) {
            int newdst = newrank ^ mask;
            int dst = newdst < rem ? newdst * 2 + 1 : newdst + rem;
            err = MPI_Sendrecv(recvbuf, count, dtype, dst, TAG_STEP,
                               tmp, count, dtype, dst, TAG_STEP,
                               comm, MPI_STATUS_IGNORE);
            if (err != MPI_SUCCESS)
                return err;
            ar_sum(recvbuf, tmp, count, dtype);
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2 == 1)
            err = MPI_Send(recvbuf, count, dtype, rank - 1, TAG_UNFOLD, comm);
        else
            err = MPI_Recv(recvbuf, count, dtype, rank + 1, TAG_UNFOLD, comm,
                           MPI_STATUS_IGNORE);
    }
    return err;
}
