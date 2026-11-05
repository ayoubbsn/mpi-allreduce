/*
 * Ring allreduce (reduce-scatter + allgather).
 *
 * The vector is split into p chunks (sizes differ by at most one element).
 *
 * Reduce-scatter, steps s = 0 .. p-2: rank r sends chunk (r - s) to its right
 * neighbour and receives chunk (r - s - 1) from its left neighbour, adding it
 * to its own copy. After p-1 steps rank r owns the full sum of chunk (r + 1).
 *
 * Allgather, steps s = 0 .. p-2: rank r sends chunk (r + 1 - s) and receives
 * chunk (r - s) directly into the output buffer.
 *
 * Each rank sends and receives 2 (p-1)/p * n elements in total, independent
 * of p, which makes the ring bandwidth-optimal. The price is 2 (p-1) message
 * latencies instead of log2(p).
 *
 * In the reduce-scatter phase chunks travel in segments: the incoming
 * segments land in two alternating buffers, so the addition of segment k runs
 * while segment k+1 is in flight, and at most SEND_WINDOW outgoing segments
 * are pending at once. Every chunk is reduced by exactly one rank and then
 * copied around, so all ranks end with identical bits.
 */
#include "allreduce.h"
#include "common.h"

#include <string.h>

#define TAG_RS 201
#define TAG_AG 202
#define DEFAULT_SEGMENT (64 * 1024)
#define SEND_WINDOW 4

static size_t segment_bytes = DEFAULT_SEGMENT;

void allreduce_ring_set_segment(size_t bytes)
{
    segment_bytes = bytes > 0 ? bytes : DEFAULT_SEGMENT;
}

static void chunk_range(int count, int size, int idx, int *off, int *len)
{
    int q = count / size;
    int r = count % size;
    *len = q + (idx < r ? 1 : 0);
    *off = idx * q + (idx < r ? idx : r);
}

static int nsegments(int len, int seg)
{
    return (len + seg - 1) / seg;
}

/* Length of segment k of a chunk of `len` elements. */
static int seg_len(int len, int seg, int k)
{
    int rest = len - k * seg;
    return rest < seg ? rest : seg;
}

int allreduce_ring(const void *sendbuf, void *recvbuf, int count,
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

    char *buf = recvbuf;
    if (sendbuf != MPI_IN_PLACE && count > 0)
        memcpy(buf, sendbuf, (size_t)count * esz);
    if (size == 1 || count == 0)
        return MPI_SUCCESS;

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    int max_chunk = count / size + (count % size != 0);
    size_t seg_elems = segment_bytes / esz;
    int seg = seg_elems < 1 ? 1 : seg_elems > (size_t)max_chunk ? max_chunk : (int)seg_elems;

    char *tmp = ar_scratch(AR_SCRATCH_RING, 2 * (size_t)seg * esz);
    if (tmp == NULL)
        return MPI_ERR_NO_MEM;
    char *slot[2] = { tmp, tmp + (size_t)seg * esz };
    MPI_Request rreq[2];
    MPI_Request sreq[SEND_WINDOW];

    /* Reduce-scatter */
    for (int s = 0; s < size - 1; s++) {
        int soff, slen, roff, rlen;
        chunk_range(count, size, (rank - s + size) % size, &soff, &slen);
        chunk_range(count, size, (rank - s - 1 + size) % size, &roff, &rlen);
        int nsend = nsegments(slen, seg);
        int nrecv = nsegments(rlen, seg);
        int nmax = nsend > nrecv ? nsend : nrecv;

        for (int k = 0; k < 2 && k < nrecv; k++) {
            err = MPI_Irecv(slot[k], seg_len(rlen, seg, k), dtype, left, TAG_RS,
                            comm, &rreq[k]);
            if (err != MPI_SUCCESS)
                return err;
        }
        for (int k = 0; k < SEND_WINDOW; k++)
            sreq[k] = MPI_REQUEST_NULL;

        /* At most SEND_WINDOW sends and two receives in flight. */
        for (int k = 0; k < nmax; k++) {
            if (k < nsend) {
                MPI_Request *r = &sreq[k % SEND_WINDOW];
                if ((err = MPI_Wait(r, MPI_STATUS_IGNORE)) != MPI_SUCCESS)
                    return err;
                char *p = buf + ((size_t)soff + (size_t)k * seg) * esz;
                err = MPI_Isend(p, seg_len(slen, seg, k), dtype, right, TAG_RS, comm, r);
                if (err != MPI_SUCCESS)
                    return err;
            }
            if (k < nrecv) {
                int b = k % 2;
                if ((err = MPI_Wait(&rreq[b], MPI_STATUS_IGNORE)) != MPI_SUCCESS)
                    return err;
                ar_sum(buf + ((size_t)roff + (size_t)k * seg) * esz, slot[b],
                       seg_len(rlen, seg, k), dtype);
                if (k + 2 < nrecv) {
                    err = MPI_Irecv(slot[b], seg_len(rlen, seg, k + 2), dtype, left,
                                    TAG_RS, comm, &rreq[b]);
                    if (err != MPI_SUCCESS)
                        return err;
                }
            }
        }
        err = MPI_Waitall(SEND_WINDOW, sreq, MPI_STATUSES_IGNORE);
        if (err != MPI_SUCCESS)
            return err;
    }

    /* Allgather */
    for (int s = 0; s < size - 1; s++) {
        int soff, slen, roff, rlen;
        MPI_Request req[2];
        chunk_range(count, size, (rank + 1 - s + size) % size, &soff, &slen);
        chunk_range(count, size, (rank - s + size) % size, &roff, &rlen);
        err = MPI_Irecv(buf + (size_t)roff * esz, rlen, dtype, left, TAG_AG, comm, &req[0]);
        if (err != MPI_SUCCESS)
            return err;
        err = MPI_Isend(buf + (size_t)soff * esz, slen, dtype, right, TAG_AG, comm, &req[1]);
        if (err != MPI_SUCCESS)
            return err;
        err = MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
        if (err != MPI_SUCCESS)
            return err;
    }
    return MPI_SUCCESS;
}
