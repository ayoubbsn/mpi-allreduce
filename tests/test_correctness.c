/*
 * Correctness and non-regression test for allreduce_tree and allreduce_ring.
 *
 * For each datatype (double, float), buffer size and implementation, checks:
 *   - the result against a sequential sum of every rank's input;
 *   - the result against the native MPI_Allreduce;
 *   - that all ranks hold bit-identical results;
 *   - that sendbuf is left untouched (and that MPI_IN_PLACE works).
 *
 * Tolerance: 1e-12 for double, 1e-4 for float (absolute; inputs are in [-1, 1]).
 * Exit status is 0 when every case passes on every rank.
 */
#include "allreduce.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*allreduce_fn)(const void *, void *, int, MPI_Datatype, MPI_Comm);

static int ring_small_segment(const void *s, void *r, int n, MPI_Datatype t, MPI_Comm c)
{
    /* 24 bytes: 3 doubles or 6 floats per segment, exercises the pipeline. */
    allreduce_ring_set_segment(24);
    int err = allreduce_ring(s, r, n, t, c);
    allreduce_ring_set_segment(0);
    return err;
}

static const struct {
    const char *name;
    allreduce_fn fn;
} impls[] = {
    { "tree", allreduce_tree },
    { "ring", allreduce_ring },
    { "ring-seg24", ring_small_segment },
};
#define NIMPLS ((int)(sizeof(impls) / sizeof(impls[0])))

static const int counts[] = {
    0, 1, 2, 3, 4, 5, 7, 8, 13, 31, 64, 100, 1000, 1023, 1025,
    4096, 4099, 65537, (1 << 20) + 3,
};
#define NCOUNTS ((int)(sizeof(counts) / sizeof(counts[0])))

/* Deterministic input in [-1, 1) that is not exactly representable, so the
 * order of additions matters. */
static double input(int rank, long i)
{
    return (double)((i * 7919L + (rank + 1) * 104729L) % 2003) / 997.0 - 1.0;
}

static void fill(void *buf, MPI_Datatype t, int rank, int n)
{
    for (int i = 0; i < n; i++) {
        if (t == MPI_DOUBLE)
            ((double *)buf)[i] = input(rank, i);
        else
            ((float *)buf)[i] = (float)input(rank, i);
    }
}

/* Sequential sum in rank order, in the working precision. */
static void reference(void *buf, MPI_Datatype t, int nprocs, int n)
{
    for (int i = 0; i < n; i++) {
        if (t == MPI_DOUBLE) {
            double s = 0.0;
            for (int r = 0; r < nprocs; r++)
                s += input(r, i);
            ((double *)buf)[i] = s;
        } else {
            float s = 0.0f;
            for (int r = 0; r < nprocs; r++)
                s += (float)input(r, i);
            ((float *)buf)[i] = s;
        }
    }
}

static double max_abs_diff(const void *a, const void *b, MPI_Datatype t, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double x = t == MPI_DOUBLE ? ((const double *)a)[i] : ((const float *)a)[i];
        double y = t == MPI_DOUBLE ? ((const double *)b)[i] : ((const float *)b)[i];
        double d = fabs(x - y);
        if (!(d <= m)) /* also catches NaN */
            m = d;
    }
    return m;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int verbose = argc > 1 && !strcmp(argv[1], "-v");
    int failures = 0, cases = 0;

    MPI_Datatype types[2] = { MPI_DOUBLE, MPI_FLOAT };
    const char *tnames[2] = { "double", "float" };
    const double tol[2] = { 1e-12, 1e-4 };

    int max_count = counts[NCOUNTS - 1];
    size_t cap = (size_t)max_count * sizeof(double) + 1;
    char *send = malloc(cap), *orig = malloc(cap), *recv = malloc(cap);
    char *ref = malloc(cap), *native = malloc(cap), *root = malloc(cap);
    if (!send || !orig || !recv || !ref || !native || !root)
        MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0)
        printf("allreduce correctness test, %d ranks\n", size);

    for (int ti = 0; ti < 2; ti++) {
        MPI_Datatype t = types[ti];
        size_t esz = t == MPI_DOUBLE ? sizeof(double) : sizeof(float);

        for (int ci = 0; ci < NCOUNTS; ci++) {
            int n = counts[ci];
            size_t bytes = (size_t)n * esz;

            fill(send, t, rank, n);
            memcpy(orig, send, bytes);
            reference(ref, t, size, n);
            MPI_Allreduce(send, native, n, t, MPI_SUM, MPI_COMM_WORLD);

            for (int ii = 0; ii < NIMPLS; ii++) {
                for (int inplace = 0; inplace < 2; inplace++) {
                    int rc;
                    memset(recv, 0xff, bytes); /* NaN pattern */
                    if (inplace) {
                        memcpy(recv, send, bytes);
                        rc = impls[ii].fn(MPI_IN_PLACE, recv, n, t, MPI_COMM_WORLD);
                    } else {
                        rc = impls[ii].fn(send, recv, n, t, MPI_COMM_WORLD);
                    }

                    double err[2] = {
                        max_abs_diff(recv, ref, t, n),
                        max_abs_diff(recv, native, t, n),
                    };
                    int bad = rc != MPI_SUCCESS;
                    bad |= !(err[0] <= tol[ti]) || !(err[1] <= tol[ti]);
                    bad |= memcmp(send, orig, bytes) != 0;

                    if (rank == 0)
                        memcpy(root, recv, bytes);
                    MPI_Bcast(root, (int)bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
                    int differs = memcmp(root, recv, bytes) != 0;
                    bad |= differs;

                    double gerr[2];
                    int gbad, gdiffers;
                    MPI_Reduce(err, gerr, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                    MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
                    MPI_Reduce(&differs, &gdiffers, 1, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

                    cases++;
                    failures += gbad;
                    if (rank == 0 && (gbad || verbose))
                        printf("%s %-10s %-6s n=%-8d inplace=%d err_ref=%.2e err_native=%.2e%s\n",
                               gbad ? "FAIL" : "ok  ", impls[ii].name, tnames[ti], n,
                               inplace, gerr[0], gerr[1],
                               gdiffers ? " (ranks differ)" : "");
                }
            }
        }
    }

    if (rank == 0)
        printf("%d/%d cases passed\n", cases - failures, cases);

    free(send);
    free(orig);
    free(recv);
    free(ref);
    free(native);
    free(root);
    allreduce_free_scratch();
    MPI_Finalize();
    return failures ? 1 : 0;
}
