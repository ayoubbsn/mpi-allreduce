/*
 * Allreduce benchmark: native MPI_Allreduce vs tree vs ring, MPI_DOUBLE sum.
 *
 * For each message size (powers of two from --min to --max bytes):
 *   - warmup calls, then `iters` timed calls between two MPI_Barrier;
 *   - the per-call time of a rank is (t1 - t0) / iters, the reported time is
 *     the maximum over ranks;
 *   - this is repeated `reps` times and the smallest value is kept.
 *
 * Output (rank 0, CSV): bytes,iters,native_us,tree_us,ring_us,max_abs_err
 * max_abs_err is the largest |tree - native| or |ring - native| over all
 * elements and ranks.
 *
 * Usage: bench [-n min_bytes] [-m max_bytes] [-r reps] [-o file.csv]
 */
#include "allreduce.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*allreduce_fn)(const void *, void *, int, MPI_Datatype, MPI_Comm);

static int iters_for(size_t bytes)
{
    size_t it = ((size_t)128 << 20) / bytes;
    if (it > 2000)
        it = 2000;
    if (it < 20)
        it = 20;
    return (int)it;
}

static double time_one(allreduce_fn fn, const double *send, double *recv,
                       int count, int iters, int warmup, int reps)
{
    double best = INFINITY;
    for (int w = 0; w < warmup; w++)
        fn(send, recv, count, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int r = 0; r < reps; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < iters; i++)
            fn(send, recv, count, MPI_DOUBLE, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        double local = (t1 - t0) / iters, global;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global < best)
            best = global;
    }
    return best * 1e6;
}

static double max_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static size_t parse_size(const char *s)
{
    char *end;
    double v = strtod(s, &end);
    switch (*end) {
    case 'k': case 'K': v *= 1024; break;
    case 'm': case 'M': v *= 1024 * 1024; break;
    default: break;
    }
    return (size_t)v;
}

static int native_allreduce(const void *s, void *r, int n, MPI_Datatype t, MPI_Comm c)
{
    return MPI_Allreduce(s, r, n, t, MPI_SUM, c);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t min_bytes = 8, max_bytes = 16u << 20;
    int reps = 3;
    const char *outpath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc)
            min_bytes = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc)
            max_bytes = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc)
            outpath = argv[++i];
        else {
            if (rank == 0)
                fprintf(stderr, "usage: %s [-n min_bytes] [-m max_bytes] [-r reps] [-o file.csv]\n", argv[0]);
            MPI_Finalize();
            return 2;
        }
    }
    if (min_bytes < sizeof(double))
        min_bytes = sizeof(double);
    if (reps < 1)
        reps = 1;

    FILE *out = stdout;
    if (rank == 0 && outpath) {
        out = fopen(outpath, "w");
        if (!out) {
            perror(outpath);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int max_count = (int)(max_bytes / sizeof(double));
    double *send = malloc((size_t)max_count * sizeof(double));
    double *ref = malloc((size_t)max_count * sizeof(double));
    double *res = malloc((size_t)max_count * sizeof(double));
    if (!send || !ref || !res)
        MPI_Abort(MPI_COMM_WORLD, 1);
    for (int i = 0; i < max_count; i++)
        send[i] = sin(0.001 * i + rank) + 0.1 * (rank + 1);
    memset(ref, 0, (size_t)max_count * sizeof(double));
    memset(res, 0, (size_t)max_count * sizeof(double));

    if (rank == 0) {
        fprintf(stderr, "# %d ranks, MPI_DOUBLE sum, best of %d reps\n", size, reps);
        fprintf(out, "bytes,iters,native_us,tree_us,ring_us,max_abs_err\n");
    }

    for (size_t bytes = min_bytes; bytes <= max_bytes; bytes *= 2) {
        int count = (int)(bytes / sizeof(double));
        int iters = iters_for(bytes);
        int warmup = iters / 10 > 2 ? iters / 10 : 2;

        double t_nat = time_one(native_allreduce, send, ref, count, iters, warmup, reps);
        double t_tree = time_one(allreduce_tree, send, res, count, iters, warmup, reps);
        double err = max_diff(res, ref, count);
        double t_ring = time_one(allreduce_ring, send, res, count, iters, warmup, reps);
        double e2 = max_diff(res, ref, count);
        if (e2 > err)
            err = e2;

        double gerr;
        MPI_Reduce(&err, &gerr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            fprintf(out, "%zu,%d,%.3f,%.3f,%.3f,%.3e\n", bytes, iters, t_nat, t_tree, t_ring, gerr);
            fflush(out);
        }
    }

    if (rank == 0 && out != stdout)
        fclose(out);
    free(send);
    free(ref);
    free(res);
    allreduce_free_scratch();
    MPI_Finalize();
    return 0;
}
