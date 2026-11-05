#include "common.h"
#include "allreduce.h"

#include <stdlib.h>

static void *scratch[AR_SCRATCH_SLOTS];
static size_t scratch_cap[AR_SCRATCH_SLOTS];

int ar_elem_size(MPI_Datatype dtype, size_t *size)
{
    if (dtype == MPI_DOUBLE) {
        *size = sizeof(double);
        return MPI_SUCCESS;
    }
    if (dtype == MPI_FLOAT) {
        *size = sizeof(float);
        return MPI_SUCCESS;
    }
    return MPI_ERR_TYPE;
}

static void sum_double(double *restrict a, const double *restrict b, int n)
{
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

static void sum_float(float *restrict a, const float *restrict b, int n)
{
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

void ar_sum(void *inout, const void *in, int n, MPI_Datatype dtype)
{
    if (dtype == MPI_DOUBLE)
        sum_double(inout, in, n);
    else
        sum_float(inout, in, n);
}

void *ar_scratch(int slot, size_t bytes)
{
    if (bytes == 0)
        bytes = 1;
    if (bytes > scratch_cap[slot]) {
        free(scratch[slot]);
        scratch[slot] = malloc(bytes);
        scratch_cap[slot] = scratch[slot] ? bytes : 0;
    }
    return scratch[slot];
}

void allreduce_free_scratch(void)
{
    for (int i = 0; i < AR_SCRATCH_SLOTS; i++) {
        free(scratch[i]);
        scratch[i] = NULL;
        scratch_cap[i] = 0;
    }
}
