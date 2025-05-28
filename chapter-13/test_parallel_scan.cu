#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include "../helper_methods/helpers.h"

using namespace helpers;

void init_array(int *A, int size) {
    for (int i =0; i < size; i++) {
        A[i] = i;
    }
}

int main() {
    int *A, size = 20000, *B;
    A = (int*)malloc(size * sizeof(int));
    B = (int*)malloc(size * sizeof(int));
    init_array(A, size);
    helpers::parallel_scan_cpu(A, B, size);
    for (int i = 0; i < 8; i++) {
        printf("%d ", B[i]);
    }
    printf("\n%d\n", B[size-1]);
    free(A);
    free(B);
}