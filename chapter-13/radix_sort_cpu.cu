#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>

#define SIZE 100000

void init_array(int *A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = size - i;
    }
}

void radix_sort(int *A, int size, int *B) {
    // Start from LSD to MSD
    int max_val = 0, current_digit = 0, zero_bucket_index = 0, one_bucket_index = 0, *zero_bucket, *one_bucket;
    zero_bucket = (int*)malloc(size * sizeof(int));
    one_bucket = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        if (max_val < A[i])max_val = A[i];
    }
    int *current_ref = A;
    while (max_val != 0) {
        zero_bucket_index = 0, one_bucket_index = 0;
        for (int i = 0; i < size; i++){
            int bit = (current_ref[i] >> current_digit) & 1;
            if (bit == 0) {
                zero_bucket[zero_bucket_index++] = current_ref[i];
            } else {
                one_bucket[one_bucket_index++] = current_ref[i];
            }
        }
        for (int i = 0; i < zero_bucket_index; i++) {
            B[i] = zero_bucket[i];
        }
        for (int i = 0; i < one_bucket_index; i++) {
            B[zero_bucket_index + i] = one_bucket[i];
        }

        current_ref = B;
        max_val = (max_val >> 1);
        current_digit += 1;
    }

    free(zero_bucket);
    free(one_bucket);
}

void print_array(int *A, int size, int *B) {
    for (int i = 0; i < size; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");
}

bool is_sorted(int *A, int size) {
    bool cond = true;
    for (int i = 1; i < size; i++) {
        if (A[i] < A[i-1]){
            cond = false;
            break;
        }
    }
    return cond;
}


int main() {
    int *A, size = SIZE, *B;
    A = (int*)malloc(size * sizeof(int));
    B = (int*)malloc(size * sizeof(int));
    init_array(A, size);
    radix_sort(A, size, B);
    printf("is_sorted : %d\n", is_sorted(B, size));
    free(A);
    free(B);
}