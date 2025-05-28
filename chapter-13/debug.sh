nvcc -G -g radix_sort_shared_memory.cu ../helper_methods/parallel_scan.cu -o radix_sort_shared_memory
cuda-gdb ./radix_sort_shared_memory