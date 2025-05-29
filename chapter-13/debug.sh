nvcc -G -g merge_sort.cu ../helper_methods/merge.cu -o merge_sort --relocatable-device-code=true
cuda-gdb ./merge_sort