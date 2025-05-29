nvcc -o res merge_sort.cu ../helper_methods/merge.cu --relocatable-device-code=true
./res