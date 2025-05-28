#ifndef HELPERS_H
#define HELPERS_H

namespace helpers {
    // Function declaration
    void parallel_scan_cpu(int *input, int *output, int size);
    void parallel_scan_gpu(int *input_d, int *output_d, int size);
    void inclusive_to_exclusive_scan(int *input_d, int *output_d, int size, bool keep_last_elem);
}

#endif