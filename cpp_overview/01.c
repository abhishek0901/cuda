#include <stdio.h> // Standard input/output for printf

// & "address of" operator
// * "dereference" operator

int main() {
    int x = 10;
    int* ptr = &x;
    printf("Address of x : %p\n", ptr);
    printf("Value of x: %d\n", *ptr);
}