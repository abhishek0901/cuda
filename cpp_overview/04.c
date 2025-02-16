#include <stdio.h>
#include <stdlib.h>

int main() {
    int* ptr = NULL;
    printf("1. Initial pointer value: %p\n", (void*)ptr);

    if (ptr == NULL) {
        printf("2. ptr is NULL, can't dereference\n");
    }

    ptr = malloc(sizeof(int));

    if (ptr == NULL) {
        printf("3. Memory allocation Failed.\n");
        return 1;
    }

    // Q: how to put int value at this place
    int num = 42;
    ptr = &num;

    printf("Ptr values is %d\n", *(int*)ptr);

}