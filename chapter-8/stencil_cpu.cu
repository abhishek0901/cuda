#include <stdio.h>
#include <stdlib.h>
#include<cmath>

#define SCOPE 5

void init_function(float *function, int X, int Y, int Z, float h) {
    for (int z = 0; z < Z; z ++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                function[z * Y * X + y * X + x] =  (float)pow((x*h), 3) + pow((y*h), 3) + pow((z*h), 3);
            }
        }
    }
}

void first_order_derivative(float *function, int X, int Y, int Z, float h, float *f_derivative) {
    float x_derivative, y_derivative, z_derivative;
    for (int z = 1; z < Z - 1; z++) {
        for (int y = 1; y < Y - 1; y++) {
            for (int x = 1; x < X - 1; x++) {
                x_derivative = (function[z * X * Y + y * X + (x+1)] - function[z * X * Y + y * X + (x-1)])/(2*h);
                y_derivative = (function[z * X * Y + (y+1) * X + x] - function[z * X * Y + (y-1) * X + x])/(2*h);
                z_derivative = (function[(z + 1) * X * Y + y * X + x] - function[(z - 1) * X * Y + y * X + x])/(2*h);
                f_derivative[z * Y * X + y * X + x] = x_derivative + y_derivative + z_derivative;
            }
        }
    }
}

int main() {
    // Function F(x,y,z) = x^3 + y^3 + z^3
    float *function, *f_derivative;
    float h = 0.1;
    int X = SCOPE/h, Y = SCOPE/h, Z = SCOPE/h;
    function = (float*)malloc(Z * Y * X * sizeof(float));
    f_derivative = (float*)malloc(Z * Y * X * sizeof(float));

    // Init function
    init_function(function, X, Y, Z, h);

    // get function values
    // int x = 2/h, y = 1/h, z = 2/h;
    // printf("%d %d %d\n", x, y, z);
    // printf("%f\n", function[z * X * Y + y * X + x]);

    first_order_derivative(function, X, Y, Z, h, f_derivative);

    //get function derivative values
    int x = 4/h, y = 2/h, z = 3/h;
    printf("%d %d %d\n", x, y, z);
    printf("%f\n", f_derivative[z * X * Y + y * X + x]);

    // Free memory
    free(function);
    free(f_derivative);
}