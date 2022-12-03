/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void addCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* device_a = 0;
    int* device_b = 0;
    int* device_c = 0;
    cudaSetDevice(0);

    cudaMalloc((void**)&device_a, size * sizeof(int));
    cudaMalloc((void**)&device_b, size * sizeof(int));
    cudaMalloc((void**)&device_c, size * sizeof(int));

    cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    addKernel <<< 1, size >>> (device_c, device_a, device_b);
    cudaDeviceSynchronize();
    cudaMemcpy(c, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_c);
    cudaFree(device_a);
    cudaFree(device_b);
}

void run()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };


    addCuda(c, a, b, arraySize);


    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
}


*/