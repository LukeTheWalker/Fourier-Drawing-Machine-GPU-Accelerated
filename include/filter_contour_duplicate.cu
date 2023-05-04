#ifndef FILTER_CONTOUR_DUPLICATE_H
#define FILTER_CONTOUR_DUPLICATE_H

#define PRINT_DUP_FLAGS 0
#define PROFILE_DUP 0

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "filter_contour.cu"
#include "streamCompaction.cu"
#include "merge_contours.cu"

__global__ void compute_duplicates_flags (point * d_contours, int * d_flags, uint64_t contours_linear_size){
    uint64_t gi = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nels = ((uint64_t)contours_linear_size * ((uint64_t)contours_linear_size - 1)) / 2;

    uint64_t point1 = (uint64_t)contours_linear_size - 2 - floor(sqrt((double)-8*gi + 4*(uint64_t)contours_linear_size*((uint64_t)contours_linear_size-1)-7)/2.0 - 0.5);
    uint64_t point2 = gi + point1 + 1 - (uint64_t)contours_linear_size*((uint64_t)contours_linear_size-1)/2 + ((uint64_t)contours_linear_size-point1)*(((uint64_t)contours_linear_size-point1)-1)/2;
    
    if (gi >= nels || point1 == point2) return;

    d_flags[point2] = d_flags[point2] && !(d_contours[point1].x == d_contours[point2].x && d_contours[point1].y == d_contours[point2].y);
    
}

void filter_contour_duplicate_wrapper(point * d_contours, int * h_contours_sizes, Sizes * sizes, int ngroups = 1024, int lws = 256){
    int *d_flags;
    cudaError_t err;

    #if PROFILE_DUP
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #endif
    
    err = cudaMalloc((void **)&d_flags, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    cuMemsetD32((CUdeviceptr)d_flags, 1, sizes->contours_linear_size);

    // uint64_t nquarts = round_div_up_64(sizes->contours_linear_size, 4);
    uint64_t nquarts = sizes->contours_linear_size;
    uint64_t nels = ((uint64_t)nquarts * ((uint64_t)nquarts - 1)) / 2;
    uint64_t gws = round_div_up_64(nels, 1024);

    #if PROFILE_DUP
    cudaEventRecord(start);
    #endif

    compute_duplicates_flags<<<gws, 1024>>>(d_contours, d_flags, nquarts);

    #if PROFILE_DUP
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for duplicate flags: %f ms\n", time);
    printf("GE/s: %f\n", (float)nels / time / 1e6);
    printf("GB/s: %f\n", 20 * nels * sizeof(int) / time / 1e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_DUP_FLAGS
    printf("Flags computed: ");
    print_array_dev(d_flags, sizes->contours_linear_size);
    printf("\n");
    #endif

    point * d_contours_out;

    err = cudaMalloc((void **)&d_contours_out, sizes->contours_linear_size * sizeof(point)); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_DUP_FLAGS
    printf("Before filter:  ");
    print_array_dev(d_contours_x, sizes->contours_linear_size);
    printf("\n");
    #endif

    filter_contour(d_contours, h_contours_sizes, d_contours_out, d_flags, sizes, ngroups, lws);

    #if PRINT_DUP_FLAGS
    printf("After filter:   ");
    print_array_dev(d_contours_x_out, sizes->contours_linear_size);
    printf("\n");
    #endif

    err = cudaMemcpy(d_contours, d_contours_out, sizes->contours_linear_size * sizeof(point), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_contours_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_flags); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILE_DUP
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return;
}

#if 0
void test_duplicate () {
    int h_contours_x [] = {4,5,1,2,4,3,1};
    int h_contours_y [] = {4,5,1,2,4,3,1};

    int h_contours_sizes [] = {3, 2, 2};

    int * d_contours_x, * d_contours_y;
    
    cudaError_t err;

    err = cudaMalloc((void **)&d_contours_x, 7 * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, 7 * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_x, h_contours_x, 7 * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, h_contours_y, 7 * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    Sizes sizes;

    sizes.contours_linear_size = 7;
    sizes.number_of_contours = 3;

    printf("Before compute: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");

    // filter_contour_duplicate_wrapper(d_contours_x, d_contours_y, h_contours_sizes, &sizes);

    printf("After: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");
}
#endif

#endif