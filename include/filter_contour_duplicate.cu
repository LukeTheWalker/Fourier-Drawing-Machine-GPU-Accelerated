#ifndef FILTER_CONTOUR_DUPLICATE_H
#define FILTER_CONTOUR_DUPLICATE_H

#define PRINT_DUP_FLAGS 0

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

__global__ void compute_duplicates_flags (int4 * d_contours_x, int4 * d_contours_y, int4 * d_flags, int nquarts_contours_linear_size){
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    int nels = (nquarts_contours_linear_size * (nquarts_contours_linear_size - 1)) / 2;

    int point1 = nquarts_contours_linear_size - 2 - floor(sqrt((double)-8*gi + 4*nquarts_contours_linear_size*(nquarts_contours_linear_size-1)-7)/2.0 - 0.5);
    int point2 = gi + point1 + 1 - nquarts_contours_linear_size*(nquarts_contours_linear_size-1)/2 + (nquarts_contours_linear_size-point1)*((nquarts_contours_linear_size-point1)-1)/2;

    if (gi >= nels || point1 == point2) return;

    d_flags[point2].x = d_flags[point2].x && !(d_contours_x[point1].x == d_contours_x[point2].x && d_contours_y[point1].x == d_contours_y[point2].x);
    d_flags[point2].y = d_flags[point2].y && !(d_contours_x[point1].y == d_contours_x[point2].y && d_contours_y[point1].y == d_contours_y[point2].y);
    d_flags[point2].z = d_flags[point2].z && !(d_contours_x[point1].z == d_contours_x[point2].z && d_contours_y[point1].z == d_contours_y[point2].z);
    d_flags[point2].w = d_flags[point2].w && !(d_contours_x[point1].w == d_contours_x[point2].w && d_contours_y[point1].w == d_contours_y[point2].w);
    
}

void filter_contour_duplicate_wrapper(int * d_contours_x, int * d_contours_y, int * h_contours_sizes, Sizes * sizes, int ngroups = 1024, int lws = 256){
    int *d_flags;
    cudaError_t err;
    
    err = cudaMalloc((void **)&d_flags, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    cuMemsetD32((CUdeviceptr)d_flags, 1, sizes->contours_linear_size);

    int nquarts_flags = round_div_up(sizes->contours_linear_size, 4);
    compute_duplicates_flags<<<round_div_up(nquarts_flags, 256), 256>>>((int4 *)d_contours_x, (int4 *)d_contours_y, (int4 *)d_flags, nquarts_flags);

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_DUP_FLAGS
    printf("Flags computed: ");
    print_array_dev(d_flags, sizes->contours_linear_size);
    printf("\n");
    #endif

    int * d_contours_x_out, * d_contours_y_out;

    err = cudaMalloc((void **)&d_contours_x_out, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y_out, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_DUP_FLAGS
    printf("Before filter:  ");
    print_array_dev(d_contours_x, sizes->contours_linear_size);
    printf("\n");
    #endif

    filter_contour(d_contours_x, d_contours_y, h_contours_sizes, d_contours_x_out, d_contours_y_out, d_flags, sizes, ngroups, lws);

    #if PRINT_DUP_FLAGS
    printf("After filter:   ");
    print_array_dev(d_contours_x_out, sizes->contours_linear_size);
    printf("\n");
    #endif

    err = cudaMemcpy(d_contours_x, d_contours_x_out, sizes->contours_linear_size * sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, d_contours_y_out, sizes->contours_linear_size * sizeof(int), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_contours_x_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_y_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_flags); cuda_err_check(err, __FILE__, __LINE__);

    return;
}

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

    filter_contour_duplicate_wrapper(d_contours_x, d_contours_y, h_contours_sizes, &sizes);

    printf("After: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");
}

#endif