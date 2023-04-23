#ifndef CLUSTER_MIN_DISTANCE_H
#define CLUSTER_MIN_DISTANCE_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cu"
#include "merge_contours.cu"

__global__ void compute_distance_matrix(int * d_contours_x, int * d_contours_y, int * d_reverse_lookup, uint32_t * d_distance_matrix, int contours_linear_size, int number_of_contours){
        uint64_t gi = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t nels = ((uint64_t)contours_linear_size * ((uint64_t)contours_linear_size - 1)) / 2;

    uint64_t point1 = (uint64_t)contours_linear_size - 2 - floor(sqrt((double)-8*gi + 4*(uint64_t)contours_linear_size*((uint64_t)contours_linear_size-1)-7)/2.0 - 0.5);
    uint64_t point2 = gi + point1 + 1 - (uint64_t)contours_linear_size*((uint64_t)contours_linear_size-1)/2 + ((uint64_t)contours_linear_size-point1)*(((uint64_t)contours_linear_size-point1)-1)/2;

    // printf("Point1: %lu, Point2: %lu\n", point1, point2);

    if (gi >= nels) return;

    if (gi >= nels || point1 == point2) return;
    
    int contour1 = d_reverse_lookup[point1];
    int contour2 = d_reverse_lookup[point2];

    if (contour1 == contour2) return;


    int x1 = d_contours_x[point1];
    int y1 = d_contours_y[point1];
    int x2 = d_contours_x[point2];
    int y2 = d_contours_y[point2];

    int distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);


    d_distance_matrix[contour1 * number_of_contours + contour2] = distance;
    d_distance_matrix[contour2 * number_of_contours + contour1] = distance;

}

void order_cluster_by_distance_wrapper(int * d_contours_x, int * d_contours_y, int * h_contours_size, Sizes * sizes, int ngroups = 1024, int lws = 256){
    int * d_scanned_sizes;
    int * d_contours_sizes;

    cudaError_t err = cudaMalloc((void **)&d_scanned_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_sizes, h_contours_size, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_contours_sizes, (int4*)d_scanned_sizes, NULL, round_div_up(sizes->number_of_contours, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    int * d_reverse_lookup;

    err = cudaMalloc((void **)&d_reverse_lookup, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    reverse_lookup_contours<<<round_div_up(sizes->number_of_contours, 256), 256>>>(d_scanned_sizes, d_reverse_lookup, sizes->number_of_contours);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    uint32_t * d_distance_matrix;

    err = cudaMalloc((void **)&d_distance_matrix, sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_distance_matrix, -1, sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__); 

    uint64_t nels = (sizes->contours_linear_size * (sizes->contours_linear_size - 1)) / 2;
    uint64_t gws = round_div_up(nels, 1024);
    compute_distance_matrix<<<gws, 1024>>>(d_contours_x, d_contours_y, d_reverse_lookup, d_distance_matrix, sizes->contours_linear_size, sizes->number_of_contours);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    int * h_distance_matrix = (int *)malloc(sizeof(int) * sizes->number_of_contours * sizes->number_of_contours);

    err = cudaMemcpy(h_distance_matrix, d_distance_matrix, sizeof(int) * sizes->number_of_contours * sizes->number_of_contours, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int * h_new_contours_relocation = (int *)malloc(sizeof(int) * sizes->number_of_contours);

    h_new_contours_relocation[0] = max_element(h_contours_size, h_contours_size + sizes->number_of_contours) - h_contours_size;

    char * h_visited = (char *)malloc(sizeof(char) * sizes->number_of_contours);
    memset(h_visited, 0, sizeof(char) * sizes->number_of_contours);

    int from = h_new_contours_relocation[0];
    int cnt = 1;
    h_visited[from] = 1;
    while (true){
        int * start_row = h_distance_matrix + (from * sizes->number_of_contours);
        int min = INT_MAX;
        int to = -1;
        for (int i = 0; i < sizes->number_of_contours; i++){
            if (h_visited[i]) continue;
            if (start_row[i] == -1) continue;
            if (start_row[i] < min){
                min = start_row[i];
                to = i;
            }
        }
        // cout << "Closest to " << from << " is " << to << endl;
        h_new_contours_relocation[cnt] = to;
        cnt++;
        h_visited[to] = 1;
        if (cnt == sizes->number_of_contours) break;
        from = to;
    }

    int * d_contours_out_x;
    int * d_contours_out_y;

    err = cudaMalloc((void **)&d_contours_out_x, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_out_y, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    int * h_scanned_sizes = (int *)malloc(sizeof(int) * sizes->number_of_contours);

    err = cudaMemcpy(h_scanned_sizes, d_scanned_sizes, sizeof(int) * sizes->number_of_contours, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int * h_new_contours_size = (int *)malloc(sizeof(int) * sizes->number_of_contours);

    for (int i = 0; i < sizes->number_of_contours; i++){
        h_new_contours_size[i] = h_contours_size[h_new_contours_relocation[i]];
    }

    int sum = 0;
    for (int i = 0; i < sizes->number_of_contours; i++){
        int contour_number = h_new_contours_relocation[i];
        int start_at = 0 ? contour_number == 0 : h_scanned_sizes[contour_number - 1];
        err = cudaMemcpy(d_contours_out_x + sum, d_contours_x + start_at, sizeof(int) * h_contours_size[contour_number], cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_contours_out_y + sum, d_contours_y + start_at, sizeof(int) * h_contours_size[contour_number], cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        sum += h_contours_size[contour_number];
    }

    memcpy(h_contours_size, h_new_contours_size, sizeof(int) * sizes->number_of_contours);

    err = cudaMemcpy(d_contours_x, d_contours_out_x, sizeof(int) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, d_contours_out_y, sizeof(int) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_contours_out_x); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_out_y); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_reverse_lookup); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_distance_matrix); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_scanned_sizes); cuda_err_check(err, __FILE__, __LINE__);

    free(h_distance_matrix);
    free(h_new_contours_relocation);
    free(h_visited);
    free(h_scanned_sizes);
    free(h_new_contours_size);

    return;
}


#endif