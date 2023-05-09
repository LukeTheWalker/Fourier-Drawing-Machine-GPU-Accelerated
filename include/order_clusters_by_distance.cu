#ifndef CLUSTER_MIN_DISTANCE_H
#define CLUSTER_MIN_DISTANCE_H

#define PROFILE_ORDER_CLUSTER_BY_DISTANCE 0
#define PRINT_ORDER_CLUSTER_BY_DISTANCE 0
#define KERNEL_SIZE_ORDER 32

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

__global__ void compute_distance_matrix(point * d_contours, int * d_reverse_lookup, uint32_t * d_distance_matrix, int nquarts_contours_linear_size, int contours_linear_size, int number_of_contours){
    uint64_t gi = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t n_comparison = ((uint64_t)nquarts_contours_linear_size * ((uint64_t)nquarts_contours_linear_size - 1)) / 2;

    uint64_t quarts_1 = (uint64_t)nquarts_contours_linear_size - 2 - floor(sqrt((double)-8*gi + 4*(uint64_t)nquarts_contours_linear_size*((uint64_t)nquarts_contours_linear_size-1)-7)/2.0 - 0.5);
    uint64_t quarts_2 = gi + quarts_1 + 1 - (uint64_t)nquarts_contours_linear_size*((uint64_t)nquarts_contours_linear_size-1)/2 + ((uint64_t)nquarts_contours_linear_size-quarts_1)*(((uint64_t)nquarts_contours_linear_size-quarts_1)-1)/2;

    // printf("Point1: %lu, Point2: %lu\n", point1, point2);

    if (gi >= n_comparison || quarts_1 == quarts_2) return;

    point before [KERNEL_SIZE_ORDER];
    int contour_before [KERNEL_SIZE_ORDER];
    for (int i = 0; i < KERNEL_SIZE_ORDER; i++) {
        if (quarts_1 * KERNEL_SIZE_MERGE + i >= contours_linear_size) { contour_before[i] = number_of_contours; continue; }
        before[i] = d_contours[quarts_1 * KERNEL_SIZE_ORDER + i]; 
        contour_before[i] = d_reverse_lookup[quarts_1 * KERNEL_SIZE_ORDER + i];
    }

    point after [KERNEL_SIZE_ORDER];
    int contour_after [KERNEL_SIZE_ORDER];
    for (int i = 0; i < KERNEL_SIZE_ORDER; i++) {
        if (quarts_2 * KERNEL_SIZE_MERGE + i >= contours_linear_size) { contour_after[i] = number_of_contours; continue; }
        after[i] = d_contours[quarts_2 * KERNEL_SIZE_ORDER + i]; 
        contour_after[i] = d_reverse_lookup[quarts_2 * KERNEL_SIZE_ORDER + i];
    }

     for (int i = 0; i < KERNEL_SIZE_ORDER; i++){
        for (int j = 0; j < KERNEL_SIZE_ORDER; j++) {
            if (!(contour_before[i] == contour_after[j] || contour_before[i] >= number_of_contours || contour_after[j] >= number_of_contours)){
                uint32_t distance = (before[i].x - after[j].x) * (before[i].x - after[j].x) + (before[i].y - after[j].y) * (before[i].y - after[j].y);
                if ((int)d_distance_matrix[contour_before[i] * number_of_contours + contour_after[j]] == -1 || d_distance_matrix[contour_before[i] * number_of_contours + contour_after[j]] > distance){
                    d_distance_matrix[contour_before[i] * number_of_contours + contour_after[j]] = distance;
                    d_distance_matrix[contour_after[j] * number_of_contours + contour_before[i]] = distance;
                }
            }
        }
    }

    // int distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    // int distance = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);

    // if ((int)d_distance_matrix[contour1 * number_of_contours + contour2] == -1 || d_distance_matrix[contour1 * number_of_contours + contour2] > distance){
    //     d_distance_matrix[contour1 * number_of_contours + contour2] = distance;
    //     d_distance_matrix[contour2 * number_of_contours + contour1] = distance;
    // }
}

void order_cluster_by_distance_wrapper(point * d_contours, int * h_contours_size, Sizes * sizes, int lws = 256){
    int * d_scanned_sizes;
    int * d_contours_sizes;

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #endif

    cudaError_t err = cudaMalloc((void **)&d_scanned_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_sizes, h_contours_size, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(start);
    #endif

    scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_contours_sizes, (int4*)d_scanned_sizes, NULL, round_div_up(sizes->number_of_contours, 4), 32);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Scan time: %f\n", time);
    printf("GE/s = %f\n", (float)sizes->number_of_contours / time / 1.e6);
    printf("GB/s = %f\n", (2 * (float)sizes->contours_linear_size * sizeof(int)) / time / 1.e6);
    #endif
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    int * d_reverse_lookup;

    err = cudaMalloc((void **)&d_reverse_lookup, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(start);
    #endif

    reverse_lookup_contours<<<round_div_up(sizes->number_of_contours, 256), 256>>>(d_scanned_sizes, d_reverse_lookup, sizes->number_of_contours);
    
    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Reverse lookup time: %f\n", time);
    printf("GE/s: %f\n", (float)sizes->number_of_contours / time / 1e06);
    printf("GB/s = %f\n", (2 * (float)sizes->number_of_contours * sizeof(int) + (float)sizes->contours_linear_size * sizeof(int)) / time / 1.e6);
    #endif

    
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    uint32_t * d_distance_matrix;

    err = cudaMalloc((void **)&d_distance_matrix, sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_distance_matrix, -1, sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__); 

    uint64_t nquarts = round_div_up_64((uint64_t)sizes->contours_linear_size, KERNEL_SIZE_ORDER) - 1;
    uint64_t nels = (nquarts * (nquarts - 1)) / 2;
    uint64_t distance_lws = 256;
    uint64_t gws = round_div_up_64(nels, distance_lws);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(start);
    #endif

    compute_distance_matrix<<<gws, distance_lws>>>(d_contours, d_reverse_lookup, d_distance_matrix, nquarts, sizes->contours_linear_size, sizes->number_of_contours);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    int write_accesses = nels * KERNEL_SIZE_ORDER * KERNEL_SIZE_ORDER * 2;
    uint64_t read_accesses = nels * 2 * KERNEL_SIZE_ORDER;
    printf("Distance matrix time: %f\n", time);
    printf("GE/s: %f\n", (float)nels / time / 1e06);
    printf("GB/s = %f\n", (write_accesses * sizeof(int) + read_accesses * sizeof(point))/ time / 1.e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    uint32_t * h_distance_matrix = (uint32_t *)malloc(sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours);

    err = cudaMemcpy(h_distance_matrix, d_distance_matrix, sizeof(uint32_t) * sizes->number_of_contours * sizes->number_of_contours, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int * h_new_contours_relocation = (int *)malloc(sizeof(int) * sizes->number_of_contours);

    h_new_contours_relocation[0] = max_element(h_contours_size, h_contours_size + sizes->number_of_contours) - h_contours_size;

    char * h_visited = (char *)malloc(sizeof(char) * sizes->number_of_contours);
    memset(h_visited, 0, sizeof(char) * sizes->number_of_contours);

    int from = h_new_contours_relocation[0];
    int cnt = 1;
    h_visited[from] = 1;
    while (true){
        uint32_t * start_row = h_distance_matrix + (from * sizes->number_of_contours);
        uint32_t min = INT_MAX;
        uint32_t to = (uint32_t)-1;
        for (int i = 0; i < sizes->number_of_contours; i++){
            if (h_visited[i]) continue;
            if (start_row[i] == 294967295) continue;
            if (start_row[i] < min || to == (uint32_t)-1){
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

    point * d_contours_out;

    err = cudaMalloc((void **)&d_contours_out, sizeof(point) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

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
        err = cudaMemcpy(d_contours_out + sum, d_contours + start_at, sizeof(point) * h_contours_size[contour_number], cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        sum += h_contours_size[contour_number];
    }

    memcpy(h_contours_size, h_new_contours_size, sizeof(int) * sizes->number_of_contours);

    err = cudaMemcpy(d_contours, d_contours_out, sizeof(point) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_contours_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_reverse_lookup); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_distance_matrix); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_scanned_sizes); cuda_err_check(err, __FILE__, __LINE__);

    free(h_distance_matrix);
    free(h_new_contours_relocation);
    free(h_visited);
    free(h_scanned_sizes);
    free(h_new_contours_size);

    #if PROFILE_ORDER_CLUSTER_BY_DISTANCE
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return;
}


#endif