#ifndef MERGE_CONTOURS_H
#define MERGE_CONTOURS_H

#define PRINT_MERGE 0

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cu"

__global__ void reverse_lookup_contours (int * d_scanned_sizes, int * d_reverse_lookup, int number_of_contours){
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= number_of_contours) return;

    int start = gi == 0 ? 0 : d_scanned_sizes[gi - 1];
    int end = d_scanned_sizes[gi];

    for (int i = start; i < end; i++){
        d_reverse_lookup[i] = gi;
    } 
}

__global__ void compute_closeness_matrix (int * d_contours_x, int * d_contours_y, int * d_reverse_lookup, char * d_closeness_matrix, int contours_linear_size, int number_of_contours, int merge_distance){
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

    if (distance < merge_distance * merge_distance){
        #if PRINT_MERGE
        printf("Contour %d and %d are close thanks to points %d and %d having distance %d vs %d merge_distance\n", contour1, contour2, point1, point2, distance, merge_distance*merge_distance);
        #endif
        // printf("Accessing matrix index: %d", contour1 * number_of_contours + contour2);
        d_closeness_matrix[contour1 * number_of_contours + contour2] = 1;
        d_closeness_matrix[contour2 * number_of_contours + contour1] = 1;
    }
}

__global__ void reallign_contours (int* d_contours_x_in, int* d_contours_y_in, int * d_contours_x_out, int * d_contours_y_out, int* d_reverse_lookup, int * d_scanned_sizes, int * d_starts_at, int * d_cumulative_positions, int * d_merge_to,  int contours_linear_size){
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= contours_linear_size) return;

    int afference_contour = d_reverse_lookup[gi];
    int new_start = d_starts_at[afference_contour];
    int father_contour = d_merge_to[afference_contour] == -1 ? afference_contour : d_merge_to[afference_contour];
    int global_offset = father_contour == 0 ? 0 : d_cumulative_positions[father_contour - 1];

    int start = afference_contour == 0 ? 0 : d_scanned_sizes[afference_contour - 1];
    int idx_in_contour = gi - start;

    #if PRINT_MERGE
    if (idx_in_contour + global_offset + new_start != gi)
    printf("Point %d is at %d in the new array, has %d as father, its contour starts at %d, its idx in contour is %d and its global offset is %d\n", gi, idx_in_contour + global_offset, father_contour, new_start, idx_in_contour, global_offset);
    #endif
    d_contours_x_out[idx_in_contour + global_offset + new_start] = d_contours_x_in[gi];
    d_contours_y_out[idx_in_contour + global_offset + new_start] = d_contours_y_in[gi];
}


void merge_contours_wrapper(int * d_contours_x, int * d_contours_y, int * h_contours_size, int merge_distance, Sizes * sizes, int ngroups = 1024, int lws = 256){
    int * d_scanned_sizes;
    int * d_contours_sizes;


    cudaError_t err = cudaMalloc((void **)&d_scanned_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_sizes, h_contours_size, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MERGE
    printf("--------------------\n");
    printf("contours sizes: ");
    print_array_dev(d_contours_sizes, sizes->number_of_contours);
    printf("\n");
    #endif

    scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_contours_sizes, (int4*)d_scanned_sizes, NULL, round_div_up(sizes->number_of_contours, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MERGE
    printf("scanned sizes: ");
    print_array_dev(d_scanned_sizes, sizes->number_of_contours);
    printf("\n");
    #endif

    int * d_reverse_lookup;

    err = cudaMalloc((void **)&d_reverse_lookup, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    reverse_lookup_contours<<<round_div_up(sizes->number_of_contours, 256), 256>>>(d_scanned_sizes, d_reverse_lookup, sizes->number_of_contours);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MERGE
    printf("reverse lookup: ");
    print_array_dev(d_reverse_lookup, sizes->contours_linear_size);
    printf("\n");
    #endif

    char * d_closeness_matrix;

    err = cudaMalloc((void **)&d_closeness_matrix, sizeof(char) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_closeness_matrix, 0, sizeof(char) * sizes->number_of_contours * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__); 

    uint64_t nels = (sizes->contours_linear_size * (sizes->contours_linear_size - 1)) / 2;
    uint64_t gws = round_div_up(nels, 1024);
    compute_closeness_matrix<<<gws, 1024>>>(d_contours_x, d_contours_y, d_reverse_lookup, d_closeness_matrix, sizes->contours_linear_size, sizes->number_of_contours, merge_distance);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    int * h_merge_to  = new int[sizes->number_of_contours];
    char * h_closeness = new char[sizes->number_of_contours * sizes->number_of_contours];

    for (int i = 0; i < sizes->number_of_contours; i++) h_merge_to[i] = -1;
    
    // #if PRINT_MERGE
    // printf("closeness matrix: \n");
    // for (int i = 0; i < sizes->number_of_contours; i++){
    //     print_array_dev(d_closeness_matrix + i * sizes->number_of_contours, sizes->number_of_contours);
    //     printf("\n");
    // }
    // #endif

    err = cudaMemcpy(h_closeness, d_closeness_matrix, sizeof(char) * sizes->number_of_contours * sizes->number_of_contours, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    for (int i = 0; i < sizes->number_of_contours; i++){
        for (int j = i + 1; j < sizes->number_of_contours; j++){
            if (h_closeness[i * sizes->number_of_contours + j] == 1){
                int father = i;
                while (h_merge_to[father] != -1) father = h_merge_to[father];
                h_merge_to[j] = father;
            }
        }
    }

    #if PRINT_MERGE
    printf("merge to: ");
    for (int i = 0; i < sizes->number_of_contours; i++)
        printf("%d ", h_merge_to[i]);
    printf("\n");
    #endif

    int * starts_at = (int *) malloc(sizeof(int) * sizes->number_of_contours);
    int * cumulative_poisitions = (int *) malloc(sizeof(int) * sizes->number_of_contours);

    memset(cumulative_poisitions, 0, sizeof(int) * sizes->number_of_contours);
    memset(starts_at, 0, sizeof(int) * sizes->number_of_contours);
    for (int i = 0; i < sizes->number_of_contours; i++){
        if (h_merge_to[i] != -1) {
            starts_at[i] = cumulative_poisitions[h_merge_to[i]];
            cumulative_poisitions[h_merge_to[i]] += h_contours_size[i];
        }
        else {
            cumulative_poisitions[i] = h_contours_size[i];
        }
    }

    #if PRINT_MERGE
    printf("cumulative positions: ");
    for (int i = 0; i < sizes->number_of_contours; i++)
        printf("%d ", cumulative_poisitions[i]);
    printf("\n");
    #endif

    int tot = 0;
    for (int i = 0; i < sizes->number_of_contours; i++){
        if (h_merge_to[i] == -1){
            h_contours_size[tot] = cumulative_poisitions[i];
            tot++;
        }
    }

    // inclusibe scan cumulative positions
    for (int i = 1; i < sizes->number_of_contours; i++)
        cumulative_poisitions[i] += cumulative_poisitions[i - 1];
    
    #if PRINT_MERGE
    printf("cumulative positions scanned: ");
    for (int i = 0; i < sizes->number_of_contours; i++)
        printf("%d ", cumulative_poisitions[i]);
    printf("\n");

    printf("starts at: ");
    for (int i = 0; i < sizes->number_of_contours; i++)
        printf("%d ", starts_at[i]);
    printf("\n");
    #endif

    int * d_starts_at, *d_cumulative_poisitions, *d_merge_to;
    err = cudaMalloc((void **)&d_starts_at, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_cumulative_poisitions, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_merge_to, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_starts_at, starts_at, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_cumulative_poisitions, cumulative_poisitions, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_merge_to, h_merge_to, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    int * d_contours_x_out, * d_contours_y_out;

    err = cudaMalloc((void **)&d_contours_x_out, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y_out, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    reallign_contours<<<round_div_up(sizes->contours_linear_size, 256), 256>>>(d_contours_x, d_contours_y, d_contours_x_out, d_contours_y_out, d_reverse_lookup, d_scanned_sizes, d_starts_at, d_cumulative_poisitions, d_merge_to, sizes->contours_linear_size);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_x, d_contours_x_out, sizeof(int) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, d_contours_y_out, sizeof(int) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    sizes->number_of_contours = tot;

    free(h_merge_to);
    free(h_closeness);
    free(starts_at);
    free(cumulative_poisitions);

    err = cudaFree(d_scanned_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_reverse_lookup); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_starts_at); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_x_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_y_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_cumulative_poisitions); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_merge_to); cuda_err_check(err, __FILE__, __LINE__);
}

void test_merge (){
    int h_contours_x [] = {4,5,1,9,4,3,1,30,11};
    int h_contours_y [] = {4,5,1,9,4,3,1,30,11};

    int h_contours_sizes [] = {3, 2, 2, 1, 1};

    int * d_contours_x, * d_contours_y;
    
    cudaError_t err;

    Sizes sizes;

    sizes.contours_linear_size = 9;
    sizes.number_of_contours = 5;

    err = cudaMalloc((void **)&d_contours_x, sizes.contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, sizes.contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_x, h_contours_x, sizes.contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, h_contours_y, sizes.contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    printf("Before compute: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");

    merge_contours_wrapper(d_contours_x, d_contours_y, h_contours_sizes, 10, &sizes);

    printf("After: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");
}



#endif