#ifndef MERGE_CONTOURS_H
#define MERGE_CONTOURS_H

#define PRINT_MERGE 0
#define PROFILING_MERGE 0
#define KERNEL_SIZE_MERGE 8

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cu"

uint64_t round_div_up_64 (uint64_t a, uint64_t b){
    return (a + b - 1)/b;
}

__global__ void reverse_lookup_contours (int * d_scanned_sizes, int * d_reverse_lookup, int number_of_contours){
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= number_of_contours) return;

    int start = gi == 0 ? 0 : d_scanned_sizes[gi - 1];
    int end = d_scanned_sizes[gi];

    if (start - end <= 16){
        for (int i = start; i < end; i++){
            d_reverse_lookup[i] = gi;
        } 
        return;
    }

    for (int i = start; i < ((start / 4) + 1) * 4; i++){
        d_reverse_lookup[i] = gi;
    }

    for (int i = ((start / 4) + 1) * 4; i < end - (end & 0x3); i++){
        ((int4 *)d_reverse_lookup)[i] = {gi,gi,gi,gi};
    } 

    for (int i = end - (end & 0x3); i < end; i++){
        d_reverse_lookup[i] = gi;
    }
}

__global__ void compute_closeness_matrix (point * d_contours, int * d_reverse_lookup, char * d_closeness_matrix, int nquarts_contours_linear_size, int number_of_contours, int merge_distance){
    uint64_t gi = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t n_comparison = ((uint64_t)nquarts_contours_linear_size * ((uint64_t)nquarts_contours_linear_size - 1)) / 2;

    uint64_t quarts_1 = (uint64_t)nquarts_contours_linear_size - 2 - floor(sqrt((double)-8*gi + 4*(uint64_t)nquarts_contours_linear_size*((uint64_t)nquarts_contours_linear_size-1)-7)/2.0 - 0.5);
    uint64_t quarts_2 = gi + quarts_1 + 1 - (uint64_t)nquarts_contours_linear_size*((uint64_t)nquarts_contours_linear_size-1)/2 + ((uint64_t)nquarts_contours_linear_size-quarts_1)*(((uint64_t)nquarts_contours_linear_size-quarts_1)-1)/2;

    // printf("Point1: %lu, Point2: %lu\n", point1, point2);

    if (gi >= n_comparison || quarts_1 == quarts_2) return;

    point before [KERNEL_SIZE_MERGE];
    int contour_before [KERNEL_SIZE_MERGE];
    for (int i = 0; i < KERNEL_SIZE_MERGE; i++) {before[i] = d_contours[quarts_1 * KERNEL_SIZE_MERGE + i]; contour_before[i] = d_reverse_lookup[quarts_1 * KERNEL_SIZE_MERGE + i];}

    point after [KERNEL_SIZE_MERGE];
    int contour_after [KERNEL_SIZE_MERGE];
    for (int i = 0; i < KERNEL_SIZE_MERGE; i++) {after[i] = d_contours[quarts_2 * KERNEL_SIZE_MERGE + i]; contour_after[i] = d_reverse_lookup[quarts_2 * KERNEL_SIZE_MERGE + i];}

    for (int i = 0; i < KERNEL_SIZE_MERGE; i++){
        for (int j = 0; j < KERNEL_SIZE_MERGE; j++) {
            if (!(contour_before[i] == contour_after[j] || contour_before[i] >= number_of_contours || contour_after[j] >= number_of_contours || d_closeness_matrix[contour_after[j] * number_of_contours + contour_before[i]])){
                int distance = (before[i].x - after[j].x) * (before[i].x - after[j].x) + (before[i].y - after[j].y) * (before[i].y - after[j].y);
                if (distance <= merge_distance * merge_distance){
                    d_closeness_matrix[contour_before[i] * number_of_contours + contour_after[j]] = 1;
                }
            }
        }
    }
    
    // if (contour1 == contour2 || d_closeness_matrix[contour1 * number_of_contours + contour2]) return;

    // int distance = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
    // int distance = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);

    // if (distance < merge_distance * merge_distance){
    //     #if PRINT_MERGE
    //     printf("Contour %d and %d are close thanks to points %d and %d having distance %d vs %d merge_distance\n", contour1, contour2, point1, point2, distance, merge_distance*merge_distance);
    //     #endif
    //     // printf("Accessing matrix index: %d", contour1 * number_of_contours + contour2);
    //     d_closeness_matrix[contour1 * number_of_contours + contour2] = 1;
    //     // d_closeness_matrix[contour2 * number_of_contours + contour1] = 1;
    // }
}

__global__ void reallign_contours (point * d_contours_in, point * d_contours_out, int* d_reverse_lookup, int * d_scanned_sizes, int * d_starts_at, int * d_cumulative_positions, int * d_merge_to,  int contours_linear_size){
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
    d_contours_out[idx_in_contour + global_offset + new_start] = d_contours_in[gi];
}


void merge_contours_wrapper(point * d_contours, int * h_contours_size, int merge_distance, Sizes * sizes, int lws = 256){
    int * d_scanned_sizes;
    int * d_contours_sizes;
    cudaError_t err;

    #if PROFILING_MERGE
    cudaEvent_t start, stop;
    float time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #endif

    err = cudaMalloc((void **)&d_scanned_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_sizes, h_contours_size, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MERGE
    printf("--------------------\n");
    printf("contours sizes: ");
    print_array_dev(d_contours_sizes, sizes->number_of_contours);
    printf("\n");
    #endif

    #if PROFILING_MERGE
    cudaEventRecord(start);
    #endif

    scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_contours_sizes, (int4*)d_scanned_sizes, NULL, round_div_up(sizes->number_of_contours, 4), 32);
    
    #if PROFILING_MERGE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Scan sliding window merge time: %f\n", time);
    printf("GE/s: %f\n", (float)sizes->number_of_contours / time / 1e06);
    printf("GB/s = %f\n", (2 * (float)sizes->contours_linear_size * sizeof(int)) / time / 1.e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MERGE
    printf("scanned sizes: ");
    print_array_dev(d_scanned_sizes, sizes->number_of_contours);
    printf("\n");
    #endif

    int * d_reverse_lookup;

    err = cudaMalloc((void **)&d_reverse_lookup, sizeof(int) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILING_MERGE
    cudaEventRecord(start);
    #endif

    reverse_lookup_contours<<<round_div_up(sizes->number_of_contours, 256), 256>>>(d_scanned_sizes, d_reverse_lookup, sizes->number_of_contours);

    #if PROFILING_MERGE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Reverse lookup merge time: %f\n", time);
    printf("GE/s: %f\n", (float)sizes->number_of_contours / time / 1e06);
    printf("GB/s = %f\n", (2 * (float)sizes->number_of_contours * sizeof(int) + 2 * (float)sizes->contours_linear_size * sizeof(int)) / time / 1.e6);
    #endif

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

    uint64_t nquarts = round_div_up_64((uint64_t)sizes->contours_linear_size, KERNEL_SIZE_MERGE);
    uint64_t nels = (nquarts * (nquarts - 1)) / 2;
    uint64_t closeness_lws = 256;
    uint64_t gws = round_div_up_64(nels, closeness_lws);

    #if PROFILING_MERGE
    cudaEventRecord(start);
    #endif

    compute_closeness_matrix<<<gws, closeness_lws>>>(d_contours, d_reverse_lookup, d_closeness_matrix, nquarts, sizes->number_of_contours, merge_distance);

    #if PROFILING_MERGE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    int write_accesses = sizes->number_of_contours * (sizes->number_of_contours - 1) / 2; 
    uint64_t read_accesses = nels * 2 * KERNEL_SIZE_MERGE;
    printf("Compute closeness matrix merge time: %f\n", time);
    printf("GE/s: %f\n", (float)nels / time / 1e06);
    printf("GB/s = %f\n", (write_accesses * sizeof(char) + read_accesses * sizeof(point))/ time / 1.e6);
    #endif

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

    point * d_contours_out;

    err = cudaMalloc((void **)&d_contours_out, sizeof(point) * sizes->contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILING_MERGE
    cudaEventRecord(start);
    #endif

    reallign_contours<<<round_div_up(sizes->contours_linear_size, 256), 256>>>(d_contours, d_contours_out, d_reverse_lookup, d_scanned_sizes, d_starts_at, d_cumulative_poisitions, d_merge_to, sizes->contours_linear_size);
    
    #if PROFILING_MERGE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("reallign contours time: %f\n", time);
    printf("GE/s: %f\n", (float)sizes->contours_linear_size / time / 1e6);
    printf("GB/s: %f\n", ( 9 * (float)sizes->contours_linear_size * sizeof(int)) / time / 1e6);
    #endif


    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours, d_contours_out, sizeof(point) * sizes->contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    sizes->number_of_contours = tot;

    free(h_merge_to);
    free(h_closeness);
    free(starts_at);
    free(cumulative_poisitions);

    err = cudaFree(d_scanned_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_reverse_lookup); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_starts_at); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_cumulative_poisitions); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_merge_to); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILING_MERGE
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
}

#if 0
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

    // merge_contours_wrapper(d_contours_x, d_contours_y, h_contours_sizes, 10, &sizes);

    printf("After: ");
    print_array_dev(d_contours_x, sizes.contours_linear_size);
    printf("\n");
}

#endif

#endif