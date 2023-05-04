#ifndef FILTER_CONTOUR_H
#define FILTER_CONTOUR_H

#define PRINT_CONTOUR 0
#define PROFILING_CONTOUR 0

#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cu"

__global__ void move_contours (int *d_contours_x, int *d_contours_y, int *dest_x, int *dest_y, int *d_flags, int *d_positions, int nels){
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    if (gi >= nels) return;
    if (!d_flags[gi]) return;
    if (gi == 0) { dest_x[0] = d_contours_x[0]; dest_y[0] = d_contours_y[0]; return; }
    
    int pos = d_positions[gi - 1];

    dest_x[pos] = d_contours_x[gi];
    dest_y[pos] = d_contours_y[gi];
}

void filter_contour (int * d_contours_x, int * d_contours_y, int * h_contours_sizes,  int * d_contours_x_out, int * d_contours_y_out, int * d_flags, Sizes * sizes, int ngroups, int lws){
    cudaError_t err;
    int *d_positions, *d_tails;

    int ntails = ngroups > 1 ? round_mul_up(ngroups, 4) : ngroups;
    err = cudaMalloc((void **)&d_positions, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_tails, ngroups * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    #if PROFILING_CONTOUR
    cudaEvent_t start, stop;
    float time;
    err = cudaEventCreate(&start); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventCreate(&stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventRecord(start, 0); cuda_err_check(err, __FILE__, __LINE__);
    #endif

    scan_sliding_window<<<ngroups, lws, lws*sizeof(int)>>>((int4*)d_flags, (int4*)d_positions, d_tails, round_div_up(sizes->contours_linear_size, 4), 32);
    #if PROFILING_CONTOUR
    err = cudaEventRecord(stop, 0); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventElapsedTime(&time, start, stop); cuda_err_check(err, __FILE__, __LINE__);
    printf("scan_sliding_window[partial] time: %f\n", time);
    printf("GE/s = %f\n", (float)sizes->contours_linear_size / time / 1e6);
    printf("GB/s = %f\n", (2 * (float)sizes->contours_linear_size * sizeof(int) + (ngroups > 1 ? (float)ngroups * sizeof(int) : 0)) / time / 1.e6);
    #endif
    
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_CONTOUR
    printf("Positions computed: ");
    print_array_dev(d_positions, sizes->contours_linear_size);
    printf("\n");
    #endif


    
    if (ngroups > 1){
        #if PROFILING_CONTOUR
        err = cudaEventRecord(start, 0); cuda_err_check(err, __FILE__, __LINE__);
        #endif

        scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_tails, (int4*)d_tails, NULL, round_div_up(ntails, 4), 32);
        #if PROFILING_CONTOUR
        err = cudaEventRecord(stop, 0); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventElapsedTime(&time, start, stop); cuda_err_check(err, __FILE__, __LINE__);
        printf("scan_sliding_window time: %f\n", time);
        printf("GE/s = %f\n", (float)ntails / time / 1e6);
        printf("GB/s = %f\n", (float)ntails * sizeof(int) / time / 1.e6);
        #endif

        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    #if PRINT_CONTOUR
    printf("Tails computed: ");
    print_array_dev(d_tails, ngroups);
    printf("\n");
    #endif

    if (ngroups > 1){
        #if PROFILING_CONTOUR
        err = cudaEventRecord(start, 0); cuda_err_check(err, __FILE__, __LINE__);
        #endif

        scan_fixup<<<ngroups, lws>>>((int4*)d_positions, d_tails, round_div_up(sizes->contours_linear_size, 4), 32);

        #if PROFILING_CONTOUR
        err = cudaEventRecord(stop, 0); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaEventElapsedTime(&time, start, stop); cuda_err_check(err, __FILE__, __LINE__);
        printf("scan_fixup time: %f\n", time);
        printf("GE/s = %f\n", (float)sizes->contours_linear_size / time / 1e6);
        printf("GB/s = %f\n", (2*(float)(sizes->contours_linear_size - lws) + ngroups) * sizeof(int) / time / 1.e6);
        #endif

        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    #if PRINT_CONTOUR
    printf("Positions computed: ");
    print_array_dev(d_positions, sizes->contours_linear_size);
    printf("\n");
    #endif

    #if PROFILING_CONTOUR
    err = cudaEventRecord(start, 0); cuda_err_check(err, __FILE__, __LINE__);
    #endif

    move_contours<<<round_div_up(sizes->contours_linear_size, lws), lws>>>(d_contours_x, d_contours_y, d_contours_x_out, d_contours_y_out, d_flags, d_positions, sizes->contours_linear_size);

    #if PROFILING_CONTOUR
    err = cudaEventRecord(stop, 0); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventSynchronize(stop); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaEventElapsedTime(&time, start, stop); cuda_err_check(err, __FILE__, __LINE__);
    printf("move_contours time: %f\n", time);
    printf("GE/s = %f\n", (float)sizes->contours_linear_size / time / 1e6);
    printf("GB/s = %f\n", 6 * (float)sizes->contours_linear_size * sizeof(int) / time / 1.e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_CONTOUR
    printf("Contour x computed: ");
    print_array_dev(d_contours_x_out, sizes->contours_linear_size);
    printf("\n");
    #endif

    int *h_positions;

    err = cudaMallocHost(&h_positions, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_positions, d_positions, sizes->contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    uint32_t cnt = 0;
    uint32_t cnt_n_contours = 0;
    for (int i = 0; i < sizes->number_of_contours; i++){
        cnt += h_contours_sizes[i];
        int tmp_contour_size = h_positions[cnt-1] - (i == 0 ? 0 : h_positions[cnt - h_contours_sizes[i] - 1] );
        
        if (tmp_contour_size > 0){
            h_contours_sizes[cnt_n_contours] = tmp_contour_size;
            cnt_n_contours++;
        }
    }

    sizes->contours_linear_size = h_positions[sizes->contours_linear_size - 1];
    sizes->number_of_contours = cnt_n_contours;

    err = cudaFree(d_positions); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_tails); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFreeHost(h_positions); cuda_err_check(err, __FILE__, __LINE__);
}

#endif