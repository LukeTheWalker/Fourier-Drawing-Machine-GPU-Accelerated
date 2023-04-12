#ifndef FILTER_VECTOR_BY_MIN_H
#define FILTER_VECTOR_BY_MIN_H

#define PRINT_MIN 0

#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cu"
#include "filter_contour_by_hand.cu"

struct is_more_than_min{
    __device__ int4 operator()(int gi, int4 * contours_sizes, int min){
        return {
            contours_sizes[gi].x > min,
            contours_sizes[gi].y > min,
            contours_sizes[gi].z > min,
            contours_sizes[gi].w > min
        };
    }
};

__global__ void fill_afference_array (int * d_scanned_sizes, int * d_flags, int number_of_contours, int * d_contours_flags){
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= number_of_contours) return;

    int start = gi == 0 ? 0 : d_scanned_sizes[gi - 1];
    int end = d_scanned_sizes[gi];
    // int size = end - start;

    // cuMemsetD32((CUdeviceptr)&d_flags[start], d_contours_flags[gi], size);   
    for (int i = start; i < end; i++){
        d_flags[i] = d_contours_flags[gi];
    } 
}

void fix_contours (int * d_contours_x, int * d_contours_y, int * d_contours_sizes, int * d_contours_flags, Sizes * sizes, int ngroups, int lws){
    int * d_scanned_sizes;

    cudaError_t err = cudaMalloc((void **)&d_scanned_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);

    scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_contours_sizes, (int4*)d_scanned_sizes, NULL, round_div_up(sizes->number_of_contours, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Scanned sizes:  ");
    print_array_dev(d_scanned_sizes, number_of_contours);
    printf("\n");
    #endif

    int * d_flags;
    int contours_linear_size;

    err = cudaMemcpy(&contours_linear_size, &d_scanned_sizes[sizes->number_of_contours - 1], sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_flags, sizeof(int) * contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    fill_afference_array<<<round_div_up(sizes->number_of_contours, 256), 256>>>(d_scanned_sizes, d_flags, sizes->number_of_contours, d_contours_flags);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    int *d_positions, *d_tails;

    int ntails = ngroups > 1 ? round_mul_up(ngroups, 4) : ngroups;
    err = cudaMalloc((void **)&d_positions, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_tails, ngroups * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    scan_sliding_window<<<ngroups, lws, lws*sizeof(int)>>>((int4*)d_flags, (int4*)d_positions, d_tails, round_div_up(contours_linear_size, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    if (ngroups > 1){
        scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_tails, (int4*)d_tails, NULL, round_div_up(ntails, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }
    

    if (ngroups > 1){
        scan_fixup<<<ngroups, lws>>>((int4*)d_positions, d_tails, round_div_up(contours_linear_size, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    int * d_contours_x_out, * d_contours_y_out;
    
    err = cudaMalloc((void **)&d_contours_x_out, sizeof(int) * contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y_out, sizeof(int) * contours_linear_size); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Positions computed: ");
    print_array_dev(d_positions, contours_linear_size);
    printf("\n");
    #endif

    #if PRINT_MIN
    printf("Contour x before:   ");
    print_array_dev(d_contours_x, contours_linear_size);
    printf("\n");
    #endif

    move_contours<<<round_div_up(contours_linear_size, lws), lws>>>(d_contours_x, d_contours_y, d_contours_x_out, d_contours_y_out, d_flags, d_positions, contours_linear_size);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Contour x computed: ");
    print_array_dev(d_contours_x_out, contours_linear_size);
    printf("\n");
    #endif

    err = cudaMemcpy(d_contours_x, d_contours_x_out, sizeof(int) * contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, d_contours_y_out, sizeof(int) * contours_linear_size, cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(&(sizes->contours_linear_size), &d_scanned_sizes[sizes->number_of_contours - 1], sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_contours_x_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_y_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_positions); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_tails); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_flags); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_scanned_sizes); cuda_err_check(err, __FILE__, __LINE__);

    return;
}

void filter_vector_by_min_wrapper(int * d_contours_x, int * d_contours_y, int * h_contours_sizes, int min_size, Sizes * sizes, int ngroups = 1024, int lws = 256){
    cudaError_t err;
    int * d_contours_sizes;
    int * d_flags;

    err = cudaMalloc((void **)&d_contours_sizes, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_flags, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_sizes, h_contours_sizes, sizeof(int) * sizes->number_of_contours, cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Contour sizes:  ");
    print_array_dev(d_contours_sizes, number_of_contours);
    printf("\n");
    #endif

    int nquarts_flags = round_div_up(sizes->number_of_contours, 4);
    compute_flags<is_more_than_min><<<round_div_up(nquarts_flags, 256), 256>>>(nquarts_flags, (int4*)d_flags, (int4*)d_contours_sizes, min_size);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Flags computed: ");
    print_array_dev(d_flags, number_of_contours);
    printf("\n");
    #endif

    int *d_positions, *d_tails;

    int ntails = ngroups > 1 ? round_mul_up(ngroups, 4) : ngroups;
    err = cudaMalloc((void **)&d_positions, sizes->number_of_contours * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_tails, ngroups * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    scan_sliding_window<<<ngroups, lws, lws*sizeof(int)>>>((int4*)d_flags, (int4*)d_positions, d_tails, round_div_up(sizes->number_of_contours, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    if (ngroups > 1){
        scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_tails, (int4*)d_tails, NULL, round_div_up(ntails, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    if (ngroups > 1){
        scan_fixup<<<ngroups, lws>>>((int4*)d_positions, d_tails, round_div_up(sizes->number_of_contours, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    int * d_contours_sizes_out;

    err = cudaMalloc((void **)&d_contours_sizes_out, sizeof(int) * sizes->number_of_contours); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("Positions comp: ");
    print_array_dev(d_positions, sizes->number_of_contours);
    printf("\n");
    #endif

    move_elements<<<round_div_up(sizes->number_of_contours, lws), lws>>>(d_contours_sizes, d_contours_sizes_out, d_flags, d_positions, sizes->number_of_contours);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    // now let's fix the x and y coordinates
    fix_contours(d_contours_x, d_contours_y, d_contours_sizes, d_flags, sizes, ngroups, lws);

    err = cudaMemcpy(h_contours_sizes, d_contours_sizes_out, sizeof(int) * sizes->number_of_contours, cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(&(sizes->number_of_contours), d_positions + sizes->number_of_contours - 1, sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_MIN
    printf("contours_sizes: ");
    print_array_dev(d_contours_sizes_out, sizes->number_of_contours);
    printf("\n");
    #endif

    err = cudaFree(d_contours_sizes); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_sizes_out); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_positions); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_tails); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_flags); cuda_err_check(err, __FILE__, __LINE__);

    return;
}

void test2() {
    vector<vector<Point>> contours;

    contours.push_back(vector<Point>());
    contours[0].push_back(Point(0, 0));
    contours[0].push_back(Point(1, 1));
    contours[0].push_back(Point(2, 2));
    contours[0].push_back(Point(3, 3));
    contours[0].push_back(Point(4, 4));
    contours[0].push_back(Point(5, 5));

    contours.push_back(vector<Point>());
    contours[1].push_back(Point(10, 10));
    contours[1].push_back(Point(11, 11));
    contours[1].push_back(Point(12, 12));

    contours.push_back(vector<Point>());
    contours[2].push_back(Point(20, 20));
    contours[2].push_back(Point(21, 21));
    contours[2].push_back(Point(22, 22));
    contours[2].push_back(Point(23, 23));
    contours[2].push_back(Point(24, 24));

    contours.push_back(vector<Point>());
    contours[3].push_back(Point(30, 30));

    contours.push_back(vector<Point>());
    contours[4].push_back(Point(40, 40));
    contours[4].push_back(Point(41, 41));

    contours.push_back(vector<Point>());
    contours[5].push_back(Point(50, 50));
    contours[5].push_back(Point(51, 51));
    contours[5].push_back(Point(52, 52));
    contours[5].push_back(Point(53, 53));

    int *d_contours_x, *d_contours_y;
    int contours_linear_size = 0;

    int number_of_countours = contours.size();
    cudaError_t err;

    for (int i = 0; i < contours.size(); i++) contours_linear_size += contours[i].size();

    int * h_contours_x = (int *)malloc(contours_linear_size * sizeof(int));
    int * h_contours_y = (int *)malloc(contours_linear_size * sizeof(int));

    int * h_contours_sizes = (int *)malloc(number_of_countours * sizeof(int));

    for (int i = 0; i < number_of_countours; i++) h_contours_sizes[i] = contours[i].size();
    
    int idx = 0;
    // printf("Positions original: ");
    for (int i = 0; i < contours.size(); i++){
        for (int j = 0; j < contours[i].size(); j++){
            h_contours_x[idx] = contours[i][j].x;
            h_contours_y[idx] = contours[i][j].y;
            idx++;
        }
    }

    err = cudaMalloc((void **)&d_contours_x, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_x, h_contours_x, contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, h_contours_y, contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    cout << "Summary: " << contours_linear_size << " points in " << number_of_countours << " contours" << endl;
    for (int i = 0; i < contours.size(); i++) cout << "Contour " << i << " has " << contours[i].size() << " points" << endl;

    // filter_vector_by_min_wrapper(d_contours_x, d_contours_y, h_contours_sizes, number_of_countours, min_size);

    vector<vector<Point>> after_filter_contours;
    get_after_filter_contours(after_filter_contours, d_contours_x, d_contours_y, contours_linear_size, h_contours_sizes, number_of_countours);
    contours = after_filter_contours;

    err = cudaFree(d_contours_x); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_contours_y); cuda_err_check(err, __FILE__, __LINE__);

    free(h_contours_x);
    free(h_contours_y);
    free(h_contours_sizes);
}

#endif