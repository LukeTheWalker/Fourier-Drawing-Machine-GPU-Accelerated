#ifndef FILTER_CONTOUR_BY_HAND_WRAPPER_H
#define FILTER_CONTOUR_BY_HAND_WRAPPER_H

#define PRINT_FLAGS 0
#define PROFILING_HAND 0
#define MANUAL_COUNTER 0

#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "filter_contour.cu"
#include "streamCompaction.cu"

using namespace std;

struct check_array_membership {
    #if not MANUAL_COUNTER
    __device__ int4 operator()(int gi, int4 * dat_x_arr, int4 * dat_y_arr, int4 * arr_x, int4 * arr_y, int n_quart_array) {
    #else
    __device__ int4 operator()(int gi, int4 * dat_x_arr, int4 * dat_y_arr, int4 * arr_x, int4 * arr_y, int n_quart_array, unsigned long long int * cnt) {
    #endif
        int4 res = {0, 0, 0, 0};
        int4 dat_x = dat_x_arr[gi];
        int4 dat_y = dat_y_arr[gi];
        for (int i = 0; i < n_quart_array; i++){
            #if MANUAL_COUNTER
            atomicAdd(cnt, 1);
            #endif
            int4 arr_x_i = arr_x[i];
            int4 arr_y_i = arr_y[i];
            res.x = res.x || (dat_x.x == arr_x_i.x && dat_y.x == arr_y_i.x) || (dat_x.x == arr_x_i.y && dat_y.x == arr_y_i.y) || (dat_x.x == arr_x_i.z && dat_y.x == arr_y_i.z) || (dat_x.x == arr_x_i.w && dat_y.x == arr_y_i.w);
            res.y = res.y || (dat_x.y == arr_x_i.x && dat_y.y == arr_y_i.x) || (dat_x.y == arr_x_i.y && dat_y.y == arr_y_i.y) || (dat_x.y == arr_x_i.z && dat_y.y == arr_y_i.z) || (dat_x.y == arr_x_i.w && dat_y.y == arr_y_i.w);
            res.z = res.z || (dat_x.z == arr_x_i.x && dat_y.z == arr_y_i.x) || (dat_x.z == arr_x_i.y && dat_y.z == arr_y_i.y) || (dat_x.z == arr_x_i.z && dat_y.z == arr_y_i.z) || (dat_x.z == arr_x_i.w && dat_y.z == arr_y_i.w);
            res.w = res.w || (dat_x.w == arr_x_i.x && dat_y.w == arr_y_i.x) || (dat_x.w == arr_x_i.y && dat_y.w == arr_y_i.y) || (dat_x.w == arr_x_i.z && dat_y.w == arr_y_i.z) || (dat_x.w == arr_x_i.w && dat_y.w == arr_y_i.w);
        }
        return {!res.x, !res.y, !res.z, !res.w};
    }
};

void load_contours_to_device (int * d_contours_x, int * d_contours_y, vector<vector<Point>> &contours, Sizes * sizes) {
    int *h_contours_x, *h_contours_y;
    cudaError_t err;

    h_contours_x = (int *)malloc(sizes->contours_linear_size * sizeof(int));
    h_contours_y = (int *)malloc(sizes->contours_linear_size * sizeof(int));

    int idx = 0;
    for (int i = 0; i < sizes->number_of_contours; i++){
        for (int j = 0; j < contours[i].size(); j++){
            h_contours_x[idx] = contours[i][j].x;
            h_contours_y[idx] = contours[i][j].y;
            idx++;
        }
    }
    
    err = cudaMemcpy(d_contours_x, h_contours_x, sizes->contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, h_contours_y, sizes->contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    
    free(h_contours_x);
    free(h_contours_y);   
}

void filter_contour_by_hand_wrapper(int * d_contours_x_out, int * d_contours_y_out, int * h_contours_sizes_out, vector<vector<Point>> &contours, unordered_set<Point, HashFunction> &_excluded_points, Sizes * sizes, int ngroups = 1024, int lws = 256){
    int *d_excluded_points_x, *d_excluded_points_y, *d_contours_x, *d_contours_y;
    int *h_excluded_points_x, *h_excluded_points_y;
    int *d_flags;
    int excluded_points_size = _excluded_points.size();
    cudaError_t err;

    // allocate memory on the host for all the points
    h_excluded_points_x = (int *)malloc(excluded_points_size * sizeof(int));
    h_excluded_points_y = (int *)malloc(excluded_points_size * sizeof(int));
    
    err = cudaMalloc((void **)&d_contours_x, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    load_contours_to_device(d_contours_x, d_contours_y, contours, sizes);

    int idx = 0;
    for (auto it = _excluded_points.begin(); it != _excluded_points.end(); it++){
        h_excluded_points_x[idx] = it->x;
        h_excluded_points_y[idx] = it->y;
        idx++;
    }

    err = cudaMalloc((void **)&d_excluded_points_x, excluded_points_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_excluded_points_y, excluded_points_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_flags, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_excluded_points_x, h_excluded_points_x, excluded_points_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_excluded_points_y, h_excluded_points_y, excluded_points_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    int nquarts_flags = round_div_up(sizes->contours_linear_size, 4);
    int nquarts_excluded_points = round_div_up(excluded_points_size, 4);

    #if MANUAL_COUNTER
    unsigned long long int * d_cnt;
    err = cudaMalloc((void **)&d_cnt, sizeof(unsigned long long int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_cnt, 0, sizeof(unsigned long long int)); cuda_err_check(err, __FILE__, __LINE__);
    #endif

    #if PROFILING_HAND
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    #endif

    compute_flags<check_array_membership><<<round_div_up(nquarts_flags, 256), 256>>>(nquarts_flags, (int4*)d_flags, (int4*)d_contours_x, (int4*)d_contours_y, (int4*)d_excluded_points_x, (int4*)d_excluded_points_y, nquarts_excluded_points);

    #if PROFILING_HAND
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    #if MANUAL_COUNTER
    unsigned long long int cnt;
    err = cudaMemcpy(&cnt, d_cnt, sizeof(unsigned long long int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    cout << "cnt: " << cnt << " nquarts_flags: " << nquarts_flags  << " nquarts_excluded_points: " << nquarts_excluded_points << " moltiplication: " << (uint64_t)nquarts_flags * (uint64_t)nquarts_excluded_points << endl;
    #endif
    float milliseconds = 0;
    double wrapper_time = (float)sizes->contours_linear_size * sizeof(int) * 2;
    double nested_loop_time = (uint64_t)nquarts_flags * (uint64_t)nquarts_excluded_points * 2 * sizeof(int4) / 8;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("compute_flags hand time: %f\n", milliseconds);
    printf("GE/s: %f\n", (float)sizes->contours_linear_size / milliseconds / 1e6);
    printf("GB/s: %f\n", (wrapper_time + nested_loop_time) / milliseconds / 1e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_FLAGS
    printf("Flags computed: ");
    print_array_dev(d_flags, sizes->contours_linear_size);
    printf("\n");
    #endif

    filter_contour(d_contours_x, d_contours_y, h_contours_sizes_out, d_contours_x_out, d_contours_y_out, d_flags, sizes, ngroups, lws);

    free(h_excluded_points_x);
    free(h_excluded_points_y);

    cudaFree(d_contours_x);
    cudaFree(d_contours_y);
    cudaFree(d_excluded_points_x);
    cudaFree(d_excluded_points_y);
    cudaFree(d_flags);

    #if PROFILING_HAND
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return;
}

int test() {
    vector<vector<Point>> contours;
    unordered_set<Point, HashFunction> excluded_points;

    ifstream in("debug.txt");
    int n;
    in >> n;
    vector<int> contour_sizes;

    for (int i = 0; i < n; i++){
        int m;
        in >> m;
        contour_sizes.push_back(m);
    }

    for (int i = 0; i < n; i++){
        vector<Point> contour;
        for (int j = 0; j < contour_sizes[i]; j++){
            int x, y;
            in >> x >> y;
            contour.push_back(Point(x, y));
        }
        contours.push_back(contour);
    }

    int m;
    in >> m;
    for (int i = 0; i < m; i++){
        int x, y;
        in >> x >> y;
        excluded_points.insert(Point(x, y));
    }

    cerr << "MISSING IMPORTANT DATA" << endl;

    int *d_contours_x, *d_contours_y;
    int *h_contours_sizes;
    cudaError_t err;
    Sizes * sizes;

    sizes = (Sizes*)malloc(sizeof(Sizes));
    sizes->number_of_contours = contours.size();
    sizes->contours_linear_size = 0;

    for (int i = 0; i < contours.size(); i++) sizes->contours_linear_size += contours[i].size();

    err = cudaMalloc((void **)&d_contours_x, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    h_contours_sizes = (int*)malloc(sizes->number_of_contours * sizeof(int));

    for (int i = 0; i < sizes->number_of_contours; i++) h_contours_sizes[i] = contours[i].size();

    filter_contour_by_hand_wrapper(d_contours_x, d_contours_y, h_contours_sizes, contours, excluded_points, sizes);

    for (int i = 0; i < contours.size(); i++){
        for (int j = 0; j < contours[i].size(); j++){
            printf("(%d, %d) ", contours[i][j].x, contours[i][j].y);
        }
        printf("\n");
    }
    return 0;
}
#endif