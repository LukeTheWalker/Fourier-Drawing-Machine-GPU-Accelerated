#ifndef FILTER_CONTOUR_BY_HAND_WRAPPER_H
#define FILTER_CONTOUR_BY_HAND_WRAPPER_H

#define PRINT_FLAGS 0
#define PROFILING_HAND 0

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
    __device__ int4 operator()(int gi, point * dat_arr, point * arr, int array_size) {
        int4 res = {0, 0, 0, 0};
        point dat_1 = dat_arr[gi * 4];
        point dat_2 = dat_arr[gi * 4 + 1];
        point dat_3 = dat_arr[gi * 4 + 2];
        point dat_4 = dat_arr[gi * 4 + 3];

        for (int i = 0; i < array_size; i++){
            point p = arr[i];
            res.x = res.x | (p.x == dat_1.x && p.y == dat_1.y);
            res.y = res.y | (p.x == dat_2.x && p.y == dat_2.y);
            res.z = res.z | (p.x == dat_3.x && p.y == dat_3.y);
            res.w = res.w | (p.x == dat_4.x && p.y == dat_4.y);
        }
        return {!res.x, !res.y, !res.z, !res.w};
    }
};

void load_contours_to_device (point * d_contours, vector<vector<Point>> &contours, Sizes * sizes) {
    point *h_contours;
    cudaError_t err;

    h_contours = (point *)malloc(sizes->contours_linear_size * sizeof(point));
    

    int idx = 0;
    for (int i = 0; i < sizes->number_of_contours; i++){
        for (int j = 0; j < contours[i].size(); j++){
            h_contours[idx] = {contours[i][j].x, contours[i][j].y};

            idx++;
        }
    }
    
    err = cudaMemcpy(d_contours, h_contours, sizes->contours_linear_size * sizeof(point), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    
    free(h_contours);
}

void filter_contour_by_hand_wrapper(point * d_contours_out, int * h_contours_sizes_out, vector<vector<Point>> &contours, unordered_set<Point, HashFunction> &_excluded_points, Sizes * sizes, int ngroups = 1024, int lws = 256){
    point *d_excluded_points, *d_contours;
    point *h_excluded_points;
    int *d_flags;
    int excluded_points_size = _excluded_points.size();
    cudaError_t err;

    // allocate memory on the host for all the points
    h_excluded_points= (point *)malloc(excluded_points_size * sizeof(point));
    
    err = cudaMalloc((void **)&d_contours, sizes->contours_linear_size * sizeof(point)); cuda_err_check(err, __FILE__, __LINE__);

    load_contours_to_device(d_contours, contours, sizes);

    int idx = 0;
    for (auto it = _excluded_points.begin(); it != _excluded_points.end(); it++){
        h_excluded_points[idx] = {it->x, it->y};
        idx++;
    }

    err = cudaMalloc((void **)&d_excluded_points, excluded_points_size * sizeof(point)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_flags, sizes->contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_excluded_points, h_excluded_points, excluded_points_size * sizeof(point), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    int nquarts_flags = round_div_up(sizes->contours_linear_size, 4);

    #if PROFILING_HAND
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    #endif

    compute_flags<check_array_membership><<<round_div_up(nquarts_flags, 256), 256>>>(nquarts_flags, (int4*)d_flags, d_contours, d_excluded_points, excluded_points_size);
    
    #if PROFILING_HAND
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("compute_flags hand time: %f\n", milliseconds);
    printf("GE/s: %f\n", (float)sizes->contours_linear_size / milliseconds / 1e6);
    printf("GB/s: %f\n", ((float)sizes->contours_linear_size * sizeof(int) * 3 + (float)sizes->contours_linear_size * excluded_points_size * sizeof(int))/ milliseconds / 1e6);
    #endif

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_FLAGS
    printf("Flags computed: ");
    print_array_dev(d_flags, sizes->contours_linear_size);
    printf("\n");
    #endif

    filter_contour(d_contours, h_contours_sizes_out, d_contours_out, d_flags, sizes, ngroups, lws);

    free(h_excluded_points);

    cudaFree(d_contours);
    cudaFree(d_excluded_points);
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


    // filter_contour_by_hand_wrapper(d_contours, d_contours_y, h_contours_sizes, contours, excluded_points, sizes);

    for (int i = 0; i < contours.size(); i++){
        for (int j = 0; j < contours[i].size(); j++){
            printf("(%d, %d) ", contours[i][j].x, contours[i][j].y);
        }
        printf("\n");
    }
    return 0;
}
#endif