#define PRINT_CONTOUR 0
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cuh"

using namespace std;

void filter_contour_by_hand_wrapper(vector<vector<Point>> &contours, unordered_set<Point, HashFunction> &_excluded_points, int ngroups, int lws){
    int *d_contours_x, *d_contours_y, *d_excluded_points_x, *d_excluded_points_y, *d_contours_sizes;
    int *h_contours_x, *h_contours_y, *h_excluded_points_x, *h_excluded_points_y, *h_contours_sizes;
    int *d_flags;
    int contours_linear_size = 0;
    int excluded_points_size = _excluded_points.size();
    int number_of_countours = contours.size();

    // calculate linear size as sum of all single sub-vectors
    for (int i = 0; i < contours.size(); i++) contours_linear_size += contours[i].size();
    
    // allocate memory on the host for all the points
    h_contours_x = (int *)malloc(contours_linear_size * sizeof(int));
    h_contours_y = (int *)malloc(contours_linear_size * sizeof(int));
    h_excluded_points_x = (int *)malloc(excluded_points_size * sizeof(int));
    h_excluded_points_y = (int *)malloc(excluded_points_size * sizeof(int));
    h_contours_sizes = (int *)malloc(number_of_countours * sizeof(int));

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

    idx = 0;
    for (auto it = _excluded_points.begin(); it != _excluded_points.end(); it++){
        h_excluded_points_x[idx] = it->x;
        h_excluded_points_y[idx] = it->y;
        idx++;
    }

    cudaError_t err;


    err = cudaMalloc((void **)&d_contours_x, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_excluded_points_x, excluded_points_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_excluded_points_y, excluded_points_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_sizes, number_of_countours * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_flags, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(d_contours_x, h_contours_x, contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_y, h_contours_y, contours_linear_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_excluded_points_x, h_excluded_points_x, excluded_points_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_excluded_points_y, h_excluded_points_y, excluded_points_size * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_contours_sizes, h_contours_sizes, number_of_countours * sizeof(int), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    int nquarts_flags = round_div_up(contours_linear_size, 4);
    int nquarts_excluded_points = round_div_up(excluded_points_size, 4);
    compute_flags<<<round_div_up(nquarts_flags, 256), 256>>>((int4*)d_contours_x, (int4*)d_contours_y, (int4*)d_excluded_points_x, (int4*)d_excluded_points_y, (int4*)d_flags, nquarts_flags, nquarts_excluded_points);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_CONTOUR
    printf("Flags computed: ");
    print_array_dev(d_flags, contours_linear_size);
    printf("\n");
    #endif

    int *d_positions, *h_positions, *d_tails;

    int ntails = ngroups > 1 ? round_mul_up(ngroups, 4) : ngroups;
    err = cudaMalloc((void **)&d_positions, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_tails, ngroups * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    scan_sliding_window<<<ngroups, lws, lws*sizeof(int)>>>((int4*)d_flags, (int4*)d_positions, d_tails, round_div_up(contours_linear_size, 4), 32);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_CONTOUR
    printf("Positions computed: ");
    print_array_dev(d_positions, contours_linear_size);
    printf("\n");
    #endif

    if (ngroups > 1){
        scan_sliding_window<<<1, lws, lws*sizeof(int)>>>((int4*)d_tails, (int4*)d_tails, NULL, round_div_up(ntails, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    // printf("Tails computed: ");
    // print_array_dev(d_tails, ngroups);
    // printf("\n");

    if (ngroups > 1){
        scan_fixup<<<ngroups, lws>>>((int4*)d_positions, d_tails, round_div_up(contours_linear_size, 4), 32);
        err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    }

    #if PRINT_CONTOUR
    printf("Positions computed: ");
    print_array_dev(d_positions, contours_linear_size);
    printf("\n");
    #endif

    int *d_contours_x_out, *d_contours_y_out;

    err = cudaMalloc((void **)&d_contours_x_out, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void **)&d_contours_y_out, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);

    move_elements<<<round_div_up(contours_linear_size, lws), lws>>>(d_contours_x, d_contours_y, d_contours_x_out, d_contours_y_out, d_flags, d_positions, contours_linear_size);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    #if PRINT_CONTOUR
    printf("Contour x computed: ");
    print_array_dev(d_contours_x_out, contours_linear_size);
    printf("\n");
    #endif

    err = cudaMallocHost(&h_positions, contours_linear_size * sizeof(int)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_positions, d_positions, contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int *after_filter_contours_sizes = (int*)malloc(number_of_countours * sizeof(int));
    uint32_t cnt = 0;
    vector<int> skip_contour;
    for (int i = 0; i < number_of_countours; i++){
        cnt += h_contours_sizes[i];
        after_filter_contours_sizes[i] = h_positions[cnt-1] - (i == 0 ? 0 : h_positions[cnt - h_contours_sizes[i] - 1]);
        if (after_filter_contours_sizes[i] == 0){
            skip_contour.push_back(i);
        }
        #if PRINT_CONTOUR
        printf("cnt = %d | ", cnt);
        printf("before: %d has size %d => ", i, h_contours_sizes[i]);
        printf("after : %d has size %d\n", i, after_filter_contours_sizes[i]);
        #endif
    }

    int *h_contours_x_out = (int*)malloc(contours_linear_size * sizeof(int));
    int *h_contours_y_out = (int*)malloc(contours_linear_size * sizeof(int));

    err = cudaMemcpy(h_contours_x_out, d_contours_x_out, contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_contours_y_out, d_contours_y_out, contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    vector<vector<Point>> after_filter_contours;
    idx = 0;
    for (int i = 0; i < number_of_countours; i++){
        if (find(skip_contour.begin(), skip_contour.end(), i) != skip_contour.end()) continue;
        
        vector<Point> contour;
        for (int j = 0; j < after_filter_contours_sizes[i]; j++){
            contour.push_back(Point(h_contours_x_out[idx], h_contours_y_out[idx]));
            idx++;
        }
        after_filter_contours.push_back(contour);
        cnt -= after_filter_contours_sizes[i];
    }
    contours = after_filter_contours;

    free(h_contours_x);
    free(h_contours_y);
    free(h_excluded_points_x);
    free(h_excluded_points_y);
    free(h_contours_sizes);
    free(h_contours_x_out);
    free(h_contours_y_out);
    free(after_filter_contours_sizes);

    cudaFreeHost(h_positions);

    cudaFree(d_contours_x);
    cudaFree(d_contours_y);
    cudaFree(d_excluded_points_x);
    cudaFree(d_excluded_points_y);
    cudaFree(d_contours_sizes);
    cudaFree(d_flags);
    cudaFree(d_positions);
    cudaFree(d_tails);
    cudaFree(d_contours_x_out);
    cudaFree(d_contours_y_out);

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

    filter_contour_by_hand_wrapper(contours, excluded_points, 256, 256);

    for (int i = 0; i < contours.size(); i++){
        for (int j = 0; j < contours[i].size(); j++){
            printf("(%d, %d) ", contours[i][j].x, contours[i][j].y);
        }
        printf("\n");
    }
    return 0;
}