#ifndef STREAM_COMPACTION_H
#define STREAM_COMPACTION_H

#define PRINT 1

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <vector>
#include <string>
#include <map>

#include "utils.cuh"

__device__ int round_div_up_dev (int a, int b){
    return (a + b - 1)/b;
}

__device__ int round_mul_up_dev (int a, int b){
    return round_div_up_dev(a, b)*b;
}

// scan of all elements in the workgroup window
__device__ int scan_single_element(
    int first_wg_el, 
    int end_wg_el, 
    int tail, 
    int4 *out, 
    int4 *in, 
    int *lmem
){
    int4 val = {0, 0, 0, 0};
    int li = threadIdx.x;
    int gi = first_wg_el + li;
    int lws = blockDim.x;
    // scan of single work-item (quart of int)
    if (gi < end_wg_el){
        val = {in[gi].x, in[gi].y, in[gi].z, in[gi].w};
        // val.s1 += val.s0 ; val.s3 += val.s2;
        val.y += val.x; val.w += val.z;
        // val.s2 += val.s1 ; val.s3 += val.s1;
        val.z += val.y; val.w += val.y;
    }

    // write work-item tail to local memory to sync with rest of workgroup
    lmem[li] = val.w;

    // scan of local memory
    __syncthreads();

    for (int active_mask = 1; active_mask < lws; active_mask *= 2) {
		int pull_mask = active_mask - 1;
		pull_mask = ~pull_mask;
		pull_mask = li & pull_mask;
		pull_mask = pull_mask - 1;
		__syncthreads();
		if (li & active_mask) lmem[li] += lmem[pull_mask];
	}
    __syncthreads();

    // each work-item adds the previous work-item tail
    if (li > 0){
        val.x += lmem[li - 1]; 
        val.y += lmem[li - 1];
        val.z += lmem[li - 1]; 
        val.w += lmem[li - 1];
    }

    // each work-item adds the previous windows's tail
	val.x += tail; 
    val.y += tail;
    val.z += tail; 
    val.w += tail;

    // only write to global memory if the work-item is in the window
	if (gi < end_wg_el)
		out[gi] = val;

    // return the last element of the local memory
	return lmem[lws - 1];
}

__global__ void scan_sliding_window(
    int4 *d_in, 
    int4 *d_out,
    int *d_tails,
    int nquarts,
	int preferred_wg_multiple
){  
    extern __shared__ int lmem[];

    int nwg = gridDim.x;
    int wg_id = blockIdx.x;
    int lws = blockDim.x;

    // elemensts per workgroup
    // (a + b - 1)/b;
    int els_per_wg = (nquarts + nwg - 1)/nwg;

    // optimize wavefront size
    //  round_div_up_dev(a, b)*b;
    els_per_wg = ((els_per_wg + preferred_wg_multiple - 1)/preferred_wg_multiple)*preferred_wg_multiple;

    // first and last element of the workgroup
    int first_el = wg_id*els_per_wg;
    int last_el = min(first_el + els_per_wg, nquarts);

    // get local id 
    int lid = threadIdx.x;

    // tail initialization
    int tail = 0;

    while (first_el < last_el) {
		tail += scan_single_element(first_el, last_el, tail, d_out, d_in, lmem);
		first_el += lws;
		__syncthreads();
	}
    
    if (nwg > 1 && lid == 0) {
		d_tails[wg_id] = tail;
	}
}

__global__ void scan_fixup (int4 *d_out, int *d_tails, int nquarts, int preferred_wg_multiple){
    const int nwg = gridDim.x;
	const int group_id = blockIdx.x;
    const int lws = blockDim.x;

	if (group_id == 0) return;

	// elements per work-group
	int els_per_wg = round_div_up_dev(nquarts, nwg);
	els_per_wg = round_mul_up_dev(els_per_wg, preferred_wg_multiple);

	// index of first element assigned to this work-group
	int first_el = els_per_wg*group_id;
	// index of first element NOT assigned to us
    const int last_el =  min(nquarts, els_per_wg*(group_id+1));

	int fixup = d_tails[group_id-1];
	int gi = first_el + threadIdx.x;
	while (gi < last_el) {
        // printf("Before fixup: %d, %d, %d, %d, gi: %d\n", d_out[gi].x, d_out[gi].y, d_out[gi].z, d_out[gi].w, gi);
		d_out[gi].x += fixup;
        d_out[gi].y += fixup;
        d_out[gi].z += fixup;
        d_out[gi].w += fixup;
        // printf("After fixup: %d, %d, %d, %d, gi: %d\n", d_out[gi].x, d_out[gi].y, d_out[gi].z, d_out[gi].w, gi);
		gi += lws;
	}
}

template <typename F>
__global__ void compute_flags(int4 *d_contours_x, int4 *d_contours_y, int4 *d_excluded_points_x, int4 *d_excluded_points_y, int4 *d_flag, int nquarts, int n_quarts_excluded_points_size){
    int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi < nquarts) {
        d_flag[gi] = F()(d_contours_x[gi], d_contours_y[gi], d_excluded_points_x, d_excluded_points_y, n_quarts_excluded_points_size);
    }
}

__global__ void move_elements (int *d_contours_x, int *d_contours_y, int *dest_x, int *dest_y, int *d_flags, int *d_positions, int nels){
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    if (gi >= nels) return;
    if (!d_flags[gi]) return;
    if (gi == 0) { dest_x[0] = d_contours_x[0]; dest_y[0] = d_contours_y[0]; return; }
    
    int pos = d_positions[gi - 1];

    dest_x[pos] = d_contours_x[gi];
    dest_y[pos] = d_contours_y[gi];
}

void get_after_filter_contours(vector<vector<Point>> &after_filter_contours, int *d_contours_x_out, int *d_contours_y_out, int after_filter_contours_linear_size, int *after_filter_contours_sizes, int number_of_contours){
    int *h_contours_x_out = (int*)malloc(after_filter_contours_linear_size * sizeof(int));
    int *h_contours_y_out = (int*)malloc(after_filter_contours_linear_size * sizeof(int));

    cudaError_t err = cudaMemcpy(h_contours_x_out, d_contours_x_out, after_filter_contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(h_contours_y_out, d_contours_y_out, after_filter_contours_linear_size * sizeof(int), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    int idx = 0;
    // cout << "Number of contours: " << after_filter_number_of_contours << endl;
    for (int i = 0; i < number_of_contours; i++){
        vector<Point> contour;
        for (int j = 0; j < after_filter_contours_sizes[i]; j++){
            contour.push_back(Point(h_contours_x_out[idx], h_contours_y_out[idx]));
            idx++;
        }
        if (contour.size() > 0)
            after_filter_contours.push_back(contour);
    }

    free(h_contours_x_out);
    free(h_contours_y_out);
}

#endif