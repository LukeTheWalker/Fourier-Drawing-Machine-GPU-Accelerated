#ifndef STREAM_COMPACTION_H
#define STREAM_COMPACTION_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__global__ void scan_fixup (int4 *d_out, int *d_tails, int nquarts, int preferred_wg_multiple);
__global__ void compute_flags(int4 *d_contours_x, int4 *d_contours_y, int4 *d_excluded_points_x, int4 *d_excluded_points_y, int4 *d_flag, int nquarts, int excluded_points_size);
__global__ void move_elements (int *d_contours_x, int *d_contours_y, int *dest_x, int *dest_y, int *d_flags, int *d_positions, int nels);
__global__ void scan_sliding_window (int4 *d_in, int4 *d_out, int *d_tails,int nquarts, int preferred_wg_multiple);

#endif