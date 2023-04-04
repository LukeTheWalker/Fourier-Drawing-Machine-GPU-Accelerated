#ifndef FILTER_CONTOUR_BY_HAND_WRAPPER_H
#define FILTER_CONTOUR_BY_HAND_WRAPPER_H

#include <vector>
#include <unordered_set>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "contour.hpp"

#include "utils.cuh"
#include "streamCompaction.cuh"

void filter_contour_by_hand_wrapper(vector<vector<Point>> &contours, unordered_set<Point, HashFunction> &_excluded_points, int ngroups = 1024, int lws = 256); 

#endif