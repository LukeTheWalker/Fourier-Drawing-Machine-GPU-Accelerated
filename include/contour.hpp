#ifndef CONTOUR
#define CONTOUR

#include <iostream>
#include <fstream>
#include <unordered_set>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

struct HashFunction
  {
    size_t operator()(const Point& point) const
    {
      size_t xHash = std::hash<int>()(point.x);
      size_t yHash = std::hash<int>()(point.y) << 1;
      return xHash ^ yHash;
    }
  };

struct Thresholds {
    int sigma;
    int canny_low;
    int canny_high;
    int min_size;
    int brush_size;
    int merging_distance;
    unordered_set<Point, HashFunction> excluded_points;
};

struct CallbackData{
    Thresholds thresholds;
    Point cursor;
    Mat src;
};

void filter_vector_by_min (vector<vector<Point>> &, int );

void remove_all_duplicate_points (vector<vector<Point>> &);

void biggest_contour_first(vector<vector<Point>> &);

double minimum_distance(vector<Point> &p1, vector<Point> &);

void compute_minimum_distance_beetween_contours(vector<vector<Point>> &, vector<vector<double>> &);

void order_clusters_by_minimum_distance (vector<vector<Point>> &, vector<vector<double>> &, vector<vector<Point>> & );

void draw_minimum_distances(vector<vector<Point>> &, vector<vector<double>> &, Mat &);

void apply_contours(Mat &, Thresholds, vector<vector <Point> > & );

static void thresh_callback(int, void* );

static Thresholds findTresholds(Mat & );

Thresholds get_treshold(string, Mat & );

#endif