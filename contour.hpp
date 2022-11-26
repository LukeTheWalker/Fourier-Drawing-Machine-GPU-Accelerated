#ifndef CONTOUR
#define CONTOUR

#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

struct callback_data{
    int treshold;
    int min_size;
    Mat src;
};

void filter_vector_by_min (vector<vector<Point>> &, int );

void remove_all_duplicate_points (vector<vector<Point>> &);

void biggest_contour_first(vector<vector<Point>> &);

double minimum_distance(vector<Point> &p1, vector<Point> &);

void compute_minimum_distance_beetween_contours(vector<vector<Point>> &, vector<vector<double>> &);

void order_clusters_by_minimum_distance (vector<vector<Point>> &, vector<vector<double>> &, vector<vector<Point>> & );

void draw_minimum_distances(vector<vector<Point>> &, vector<vector<double>> &, Mat &);

void apply_contours(Mat & , int , int , vector<vector <Point> > & );

static void thresh_callback(int, void* );

static pair<int, int> findTresholds(Mat & );

pair<int, int> get_treshold(string, Mat & );

#endif