#include <iostream>
#include <fstream>
#include <unordered_set>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.cuh"
#include "filter_contour_by_hand.cu"
#include "contour.hpp"

using namespace cv;
using namespace std;

static const char *contour_window = "Contour Map";
static const char *original_window = "Original";

static vector<vector<double> > DEBUG_distances;

/*void from_device_to_host(int * d_contours_x, int * d_contours_y, int * d_contours_sizes, int number_of_countours){
    int * h_positions;
    cudaError_t err;

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
}*/

void call_back_func(int event, int x, int y, int flags, void* _data){
    CallbackData *data = (CallbackData *) _data;
    if (event == EVENT_LBUTTONDOWN)
        data->thresholds.brush_size+=5;
    if (event == EVENT_RBUTTONDOWN)
        data->thresholds.brush_size-=5;
    if (flags & EVENT_FLAG_ALTKEY){
        int brush_size = data -> thresholds.brush_size;
        data->cursor = Point(x, y);
        // exclude all points in a radius of 10 pixels
        for (int i = -brush_size; i < brush_size; i++){
            for (int j = -brush_size; j < brush_size; j++){
                data->thresholds.excluded_points.insert(Point(x + i, y + j));
            }
        }
        thresh_callback(0, _data);
    }
}

void filter_contour_by_hand(vector<vector<Point>> &contours, unordered_set<Point, HashFunction> &_excluded_points){
    contours.erase(
        remove_if (contours.begin(), contours.end(), 
            [&_excluded_points](vector<Point> vec){
                for (int i = 0; i < vec.size(); i++){
                    if (_excluded_points.find(vec[i]) != _excluded_points.end()){
                        return true;
                    }
                }
                return false;
            }
        ),
        contours.end()
    );
}

void merge_close_contours (vector<vector<Point>> &contours, double min_distance){
    // merge contours that are close to each other
    if (min_distance < 1){
        return;
    }
    vector<vector<Point>> new_contours;
    vector<bool> used(contours.size(), false);
    for (int i = 0; i < contours.size(); i++){
        if (used[i]){
            continue;
        }
        vector<Point> new_contour = contours[i];
        for (int j = i + 1; j < contours.size(); j++){
            if (used[j]){
                continue;
            }
            for (int k = 0; k < contours[j].size(); k++){
                for (int l = 0; l < new_contour.size(); l++){
                    if (norm(contours[j][k] - new_contour[l]) < min_distance){
                        new_contour.insert(new_contour.end(), contours[j].begin(), contours[j].end());
                        used[j] = true;
                        break;
                    }
                }
            }
        }
        new_contours.push_back(new_contour);
    }
    contours = new_contours;
}

void filter_vector_by_min (vector<vector<Point>> &contours, int min_size){
    contours.erase(
        remove_if (contours.begin(), contours.end(), 
            [min_size](vector<Point> vec){
                return vec.size() < min_size;
            }
        ),
        contours.end()
    );
}

void remove_all_duplicate_points (vector<vector<Point>> &contours){
    for (int i = 0; i < contours.size(); i++){
        remove_vector_duplicates(contours[i]);
    }
}

void biggest_contour_first(vector<vector<Point>> &contours){
    // put biggest contour first without sorting the whole vector
    int max_index = 0;
    for (int i = 1; i < contours.size(); i++){
        if (contours[i].size() > contours[max_index].size()){
            max_index = i;
        }
    }
    swap(contours[0], contours[max_index]);
}

double minimum_distance(vector<Point> &p1, vector<Point> &p2){
    return norm(p1.back() - p2[0]);
}

void compute_minimum_distance_beetween_contours(vector<vector<Point>> &contours, vector<vector<double>> &distances){
    distances = vector<vector<double>>(contours.size(), vector<double>(contours.size(), 0));
    for (int i = 0; i < contours.size(); i++){
        for (int j = 0; j < contours.size(); j++){
            if (i == j){
                distances[i][j] = 0;
                continue;
            }
            distances[i][j] = minimum_distance(contours[i], contours[j]);
        }
    }
}

void order_clusters_by_minimum_distance (vector<vector<Point>> &contours, vector<vector<double>> &distances, vector<vector<Point>> & points){
    vector<bool> visited(contours.size(), false);
    int node = 0;
    visited[node] = true;
    points.push_back(contours[node]);

    // chissa se funziona
    for (size_t i = 1; i < contours.size(); i++)
    {
        // find the closest node
        double min_dist = 100000000;
        int min_index = -1;
        for (size_t j = 0; j < contours.size(); j++)
        {
            if (visited[j])
                continue;
            double dist = distances[node][j];
            if (dist < min_dist)
            {
                min_dist = dist;
                min_index = j;
            }
        }
        visited[min_index] = true;
        node = min_index;
        points.push_back(contours[node]);
    }    
}

// use when unsure that the reordering actually works, maybe we calculate an ordering but for some reason we don't use it
void draw_minimum_distances(vector<vector<Point>> &points, vector<vector<double>> &distances, Mat &dst){
    for (int i = 1; i < points.size(); i++){
        // draw distance between cluster i and i-1
        Point p1 = points[i-1].back();
        Point p2 = points[i][0];
        double distance = distances[i-1][i];

        // draw line between cluster i and i-1
        line(dst, p1, p2, Scalar(0, 0, 255), 1, LINE_AA);

        // draw point p1
        circle(dst, p1, 3, Scalar(255, 0, 255), -1, LINE_AA);

        // draw point p2
        circle(dst, p2, 3, Scalar(0, 255, 255), -1, LINE_AA);

        // imshow(window_name, dst);

        // waitKey();
    }
}

void cpu_pipeline(vector<vector<Point> > & contours, unordered_set<Point, HashFunction> & excluded_points, double merging_distance, int min_size) {
    // filter_contour_by_hand(contours, excluded_points);
    filter_contour_by_hand_wrapper(contours, excluded_points);
    merge_close_contours(contours, merging_distance);
    filter_vector_by_min(contours, min_size);
    remove_all_duplicate_points(contours);
    biggest_contour_first(contours);
}

void cuda_pipeline(vector<vector<Point> > & contours, unordered_set<Point, HashFunction> & excluded_points, double merging_distance, int min_size) {
    filter_contour_by_hand_wrapper(contours, excluded_points);
    // merge_close_contours(contours, merging_distance);
    // filter_vector_by_min(contours, min_size);
    // remove_all_duplicate_points(contours);
    // biggest_contour_first(contours);
}

void apply_contours(Mat & src, Thresholds thresholds ,vector<vector <Point> > & points){
    int canny_low = thresholds.canny_low;
    int canny_high = thresholds.canny_high;
    int sigma = thresholds.sigma;
    int min_size = thresholds.min_size;
    double merging_distance = thresholds.merging_distance/10.0;
    Mat canny_output, src_gray;
    vector<vector<Point> > contours;

    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    // blur( src_gray, src_gray, Size(3,3) );
    GaussianBlur(src_gray, src_gray, Size(5,5), sigma);
    
    Canny( src_gray, canny_output, canny_low, canny_high );

    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );

    cpu_pipeline(contours, thresholds.excluded_points, merging_distance, min_size);
    // cuda_pipeline(contours, thresholds.excluded_points, merging_distance, min_size);

    vector<vector<double>> distances;
    compute_minimum_distance_beetween_contours(contours, distances);
    order_clusters_by_minimum_distance(contours, distances, points);
    
    // return points;
}

static void thresh_callback(int, void* _data)
{
    CallbackData * data = (CallbackData *)_data;

    vector<vector<Point> > contours;
    Mat drawing = Mat::zeros( data->src.size(), CV_8UC3 );
    Mat src_contours = data->src.clone();

    apply_contours(data->src, data->thresholds, contours);

    polylines(drawing, contours, false, Scalar(255, 255, 255), 2, LINE_AA, 0);

    // draw distances
    // draw_minimum_distances(contours, DEBUG_distances, drawing);
    for (int i = 1; i < contours.size(); i++){
        circle(drawing, contours[i-1][0], 3, Scalar(0, 255, 0), -1, LINE_AA);
        circle(drawing, contours[i].back(), 3, Scalar(255, 0, 0), -1, LINE_AA);
        line(drawing, contours[i-1][0], contours[i].back(), Scalar(0, 0, 255), 1, LINE_AA);
    }

    // draw contours on the original image
    // polylines(src_contours, contours, false, Scalar(0, 0, 255), 2, LINE_AA, 0);
    RNG rng;
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        polylines(src_contours, contours[i], false, color, 2, LINE_AA, 0);
    }

    circle( src_contours, data->cursor, data->thresholds.brush_size, Scalar( 255, 0, 0 ), -1);

    data->cursor = Point(0,0);

    imshow( contour_window, drawing );
    imshow( original_window, src_contours );
}

static Thresholds findTresholds(Mat & src)
{
    CallbackData data;
    data.src = src;
    data.cursor = Point(-1, -1);
    data.thresholds.sigma = 3;
    data.thresholds.canny_low = 200;
    data.thresholds.canny_high = 200;
    data.thresholds.min_size = 0;
    data.thresholds.brush_size = 10;
    data.thresholds.merging_distance = 0;

    
    /// Create Window
    namedWindow( contour_window );
    imshow( original_window, src );

    const int max_thresh = 255;
    createTrackbar( "Sigma:", contour_window, &data.thresholds.sigma, 10, thresh_callback, &data );
    createTrackbar( "Canny low:", contour_window, &data.thresholds.canny_low, max_thresh, thresh_callback, &data );
    createTrackbar( "Canny high:", contour_window, &data.thresholds.canny_high, max_thresh, thresh_callback, &data );
    createTrackbar( "Merging distance", contour_window, &data.thresholds.merging_distance, 100, thresh_callback, &data );
    createTrackbar( "Brush size", contour_window, &data.thresholds.brush_size, 100, thresh_callback, &data );
    createTrackbar( "Min Size:", contour_window, &data.thresholds.min_size, 100, thresh_callback, &data );
    setMouseCallback(original_window, call_back_func, &data);
    thresh_callback( 0, &data );

    int key = waitKey();
    while (key != 114){
        key = waitKey();
    }

    return data.thresholds;
}

Thresholds get_treshold(string file_name, Mat & src)
{
    // read tresholds from file if file exists else get them from function
    Thresholds tresholds;
    ifstream file(file_name);
    if (file.is_open()){
        file >> tresholds.sigma;
        file >> tresholds.canny_low;
        file >> tresholds.canny_high;
        file >> tresholds.min_size;
        file >> tresholds.merging_distance;
        file >> tresholds.brush_size;
        // load all excluded points
        int x, y;
        while (file >> x >> y){
            tresholds.excluded_points.insert(Point(x, y));
        }
        file.close();
    } else {
        tresholds = findTresholds(src);
        ofstream file(file_name);
        if (file.is_open()){
            file << tresholds.sigma << endl;
            file << tresholds.canny_low << endl;
            file << tresholds.canny_high << endl;
            file << tresholds.min_size << endl;
            file << tresholds.merging_distance << endl;
            file << tresholds.brush_size << endl;
            // save all excluded points
            for (auto point : tresholds.excluded_points){
                file << point.x << " " << point.y << endl;
            }
            file.close();
        }
    }
    return tresholds;
}