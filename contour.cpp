#include <iostream>
#include <fstream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

static const char *window_name = "Contour Map";

 struct callback_data{
    int treshold;
    int min_size;
    Mat src;
};

vector<vector<double> > DEBUG_distances;

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

void apply_contours(Mat & src, int threshold, int min_size, vector<vector <Point> > & points){
    Mat canny_output, src_gray;
    vector<vector<Point> > contours;

    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );
    
    Canny( src_gray, canny_output, threshold, threshold*2 );

    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1 );

    filter_vector_by_min(contours, min_size);

    remove_all_duplicate_points(contours);

    biggest_contour_first(contours);

    vector<vector<double>> distances;
    compute_minimum_distance_beetween_contours(contours, distances);
    order_clusters_by_minimum_distance(contours, distances, points);
    
    // return points;
}

static void thresh_callback(int, void* _data)
{
    callback_data * data = (callback_data *)_data;

    vector<vector<Point> > contours;
    Mat drawing = Mat::zeros( data->src.size(), CV_8UC3 );

    apply_contours(data->src, data->treshold, data->min_size, contours);

    polylines(drawing, contours, false, Scalar(255, 255, 255), 2, LINE_AA, 0);

    // draw distances
    // draw_minimum_distances(contours, DEBUG_distances, drawing);
    for (int i = 1; i < contours.size(); i++){
        circle(drawing, contours[i-1][0], 3, Scalar(0, 255, 0), -1, LINE_AA);
        circle(drawing, contours[i].back(), 3, Scalar(255, 0, 0), -1, LINE_AA);
        line(drawing, contours[i-1][0], contours[i].back(), Scalar(0, 0, 255), 1, LINE_AA);
    }

    // Show in a window 
    imshow( window_name, drawing );
}

static pair<int, int> findTresholds(Mat & src)
{
    callback_data data;
    data.src = src;
    data.treshold = 10;
    data.min_size = 0;
    
    /// Create Window
    namedWindow( window_name );
    imshow( "Original", src );

    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", window_name, &data.treshold, max_thresh, thresh_callback, &data );
    createTrackbar( "Min Size:", window_name, &data.min_size, 100, thresh_callback, &data );
    thresh_callback( 0, &data );

    waitKey();

    return make_pair(data.treshold, data.min_size);
}

pair<int, int> get_treshold(string file_name, Mat & src)
{
    // read tresholds from file if file exists else get them from function
    pair<int, int> tresholds;
    ifstream file(file_name);
    if (file.is_open())
    {
        file >> tresholds.first >> tresholds.second;
        file.close();
    }
    else
    {
        tresholds = findTresholds(src);
        ofstream file(file_name);
        file << tresholds.first << " " << tresholds.second;
        file.close();
    }
    return tresholds;
}