#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include <string>
#include <set>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

template <typename T>
Mat& draw_points(Mat &img, const vector<Point> &points)
{
    for (int i = 0; i < points.size(); i++)
    {
        img.at<T>(points[i]) = 255;
    }
    return img;
}

template <typename T>
void checkpoint (const vector<Point> &points, Size size, int type, string window_name)
{
    Mat tmp = Mat::zeros(size, type);
    tmp = draw_points<T>(tmp, points);
    imshow(window_name, tmp);
}

template <typename T>
void animated_checkpoint (const vector<Point> points, Size size, int type, string window_name)
{
    for (int i = 0; i < points.size(); i++)
    {
        checkpoint<T>(vector<Point>(points.begin(), points.begin() + i), size, type, window_name);
        if (i % 100 == 0)
        {
            waitKey(1);
        }
    }
}

template<typename T>
void remove_vector_duplicates(vector<T>& input)
{
    vector<T> s;
    vector<T> v;
    for (auto it = input.begin(); it != input.end(); ++it)
    {
        if (find(s.begin(), s.end(), *it) == s.end())
        {
            s.push_back(*it);
            v.push_back(*it);
        }
    }
    input = v;
}

string type2str(int);

#endif