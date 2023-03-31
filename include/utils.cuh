#ifndef UTILS
#define UTILS

#include <vector>
#include <iostream>
#include <string>
#include <set>
#include <cuda.h>
#include <cuda_runtime_api.h>

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

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

template<typename T>
void print_vector(std::vector<T> vec)
{
  std::cout << "size: " << vec.size() << std::endl;
  for (T d : vec)
    std::cout << d << " ";
  std::cout << std::endl;
}

void cuda_err_check (cudaError_t err, const char *file, int line);

template<typename T>
void print_array_dev (T * d_a, int n) {
    for (int i = 0; i < n; i++){
        T tail;
        printf("Accessing address %p\n", d_a+i);
        cudaError_t err = cudaMemcpy(&tail, d_a+i, sizeof(T), cudaMemcpyDeviceToHost);
        cuda_err_check(err, __FILE__, __LINE__);
        printf("%u ", tail);
    }
}

string type2str(int);

int round_div_up (int a, int b);
int round_mul_up (int a, int b);

#endif