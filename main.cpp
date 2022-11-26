#include <iostream>
#include <fstream>
#include <gmpxx.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "contour.hpp"
#include "utils.hpp"
#include "dft.hpp"

using namespace cv;
using namespace std;

typedef vector<Point> vp;

Mat src, edges;

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, "{@input | fruits.jpg | input image}");
    src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR); // Load an image

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << std::endl;
        cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }

    Size src_size = src.size();
    int src_type = CV_8UC1;

    pair<int, int> threshold = get_treshold("contour.txt", src);

    vector<vp> points;
    apply_contours(src, threshold.first, threshold.second, points);

    vector<epycicle> fourierXY; 
    dft(points, fourierXY);

    sort_epycicles(fourierXY);

    Point prev;
    mpf_class time = 0;

    Mat drawing;

    char filename[128];

    #if 1

    int n_frames = 600;
    int last_frame = 0;
    int cnt = 0;
    int line_thickness = 2;
    vector<vector<Point> > points_drawn{vector<Point>()};
    
    for (int i = 0; i < points.size(); i++){
        for (int j = 0; j < points[i].size(); j++){
            drawing = Mat::zeros(src_size, src_type);

            Point v = fourier_drawer(drawing, 0, 0, 0, fourierXY, time);

            points_drawn.back().push_back(v);

            // for(int k = 0; k < i; k++)
            //     polylines(drawing, points_drawn[k], false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
            // polylines(drawing, points_drawn.back(), false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
            
            polylines(drawing, points_drawn, false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);

            prev = v;
            time += ((2 * M_PI) / fourierXY.size());
            cnt++;

            if (cnt % (fourierXY.size() / n_frames) == 0){
                last_frame = cnt / (fourierXY.size() / n_frames);
                sprintf(filename, "output/gif-%05d.png", last_frame);
                imwrite(filename, drawing);
            }

            // imshow("drawing", drawing);
            // waitKey(1);
        }
        points_drawn.push_back(vector<Point>());
    }

    for (int j = 1; j < 60; j++){
        drawing = Mat::zeros(src_size, src_type);
        for(auto &p : points)
            polylines(drawing, p, false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
        sprintf(filename, "output/gif-%05d.png", j + last_frame);
        imwrite(filename, drawing);
    } 
    
    #endif
    
    printf("All frames generated\n");

    // waitKey(0);

    return 0;
}