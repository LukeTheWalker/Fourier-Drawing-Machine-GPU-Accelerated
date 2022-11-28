#include <iostream>
#include <fstream>
#include <gmpxx.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "contour.hpp"
#include "utils.hpp"
#include "dft.hpp"
#include "progress.hpp"

using namespace cv;
using namespace std;

typedef vector<Point> vp;

Mat src, edges;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: %s <Image_Path>\n", argv[0]);
        return -1;
    }

    src = imread(argv[1], IMREAD_COLOR); // Load an image

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << std::endl;
        return -1;
    }

    Size src_size = src.size();
    int src_type = CV_8UC1;

    Thresholds threshold = get_treshold("data/contour.txt", src);

    vector<vp> points;
    apply_contours(src, threshold, points);

    vector<epycicle> fourierXY; 
    size_t n_points = dft(points, fourierXY);

    sort_epycicles(fourierXY);

    Point prev;
    mpf_class time = 0;

    Mat drawing;

    char filename[128];

    int n_frames = 600;
    int last_frame = 0;
    int cnt = 0;
    int line_thickness = 2;
    vector<vector<Point> > points_drawn{vector<Point>()};

    progressbar bar(n_frames);

    vector<double> linspaced = linspace(0, (int)n_points, n_frames);
    
    for (int i = 0; i < points.size(); i++){
        for (int j = 0; j < points[i].size(); j++){
            fourier_state state;
            fourier_calculator(0, 0, 0, fourierXY, time, state);
            Point v = state.arm.back();

            points_drawn.back().push_back(v);

            // for(int k = 0; k < i; k++)
            //     polylines(drawing, points_drawn[k], false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
            // polylines(drawing, points_drawn.back(), false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
            
            prev = v;
            time += ((2 * M_PI) / n_points);
            cnt++;

            if ( linspaced[last_frame] - cnt < 1 ){
                drawing = Mat::zeros(src_size, src_type);
                fourier_drawer(drawing, state);
                polylines(drawing, points_drawn, false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
                bar.update();
                last_frame++;
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
        polylines(drawing, points_drawn, false, Scalar(255, 255, 255), line_thickness, LINE_AA, 0);
        sprintf(filename, "output/gif-%05d.png", j + last_frame);
        imwrite(filename, drawing);
    } 
        
    printf("\nAll frames generated\n");

    // waitKey(0);

    return 0;
}