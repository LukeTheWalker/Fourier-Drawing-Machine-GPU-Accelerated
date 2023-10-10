#define FAST 1

#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <complex>
#include <gmpxx.h>
#include "opencv2/imgproc.hpp"
#include "utils.hpp"
#include "dft.hpp"

using namespace std;
using namespace cv;

void dft(vector<Point> &sig, vector<epycicle> &X, int K)
{
    int N = sig.size();
    for (int k = 0; k < K; k++)
    {
        cd Xk(0.0, 0.0);
        for (int n = 0; n < N; n++)
        {
            double phi = (2.0 * double(k) * M_PI * double(n)) / double(N);
            cd c(cos(phi), -sin(phi));
            cd sig_n(double(sig[n].x), double(sig[n].y));
            Xk = Xk + (sig_n * c);
        }
        Xk = Xk / double(N);

        double amp = sqrt(Xk.real() * Xk.real() + Xk.imag() * Xk.imag());
        double phase = arg(Xk);
        double freq = k;
        X.push_back({amp, phase, freq});
    }
}

size_t dft (vector<vector<Point> > &sig, vector<epycicle> &X){
    printf("Starting DFT\n");
    vector<Point> linearized_sig;
    for (vector<Point> &s : sig)
        linearized_sig.insert(linearized_sig.end(), s.begin(), s.end());

    printf("Running DFT on %lu points\n", linearized_sig.size());
    
    dft(linearized_sig, X, linearized_sig.size());

    printf("DFT done\n");

    return linearized_sig.size();
}

void sort_epycicles(vector<epycicle> &X)
{
    sort(
        X.begin(), X.end(),
        [](const auto &lhs, const auto &rhs)
        {
            return lhs.amp > rhs.amp;
        }
    );
}

Vec2d vec_from_angle_and_mag(double angle, double mag)
{
    return Vec2d(mag * cos(angle), mag * sin(angle));
}

// return point drawn
/*
Point fourier_drawer(Mat& img, double x, double y, float angle, vector<epycicle> &fourier, mpf_class time)
{
    double center_x = x;
    double center_y = y;

    Vec2d p_target;
    Vec2d target;
    vector<Point> arm;
    vector<Point> degenerate_ellipses;

    for (int k = 0; k < fourier.size(); k++)
    {
        float amp = fourier[k].amp;
        float phase = fourier[k].phase;
        float freq = fourier[k].freq;
        target += vec_from_angle_and_mag(time.get_d() * freq + phase + angle, amp);
        // ellipse (p_target.x, p_target.y, amp*2, amp*2);
        if (k != 0 && amp > 1){
            #if FAST
            ellipse(img, 
                Point(center_x + p_target[0], center_y + p_target[1]), 
                Size(amp, amp), 0, 0, 360, Scalar(100), 1, LINE_AA, 0);
            #else
            Mat overlay = Mat::zeros(img.size(), CV_8UC1);
            ellipse(overlay, 
                Point(center_x + p_target[0], center_y + p_target[1]), 
                Size(amp, amp), 0, 0, 360, Scalar(50), 1, LINE_AA, 0);
            img+=overlay;
            #endif
        }
        else if (amp <= 1)
            degenerate_ellipses.push_back(Point(center_x + p_target[0], center_y + p_target[1]));
        
        p_target = target;
        arm.push_back(Point(center_x + target[0], center_y + target[1]));
    }
    Mat overlay = Mat::zeros(img.size(), CV_8UC1);
    polylines(overlay, arm, false, Scalar(150), 1, LINE_AA);
    img+=overlay;

    overlay = Mat::zeros(img.size(), CV_8UC1);
    polylines(overlay, degenerate_ellipses, false, Scalar(100), 1, LINE_AA);
    img+=overlay;

    return Point(p_target) + Point(center_x, center_y);
}
*/

void fourier_calculator(double x, double y, float angle, vector<epycicle> &fourier, mpf_class time, fourier_state& state)
{
    double center_x = x;
    double center_y = y;

    Vec2d p_target;
    Vec2d target;
    
    for (int k = 0; k < fourier.size(); k++)
    {
        float amp = fourier[k].amp;
        float phase = fourier[k].phase;
        float freq = fourier[k].freq;
        target += vec_from_angle_and_mag(time.get_d() * freq + phase + angle, amp);
        // ellipse (p_target.x, p_target.y, amp*2, amp*2);
        if (k != 0 && amp > 1)
            state.ellipses.push_back({Point(center_x + p_target[0], center_y + p_target[1]),  amp});
        else if (amp <= 1)
            state.degenerate_ellipses.push_back(Point(center_x + p_target[0], center_y + p_target[1]));
        
        p_target = target;
        state.arm.push_back(Point(center_x + target[0], center_y + target[1]));
    }
    // return state;
}

void fourier_drawer(Mat& img, fourier_state& state)
{
    Mat overlay = Mat::zeros(img.size(), CV_8UC1);
    for (int i = 0; i < state.ellipses.size(); i++)
    {
        #if FAST
        ellipse(img, 
            state.ellipses[i].first, 
            Size(state.ellipses[i].second, state.ellipses[i].second), 0, 0, 360, Scalar(100), 1, LINE_AA, 0);
        #else
        overlay = Mat::zeros(img.size(), CV_8UC1);
        ellipse(overlay, 
            state.ellipses[i].first, 
            Size(state.ellipses[i].second, state.ellipses[i].second), 0, 0, 360, Scalar(50), 1, LINE_AA, 0);
        img+=overlay;
        #endif
    }        

    overlay = Mat::zeros(img.size(), CV_8UC1); 
    polylines(overlay, state.arm, false, Scalar(150), 1, LINE_AA);
    img+=overlay;

    overlay = Mat::zeros(img.size(), CV_8UC1);
    polylines(overlay, state.degenerate_ellipses, false, Scalar(100), 1, LINE_AA);
    img+=overlay;

}