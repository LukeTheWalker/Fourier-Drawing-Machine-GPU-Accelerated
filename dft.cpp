#ifndef DFT
#define DFT

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <gmpxx.h>
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

typedef complex<double> cd;

struct epycicle
{
    double amp;
    double phase;
    double freq;
    epycicle(double amp, double phase, double freq) : amp(amp), phase(phase), freq(freq) {}
};

void dft(vector<Point> &sig, vector<epycicle> &X)
{
    int N = sig.size();
    for (int k = 0; k < N; k++)
    {
        cd Xk(0.0, 0.0);
        for (int n = 0; n < N; n++)
        {
            double phi = (2.0 * double(k) * M_PI * double(n)) / double(N);
            cd c(cos(phi), -sin(phi));
            // Xk.add( PVector.fromAngle(sig.get(n).heading()+c.heading()).setMag(sig.get(n).mag()*c.mag())); //alternative multiplication beetween complex numbers
            // Xk.add(mult(sig.get(n),c));
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

void dft (vector<vector<Point> > &sig, vector<epycicle> &X){
    printf("Starting DFT\n");
    vector<Point> linearized_sig;
    for (vector<Point> &s : sig)
        linearized_sig.insert(linearized_sig.end(), s.begin(), s.end());

    printf("Running DFT on %lu points\n", linearized_sig.size());
    
    dft(linearized_sig, X);

    printf("DFT done\n");

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
Point fourier_drawer(Mat& img, double x, double y, float angle, vector<epycicle> &fourier, mpf_class time)
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
        if (k != 0){
            // Mat overlay;
            // img.copyTo(overlay);
            ellipse(img, 
                Point(center_x + p_target[0], center_y + p_target[1]), 
                Size(amp, amp), 0, 0, 360, Scalar(50, 50, 50), 1, LINE_AA);
            // addWeighted(overlay, 0, img, .5, 0, img);
            line(img, 
                Point(center_x + p_target[0], center_y + p_target[1]),
                Point(center_x + target[0]  , center_y + target[1]  ), 
                Scalar(100, 100, 100), 1, LINE_AA);
        }
        // line ( Point(p_target), Point(target));
        p_target = target;
    }
    return Point(p_target) + Point(center_x, center_y);
}

#endif

