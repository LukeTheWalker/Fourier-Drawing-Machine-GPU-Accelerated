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
    vector<vector<Point> >non_zero;
    for (int k = 0; k < fourier.size(); k++)
    {
        float amp = fourier[k].amp;
        float phase = fourier[k].phase;
        float freq = fourier[k].freq;
        target += vec_from_angle_and_mag(time.get_d() * freq + phase + angle, amp);
        // ellipse (p_target.x, p_target.y, amp*2, amp*2);
        if (k != 0){
            vector<Point> non_zero_tmp;
            // img.copyTo(overlay);
            ellipse2Poly(Point(center_x + p_target[0], center_y + p_target[1]), Size(amp, amp), 0, 0, 360, 1, non_zero_tmp);
            // addWeighted(overlay, 0, img, .5, 0, img);
            // ellipse(img, 
            //     Point(center_x + p_target[0], center_y + p_target[1]), 
            //     Size(amp, amp), 0, 0, 360, Scalar(50), 1, LINE_AA);
            // addWeighted(overlay, 0, img, .5, 0, img);
            line(img, 
                Point(center_x + p_target[0], center_y + p_target[1]),
                Point(center_x + target[0]  , center_y + target[1]  ), 
                Scalar(150), 1, LINE_AA);
            non_zero.push_back(non_zero_tmp);
        }
        // line ( Point(p_target), Point(target));
        p_target = target;
    }
    
    unordered_map<int, int> occurences;
    // for (vector<Point> &v : non_zero)
    //     for (Point &p : v)
    //         occurences[p.x + p.y * img.cols]++;


    // for (auto &p: non_zero){
    //     for (int i = 1; i < p.size(); i++){
    //         LineIterator it (img, p[i-1], p[i], 8);
    //         for (int j = 0; j < it.count; j++, ++it){
    //             Point pt = it.pos();
    //             int occ = occurences[pt.x + pt.y * img.cols];
    //             img.at<uchar>(pt) += 50 * (occ + !(occ));
    //         }
    //     }
    // }
    for (auto &p: non_zero){
        for (int i = 1; i < p.size(); i++){
            LineIterator it (img, p[i-1], p[i], 8);
            for (int j = 0; j < it.count - 1; j++, ++it){
                Point pt = it.pos();
                occurences[pt.x + pt.y * img.cols]++;
            }
        }
    }

    // for each occurence, draw a point
    for (auto &p : occurences){
        // Point pnt(p.first % img.cols, p.first / img.cols);
        // circle(img, pnt, 1, Scalar(p.second*50), -1, LINE_AA);
        // if (p.second > 1)
        img.at<uchar>(p.first / img.cols, p.first % img.cols) += p.second*50;    
    }
    return Point(p_target) + Point(center_x, center_y);
}
