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

struct fourier_state {
    vector<Point> arm;
    vector<Point> degenerate_ellipses;
    vector<pair<Point, double> > ellipses;
};

void dft(vector<Point> &, vector<epycicle> &, int);

size_t dft (vector<vector<Point> > &, vector<epycicle> &);

void sort_epycicles(vector<epycicle> &);

Vec2d vec_from_angle_and_mag(double, double);

void fourier_calculator(double, double, float, vector<epycicle> &, mpf_class, fourier_state&);

void fourier_drawer(Mat&, fourier_state&);

#endif