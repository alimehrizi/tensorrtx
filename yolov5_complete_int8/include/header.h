#ifndef HEADER_H
#define HEADER_H

#ifndef DEBUG
#define DEBUG
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <map>
#include<string>
#include <ctime>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>

#define AVG_L 0.3*36
#define AVG_BOX_L 200

#define noop ((void)0)

enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};


typedef struct{
    float score=0.;
    cv::Rect2f bbox;
    int label=-1;
    float iou=-1.;
} Detection;
#endif // HEADER_H
