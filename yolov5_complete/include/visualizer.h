#ifndef VISUALIZER_H
#define VISUALIZER_H
#include "header.h"

class Visualizer
{
public:
    std::vector <cv::Scalar> Colors;
    Visualizer();
    void drawBboxes(cv::Mat &image, std::vector<Detection> detections,bool show_labels, int font, float font_size, int thickness);

};

#endif // VISUALIZER_H


