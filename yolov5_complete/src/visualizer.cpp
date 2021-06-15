#include "visualizer.h"


Visualizer::Visualizer()
{

    Colors =
    {
        cv::Scalar(0,255,0), cv::Scalar(255,200,0), cv::Scalar(0,0,255), cv::Scalar(153,153,255)
        };


}


void Visualizer::drawBboxes(cv::Mat &image, std::vector<Detection> detections,bool show_labels, int font, float font_size, int thickness){

    int w = image.size().width;
    int h = image.size().height;
    cv::Rect2f frame_box(0,0,w-1,h-1);
    int baseLine=0;
    for(int i=detections.size()-1;i>=0;i--){
        auto box = detections[i].bbox & frame_box;
        auto cls = detections[i].label;
        auto score = detections[i].score;
        auto color = Colors[cls];
        cv::rectangle(image,box,color,thickness);
        int baseline=0;
        if(show_labels){
            std::string cls_string = std::to_string(cls);
            std::string text = cls_string + ","+std::to_string(score).substr(0,3);
            auto text_size = cv::getTextSize(text,font,font_size,thickness,&baseline);
            auto pt1 = cv::Point(box.x,box.y);
            int w_ = std::min(text_size.width,int(box.width));
            cv::rectangle(image,cv::Rect2f(box.x, box.y-text_size.height,w_,text_size.height),color, -1);
            cv::putText(image,text,pt1,font,font_size,cv::Scalar(0),thickness);

        }

    }

    return;
}
