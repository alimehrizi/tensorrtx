#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include<opencv2/cudaarithm.hpp>
#include "cuda_runtime.h"

#include<chrono>

static inline void preprocess_img(cv::Mat& src, cv::Mat &dst, float out_w, float out_h) {
    float in_h = src.size().height;
    float in_w = src.size().width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return;
}

//static inline void preprocess_img_cuda(cv::cuda::GpuMat& src, cv::cuda::GpuMat &dst, float out_w, float out_h, cv::cuda::Stream &stream) {
//    float in_h = src.size().height;
//    float in_w = src.size().width;

//    float scale = std::min(out_w / in_w, out_h / in_h);

//    int mid_h = static_cast<int>(in_h * scale);
//    int mid_w = static_cast<int>(in_w * scale);
//    cv::cuda::resize(src, dst, cv::Size(mid_w, mid_h),0,0,cv::INTER_LINEAR,stream);
//    int top = (static_cast<int>(out_h) - mid_h) / 2;
//    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
//    int left = (static_cast<int>(out_w)- mid_w) / 2;
//    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

//    cv::cuda::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0),stream);

//    return;
//}



static inline std::vector<float> preprocess_img_cuda(cv::cuda::GpuMat& src, cv::cuda::GpuMat &dst, float out_w, float out_h, cv::cuda::Stream &stream) {
    auto h0 = static_cast<float>(src.rows);
    auto w0 = static_cast<float>(src.cols);

    float r = std::min(out_w / w0, out_h / h0);
    auto interp = cv::INTER_AREA;
    if(r>1)
        interp = cv::INTER_LINEAR;
    int h = r*h0;
    int w = r*w0;
    //std::cout<<"step before resize "<<dst.step<<std::endl;

    cv::cuda::resize(src, dst, cv::Size(w, h),0,0,interp,stream);

    //std::cout<<"step after resize "<<dst.step<<std::endl;
    int new_unpad_w = w;
    int new_unpad_h = h;
    float dh = out_h - new_unpad_h;
    float dw = out_w - new_unpad_w;
    dw /= 2;
    dh /= 2;

    int top = round(dh-0.1);
    int bot = round(dh+0.1);
    int left = round(dw-0.1);
    int right = round(dw+0.1);


    cv::cuda::copyMakeBorder(dst, dst, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114),stream);
    //std::cout<<"step after border "<<dst.step<<std::endl;
    std::vector<float> pad_info{h0,w0, float(h/h0), float(w/w0), dw, dh};
    return pad_info;
}




static inline std::vector<float> preprocess_img_cpu(const cv::Mat& src, cv::Mat &dst, float out_w, float out_h) {
    auto h0 = static_cast<float>(src.rows);
    auto w0 = static_cast<float>(src.cols);


    float r = std::min(out_w / w0, out_h / h0);
    auto interp = cv::INTER_AREA;
    if(r>1)
        interp = cv::INTER_LINEAR;
    int h = r*h0;
    int w = r*w0;
    if(r>1 | r<1)
        cv::resize(src, dst, cv::Size(w, h),0,0,interp);
    else
        dst = src.clone();

    int new_unpad_w = w;
    int new_unpad_h = h;
    float dh = out_h - new_unpad_h;
    float dw = out_w - new_unpad_w;
    dw /= 2;
    dh /= 2;

    int top = round(dh-0.1);
    int bot = round(dh+0.1);
    int left = round(dw-0.1);
    int right = round(dw+0.1);

    cv::copyMakeBorder(dst, dst, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{h0,w0, float(h/h0), float(w/w0), dw, dh};
    return pad_info;
}




static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

#endif  // TRTX_YOLOV5_UTILS_H_

