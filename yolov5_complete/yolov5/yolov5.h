#ifndef _YOLO_ 
#define _YOLO_
#include<iostream>
#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda.hpp>



#include "common.hpp"
#include "utils.h"




#define USE_FP32  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define N_PPREPROCESSING_THREAD 1
#define NET 's'

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";




class YOLOv5Engine{
private:
        IExecutionContext* context;
        ICudaEngine* engine;
        IRuntime* runtime;
        int batchSize;
public:
        // prepare input data ---------------------------
        //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        //    data[i] = 1.0;
        unsigned char *gpu_data;
        float* prob;
        void* buffers[2];
        float* final_result;
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()

        YOLOv5Engine(std::string net, std::string wts_name, std::string engine_name, int batch_size);
        std::vector<std::vector<Yolo::Detection> > runInference(std::vector<cv::Mat> &frames, unsigned char *gpu_data_d, int data_size);
        void destroyEngine();
        double engine_time, model_time;
        double preprocess_time;
        double post_time;
        double detection_time;
        double image2gpu_time, t_copy, t_resize;
        int nF;
};


class CpuTimer{
public:
  std::chrono::_V2::steady_clock::time_point start_t;
  CpuTimer();
  void Start();
  double TimeSpent();
  std::chrono::_V2::steady_clock::time_point GetTime();

};



#endif
