#include <iostream>
#include"header.h"
#include"visualizer.h"
#include"yolov5.h"


using namespace std;



std::vector<std::string> split(std::string s,std::string delimiter = ">="){

    std::vector<std::string> results;

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        results.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    results.push_back(s);
    return results;
}


int main(int argc, const char* argv[])
{



    cout << "opencv version: "<< CV_VERSION<<endl;
    std::string modelFilepath{"/media/altex/XcDrive/MScProject/Codes/Models/yolov5s.wts"};
    std::string imageFolderPath =  "/home/altex/test_images/images";
    std::string labelFilepath{"/home/altex/fake.txt"};
    std::string saveResultPath = "/home/altex/test_images/tensorrt-result";


    YOLOv5Engine model("s", modelFilepath, "deep.engine", 1);

    std::vector<cv::Mat> images;
     std::vector<std::string> names;
    int n_images=0;
    for (auto & entry : boost::filesystem::directory_iterator(imageFolderPath)){
        std::string img_path = entry.path().string();
        auto img = cv::imread(img_path);
        if(img.empty())continue;
        std::vector<cv::Mat> frames;
        frames.push_back(img);
         names.push_back(img_path);
         images.push_back(img);
         std::cout<<img_path<<std::endl;
         int data_size = img.size().width*img.size().height*3*sizeof(uchar);
         unsigned char *gpu_data;
         HANDLE_ERROR(cudaMalloc((void**)&gpu_data,data_size));
         HANDLE_ERROR(cudaMemcpy((void*)gpu_data,(void*)frames.back().data,data_size,cudaMemcpyHostToDevice));
         auto detections = model.runInference(frames,gpu_data,data_size);
         HANDLE_ERROR(cudaFree(gpu_data));

         std::ofstream result_file;
         std::string image_name = split(split(img_path,"/").back(),".")[0];
         std::string result_path = saveResultPath + "/"+image_name+".txt";
         result_file.open(result_path,std::ios::out);
         auto s = img.size();
         for(int i=0;i<detections[0].size();i++){
             cv::Rect2f box = get_rect(img, detections[0][i].bbox);
             //cv::rectangle(img,box, cv::Scalar(255),2);
             box.y = (box.y + box.height/2)/s.height;
             box.x = (box.x + box.width/2)/s.width;
             box.height /= s.height;
             box.width /= s.width;
             result_file<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<" "<<detections[0][i].conf<<" "<<detections[0][i].class_id<<"\n";
         }
//         cv::imshow("frames",img);
//         cv::waitKey(0);
    }


    return 0;
}
