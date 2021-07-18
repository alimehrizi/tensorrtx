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
    std::string modelFilepath{"/home/altex/Mehrizi/Models/TODv1.0-640s/exp/weights/best.wts"};
    std::string imageFolderPath = "/home/altex/Mehrizi/Datasets/Traffic Object Detection-v1.0/cctv_car_all/test/images";//"/home/altex/test_images/images" ;//
    std::string labelFilepath{"/home/altex/fake.txt"};
    std::string saveResultPath = "/home/altex/Result/trt_nms";
    int batch_size=1;

    YOLOv5Engine model("s", modelFilepath, "deep.engine", batch_size);

    std::vector<cv::Mat> images;
     std::vector<std::string> names;
    int n_images=0;
    int data_size = 0;
    std::vector<cv::Mat> frames;
    for (auto & entry : boost::filesystem::directory_iterator(imageFolderPath)){
        std::string img_path = entry.path().string();
        auto img = cv::imread(img_path);
        if(img.empty())continue;


        frames.push_back(img);
         names.push_back(img_path);
         data_size += img.size().width*img.size().height*3*sizeof(uchar);
        if(frames.size()==batch_size){
             unsigned char *gpu_data;
             HANDLE_ERROR(cudaMalloc((void**)&gpu_data,data_size));
             int offset=0;
             for(int f=0;f<frames.size();f++){
                 int ds = frames[f].size().width*frames[f].size().height*3*sizeof(uchar);
                HANDLE_ERROR(cudaMemcpy((void*)&gpu_data[offset],(void*)frames[f].data,ds,cudaMemcpyHostToDevice));
                offset += ds;
             }
             auto detections = model.runInference(frames,gpu_data,data_size);
             HANDLE_ERROR(cudaFree(gpu_data));
             data_size = 0;
             for(int f=0;f<names.size();f++){
                 std::ofstream result_file;
                 std::string image_name = split(split(names[f],"/").back(),".jpg")[0];
                 std::string result_path = saveResultPath + "/"+image_name+".txt";
                 result_file.open(result_path,std::ios::out);

                 auto s = frames[f].size();
                 result_file<<s.width<<" "<<s.height<<" \n";
                 for(int i=0;i<detections[f].size();i++){
                     auto fbox = detections[f][i].bbox;
                     cv::Rect2f box(fbox[0], fbox[1], fbox[2], fbox[3]);
                     //cv::rectangle(img,box, cv::Scalar(255),2);
                     box.y = (box.y + box.height/2)/s.height;
                     box.x = (box.x + box.width/2)/s.width;
                     box.height /= s.height;
                     box.width /= s.width;
                     if(detections[f][i].conf>=0.3)
                     result_file<<box.x<<" "<<box.y<<" "<<box.width<<" "<<box.height<<" "<<detections[f][i].conf<<" "<<detections[f][i].class_id<<"\n";
                 }
             }
//         cv::imshow("frames",img);
//         cv::waitKey(0);
           frames.clear();
           names.clear();
           images.clear();
        }
    }


    return 0;
}
