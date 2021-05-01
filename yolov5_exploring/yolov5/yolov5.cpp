
#include "yolov5.h"
#include "logging.h"
static Logger gLogger;



CpuTimer::CpuTimer(){

}
void CpuTimer::Start(){
    start_t = std::chrono::steady_clock::now();

}
double CpuTimer::TimeSpent(){
    auto end_t = std::chrono::steady_clock::now();
    double t_ = ( end_t - start_t).count()/1e6;
    return t_;
}
std::chrono::_V2::steady_clock::time_point CpuTimer::GetTime(){
    auto t = std::chrono::steady_clock::now();
    return t;
}



static int get_width(int x, float gw, int divisor = 8) {
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0) {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) {
        return 1;
    } else {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * get_width(512, gw) * 2 * 2));
    for (int i = 0; i < get_width(512, gw) * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, get_width(512, gw) * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), get_width(512, gw), DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(get_width(512, gw));
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, get_width(256, gw) * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), get_width(256, gw), DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(get_width(256, gw));
    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    //yolo layer 1
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers,unsigned char* gpu_data, float* output, int batchSize, double &engine_time) {
    convertAndCopy(gpu_data, (float *)buffers[0], batchSize, stream);
    GpuTimer gpu_timer;
    gpu_timer.Start(stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    gpu_timer.Stop(stream);
    engine_time += gpu_timer.Elapsed();
    int* sorted_indices;
    HANDLE_ERROR(cudaMalloc((void**)&sorted_indices,batchSize*Yolo::MAX_OUTPUT_BBOX_COUNT*sizeof(int)));
    gpuKernel((float*)buffers[1],sorted_indices, batchSize, stream);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * Yolo::OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    HANDLE_ERROR(cudaFree(sorted_indices));
    cudaStreamSynchronize(stream);
}

bool parse_args(std::string net,float& gd, float& gw) {

    if (net == "s") {
        gd = 0.33;
        gw = 0.50;
    } else if (net == "m") {
        gd = 0.67;
        gw = 0.75;
    } else if (net == "l") {
        gd = 1.0;
        gw = 1.0;
    } else if (net == "x") {
        gd = 1.33;
        gw = 1.25;
    } else {
        return false;
    }
    return true;
}

YOLOv5Engine::YOLOv5Engine(std::string net,std::string wts_name, std::string engine_name, int batch_size) {
    engine_time=0;
    preprocess_time=0;
    post_time = 0;
    nF = 0;
    detection_time = 0;
    image2gpu_time = 0;
    t_copy = 0;
    batchSize = batch_size;
    t_resize = 0;
    std::cout<<" Optimizing engine for batch size = "<<batch_size<<std::endl;
    cudaSetDevice(DEVICE);
    float gd = 0.0f, gw = 0.0f;
    if (!parse_args(net,gd, gw)) {
        std::cerr << "arguments not right!" << std::endl;
        exit(10);
    }

    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(batchSize, &modelStream, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            exit(11);
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();

    }
    std::cout<<"Initialize engine .... "<<std::endl;
    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        exit(12);
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();


    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&gpu_data, batchSize * 3 * INPUT_H * INPUT_W * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * Yolo::OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    prob = (float*)malloc((batchSize * Yolo::OUTPUT_SIZE)*sizeof(float));
    // Run inference
    auto start = std::chrono::system_clock::now();
    double temp;
    doInference(*context, stream, buffers, gpu_data, prob, batchSize, temp);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(batchSize);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    std::cout<<"Engine is ready! "<<std::endl;

    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;


}


std::vector<std::vector<Yolo::Detection>> YOLOv5Engine::runInference(std::vector<cv::Mat> &frames,  unsigned char * gpu_data_d, int data_size) {
    GpuTimer gpu_timer;
    CpuTimer cpu_timer1, cpu_timer2,cpu_timer3;
    cpu_timer1.Start();
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cv::cuda::Stream opencv_stream = cv::cuda::StreamAccessor::wrapStream(stream);
    nF += 1;

    cpu_timer2.Start();

    int offset = 0;

    for (int b = 0; b < frames.size(); b++) {

        int frame_data_size = frames[b].size().width * frames[b].size().height*3*sizeof(unsigned char);
        cv::cuda::GpuMat gpu_img(frames[b].size(),CV_8UC3, &gpu_data_d[offset]);
        offset += frame_data_size;
        cv::cuda::GpuMat pr_gpu_img;
        if (gpu_img.empty()) continue;
        preprocess_img_cuda(gpu_img, pr_gpu_img, INPUT_W, INPUT_H, opencv_stream);

        //preprocess_img(pr_img, pr_img, INPUT_W, INPUT_H); // letterbox BGR to RGB

        if(!pr_gpu_img.isContinuous()){
            matCopy(pr_gpu_img.data, &gpu_data[b * 3 * INPUT_H * INPUT_W],pr_gpu_img.step,stream);
        }else{
            //CUDA_CHECK(cudaMemcpy((void*)&gpu_data[b * 3 * INPUT_H * INPUT_W], (void*)pr_img.data, 3 * INPUT_H * INPUT_W * sizeof(unsigned char), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync((void*)&gpu_data[b * 3 * INPUT_H * INPUT_W], (void*)pr_gpu_img.data, 3 * INPUT_H * INPUT_W * sizeof(unsigned char), cudaMemcpyDeviceToDevice,stream));
        }

        //memcpy((void*)&cpu_data[b * 3 * INPUT_H * INPUT_W],(void*)pr_img.data,3 * INPUT_H * INPUT_W*sizeof(unsigned char));



    }


    cudaDeviceSynchronize();
    preprocess_time += cpu_timer2.TimeSpent();

    // Run inference

    gpu_timer.Start(stream);
    doInference(*context, stream, buffers, gpu_data, prob, batchSize, engine_time);
    cudaDeviceSynchronize();
     gpu_timer.Stop(stream);
     cudaStreamDestroy(stream);

    model_time += gpu_timer.Elapsed();
    CpuTimer post_timer;
    post_timer.Start();
    std::vector<std::vector<Yolo::Detection>> batch_res(batchSize);

    for (int b = 0; b < frames.size(); b++) {
        auto& res = batch_res[b];
        nms(res, &prob[b * Yolo::OUTPUT_SIZE], Yolo::CONF_THRESH, Yolo::NMS_THRESH);
    }
    post_time += post_timer.TimeSpent();

    detection_time += cpu_timer1.TimeSpent();
    std::cout<<"Preprocess time = "<<preprocess_time/nF<<" ms for "<<frames.size()<<" images"<<std::endl;
    std::cout<<"Copy time = "<<t_copy/nF<<" ms for "<<frames.size()<<" images"<<std::endl;
    std::cout<<"resize time = "<<t_resize/nF<<" ms"<<std::endl;
    std::cout<<"Model time = "<<model_time/nF<<" ms"<<std::endl;
    std::cout<<"Engine time = "<<engine_time/nF<<" ms"<<std::endl;
    std::cout<<"cuda_nms+converting_data time = "<<(model_time-engine_time)/nF<<" ms"<<std::endl;
    std::cout<<"extract detections time = "<<post_time/nF<<" ms"<<std::endl;
    std::cout<<"Detection time = "<<detection_time/nF<<" ms"<<std::endl;

    return batch_res;
}



void YOLOv5Engine::destroyEngine(){
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    CUDA_CHECK(cudaFree(gpu_data));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

}
