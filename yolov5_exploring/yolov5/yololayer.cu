#include <assert.h>
#include <vector>
#include <iostream>
#include "yololayer.h"
#include "cuda_utils.h"

namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

using namespace Yolo;


#define NUM_DET 3*21*((Yolo::INPUT_H*Yolo::INPUT_W)/(32*32))

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize()
    {
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

    const char* YoloLayerPlugin::getPluginType() const
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void YoloLayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const
    {
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int noElements,
        const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float *res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char* data = (char *)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth 
            det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth  
            // v5: https://github.com/ultralytics/yolov5/issues/471
            det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
            det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize)
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize; 
            if (numElem < mThreadCount)
                mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
            CalDetection << < (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount >> >
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount, outputElem);
        }
    }


    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        int class_count = -1;
        int input_w = -1;
        int input_h = -1;
        int max_output_object_count = -1;
        std::vector<Yolo::YoloKernel> yolo_kernels(3);

        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "netdata") == 0) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                class_count = tmp[0];
                input_w = tmp[1];
                input_h = tmp[2];
                max_output_object_count = tmp[3];
            } else if (strstr(fields[i].name, "yolodata") != NULL) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                YoloKernel kernel;
                kernel.width = tmp[0];
                kernel.height = tmp[1];
                for (int j = 0; j < fields[i].length - 2; j++) {
                    kernel.anchors[j] = tmp[j + 2];
                }
                yolo_kernels[2 - (fields[i].name[8] - '1')] = kernel;
            }
        }
        assert(class_count && input_w && input_h && max_output_object_count);
        YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, yolo_kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}
__global__ void convert2float( unsigned char* char_data, float*data){
    int idx = blockIdx.x*3*INPUT_H*INPUT_W+blockIdx.y*3*INPUT_W;
    data[ blockIdx.x*3*INPUT_H*INPUT_W+blockIdx.y*INPUT_W+threadIdx.x] = (float)char_data[idx+3*threadIdx.x]/255.0;
    data[blockIdx.x*3*INPUT_H*INPUT_W+blockIdx.y*INPUT_W+threadIdx.x+INPUT_W*INPUT_H] = (float)char_data[idx+3*threadIdx.x+1]/255.0;
    data[blockIdx.x*3*INPUT_H*INPUT_W+blockIdx.y*INPUT_W+threadIdx.x+2*INPUT_W*INPUT_H] = (float)char_data[idx+3*threadIdx.x+2]/255.0;

}

void convertAndCopy(unsigned char* char_data, float*data, int batch_size, cudaStream_t &stream){
    dim3 dimGrid_0(batch_size,INPUT_H,1);
    dim3 dimBlock_0(INPUT_W,1,1);
    convert2float<<< dimGrid_0, dimBlock_0,0,stream>>>(char_data, data);

}


__global__ void copyGpuMat( unsigned char* src, unsigned char*dst, int step){
    int idx_src = blockIdx.y*step;
    int idx_dst = blockIdx.y*3*INPUT_W;

    dst[idx_dst+3*threadIdx.x] = src[idx_src+3*threadIdx.x];
    dst[idx_dst+3*threadIdx.x+1] = src[idx_src+3*threadIdx.x+1];
    dst[idx_dst+3*threadIdx.x+2] = src[idx_src+3*threadIdx.x+2];

}

void matCopy(unsigned char* src, unsigned char*dst, int step, cudaStream_t &stream){
    dim3 dimGrid_0(1,INPUT_H,1);
    dim3 dimBlock_0(INPUT_W,1,1);
    copyGpuMat<<< dimGrid_0, dimBlock_0,0,stream>>>(src, dst, step);

}




///////////////////////////NMS///////////////////////////////////

__device__ float calc_iou(float *lbox, float*rbox){
    float interBox[] = {
        max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}


__global__ void naive_nms_kernel(float*output, int* sorted_indices){
    int tid = threadIdx.x;
    int idx1 = sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+tid];
    bool valid=true;
    float *data_ptr = &output[blockIdx.y*Yolo::OUTPUT_SIZE];
    float *box1 = &data_ptr[1+idx1*Yolo::DET_SIZE];
    if(box1[4]<Yolo::CONF_THRESH){
        data_ptr[1+idx1*Yolo::DET_SIZE+4] = 0;
        return;
    }
    for(int i=0;i<Yolo::MAX_OUTPUT_BBOX_COUNT & i<tid;i++){
        int idx2 = sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+i];
        float *box2 = &data_ptr[1+idx2*Yolo::DET_SIZE];
        float iou = calc_iou( box1, box2);
        if(iou>Yolo::NMS_THRESH){
            valid = false;
            break;
        }

    }
    if(!valid)
        data_ptr[1+idx1*Yolo::DET_SIZE+4] = 0;

}


__global__ void sort_kernel(float* detections, int* sorted_indices){
    int idx = threadIdx.x;
    sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+idx] = idx;
    sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+idx+blockDim.x] = idx+blockDim.x;
    __syncthreads();
    float *data_ptr = &detections[blockIdx.y*Yolo::OUTPUT_SIZE];

    // sort indices:
    for(int k=0;k<Yolo::MAX_OUTPUT_BBOX_COUNT;k++){
        int m = k%2;
        if((2*idx+m+1)<Yolo::MAX_OUTPUT_BBOX_COUNT){
            int idx1 = sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+2*idx+m];
            int idx2 = sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+2*idx+m+1];
            if(data_ptr[1+idx1*Yolo::DET_SIZE+4]<data_ptr[1+idx2*Yolo::DET_SIZE+4]){
                sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+2*idx+m] = idx2;
                sorted_indices[blockIdx.y*Yolo::MAX_OUTPUT_BBOX_COUNT+2*idx+m+1] = idx1;
            }
        }
        __syncthreads();

    }

}


__global__ void zero_detections(float* detections){
    int idx = blockIdx.y*blockDim.x*DET_SIZE+threadIdx.x*DET_SIZE;
    for(int i=0;i<DET_SIZE;i++){
        detections[idx+i] = 0;
    }

}


void gpuKernel(float *output,int*sorted_indices, int batch_size, cudaStream_t &stream){

    int batch_per_stream = batch_size;
    int batch_i = 0;

//    dim3 dimGrid_0(1,batch_per_stream,1);
//    dim3 dimBlock_0(Yolo::MAX_NMS_BBOX_COUNT,1,1);
//    zero_detections<<< dimGrid_0, dimBlock_0,0,stream>>>(result);

    dim3 dimGrid_1(1,batch_per_stream,1);
    dim3 dimBlock_1(Yolo::MAX_OUTPUT_BBOX_COUNT/2,1,1);
    sort_kernel<<< dimGrid_1, dimBlock_1, 0,stream>>>(output, sorted_indices);

    dim3 dimGrid_2(1,batch_per_stream,1);
    dim3 dimBlock_2(Yolo::MAX_OUTPUT_BBOX_COUNT,1,1);
    naive_nms_kernel<<< dimGrid_2, dimBlock_2, 0,stream>>>(output, sorted_indices);

    return;
}

