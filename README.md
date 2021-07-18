# TensorRTx

TensorRTx aims to implement popular deep learning networks with tensorrt network definition APIs. As we know, tensorrt has builtin parsers, including caffeparser, uffparser, onnxparser, etc. But when we use these parsers, we often run into some "unsupported operations or layers" problems, especially some state-of-the-art models are using new type of layers.

So why don't we just skip all parsers? We just use TensorRT network definition APIs to build the whole network, it's not so complicated.

I wrote this project to get familiar with tensorrt API, and also to share and learn from the community.

All the models are implemented in pytorch/mxnet/tensorflown first, and export a weights file xxx.wts, and then use tensorrt to load weights, define network and do inference. Some pytorch implementations can be found in my repo [Pytorchx](https://github.com/wang-xinyu/pytorchx), the remaining are from polular open-source implementations.


### yolov5 
* original repo implementation 

### yolov5-org
* original repo old implementation 

### yolov5_complete
* my implenation wiht FP16 and cuda NMS 


### yolov5_complete_int8
* my implenation wiht FP16/INT32 and cuda NMS 
* INT32 is a little faster than FP16 but has low accuracy  


### yolov5_exploring
* explore implementation
