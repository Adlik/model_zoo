# Model Optimizer

## Content

- Object detection
- Classification

## Classification

| Model         | strategy           | Introduction                                                 | Pretrain weight                                              | accuracy |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| resnet18      | normal training    |                    adlik resnet18  trained with imagenet-1k                                          | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_18_imagenet_1000.h5) | 66.834%  |
| resnet50      | normal training    |                adlik resnet50  trained with imagenet-1k                                              | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_50_imagenet_1000.h5) | 76.0%    |
| resnet101     | normal training    |                      adlik resnet101  trained with imagenet-1k                                        | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_101_imagenet_1000.h5) | 77.12%   |
| resnet34      | normal training    |                  adlik resnet34  trained with imagenet-1k                                            | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_34_imagenet_1000.h5) | 70.346%  |
| resnet50-slim | prune+distillation | pruning resnet50 model and using senet154 and resnet152b as teacher models to ensemble disitll pruned model | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50_slim.h5) | 76.458%  |
| mobilenet v1  | normal training    |                      adlik mobilenet-v1  trained with imagenet-1k                                        | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/mobilenet_v1_imagenet_1000.h5) | 70.63%   |
| mobilenet v2  | normal training    |                             adlik mobilenet-v2  trained with imagenet-1k                                 | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/mobilenet_v2_imagenet_1000.h5) | 72.396%  |

## Object detection

|  Model  |    Compression strategy     |                                         Introduction                                         |                                                                                   Pretrain weight                                                                                    |
| ------- | --------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| yolov5s | distillation                | Based on the method of "Object Detection at 200 Frames Per Second"                           | [pth](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.pt) &#124; [onnx](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.onnx)            |
| yolov5s | quantization                | INT8 quantization uses OpenVINO's POT tool                                                   | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.bin)                 |
| yolov5s | distillation + quantization | In order not to lose accuracy, distillation and INT8 quantization can be used simultaneously | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.bin) |

Table Notes:

- Refer to [model optimizer](https://github.com/Adlik/model_optimizer) for more infomation.
- Refer to [Adlik YOLOv5](https://github.com/Adlik/yolov5) for details.
