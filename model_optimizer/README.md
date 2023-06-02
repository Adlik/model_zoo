# Model Optimizer

## Content

- Foundation model
- Object detection
- Classification
- Instance segmentation

## Classification

| Model         | strategy           | Introduction                                                 | Pretrain weight                                              | accuracy |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| resnet18      | normal training    |                    adlik resnet18  trained with imagenet-1k                                          | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_18_imagenet_1000.h5) | 66.834%  |
| resnet50      | normal training    |                adlik resnet50  trained with imagenet-1k                                              | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_50_imagenet_1000.h5) | 76.0%    |
| resnet101     | normal training    |                      adlik resnet101  trained with imagenet-1k                                        | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_101_imagenet_1001.h5) | 77.12%   |
| resnet34      | normal training    |                  adlik resnet34  trained with imagenet-1k                                            | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_34_imagenet_1000.h5) | 70.346%  |
| resnet50-distill  |  distillation  |   using resnet101 as teacher model to distill resnet50 |  [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet_50_distill.h5)     | 77.14% |
| resnet50-slim | pruning+distillation | pruning 50% channels of resnet50 model and using senet154 and resnet152b as teacher models to ensemble distill pruned model | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50_slim.h5) | 76.376%  |
| resnet50-pruned-0.25 |  pruning+distillation | pruning 25% channels of resnet50 and using senet154 and resnet152b as teacher models to ensemble distill pruned model   |  [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50_pruned_0.25.h5)  |  78.79%   |
| resnet50-pruned-0.375 |  pruning+distillation  | pruning 37.5% channels of resnet50 and using senet154 and resnet152b as teacher models to ensemble distill pruned model  | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50_pruned_0.375.h5)  | 78.07% |
| mobilenet v1  | normal training    |                      adlik mobilenet-v1  trained with imagenet-1k                                        | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/mobilenet_v1_imagenet_1000.h5) | 70.63%   |
| mobilenet v2  | normal training    |                             adlik mobilenet-v2  trained with imagenet-1k                                 | [h5](https://adlik-models.oss-cn-beijing.aliyuncs.com/mobilenet_v2_imagenet_1001.h5) | 72.396%  |

### inference

When using the above models for inference, please note that both mobilenetV1 and resnet101 are trained with outputs with 1001 classification and 0-index in outputs is background. Besides, do not normalize input images when inferring the above two models. In model optimizer, the related codes are [here](https://github.com/Adlik/model_optimizer/blob/master/src/model_optimizer/utils/imagenet_preprocessing.py#:~:text=return-,image_normalization(img),-Footer).

## Foundation model

| Model         | Training strategy           | Introduction                                                 | Pretrain weight                                              | top1 acc |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| ViT-SG | Pretrain and finetune                 | To be a teacher model in the downstream classification with distillation                         | [oneflow](https://adlik-models.oss-cn-beijing.aliyuncs.com/Adlik_ViT_SG.zip)           |  86.046   |

## Classification with distillation

| Model         | Distillation strategy           | Introduction                                                 | Pretrain weight                                              | top1 acc |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| ResNet50-81.33 | KD & DKD                 | First by KD method and then by DKD method with two teachers                         | [oneflow](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50_dkd.zip)           |  81.33   |
| ResNet50-80.754 |  KD                    |  Use resnet50d in Timm as the teacher to distill the resnet50    |    [tar](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50best-base.pth.tar)    |    80.754     |

## Classification with quantization

| Model         | Quantization strategy         | Introduction                                                  | Pretrain weight                                              | top1 acc  |
|  --------------- | ----------------------------- | -------------------------- | -------------------------- | ------------------------------- |
| ResNet50-W3A4-77.34  |    LSQ + distillation      |      Quantize the model which weight is 3bit and activation is 4bit using LSQ algorithm and distill using resnet50d in Timm.  We use our implementation of the python simulator to evaluate the accuracy of low-bit models    | [jit](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50best_qat_quant_W3A4.jit)        |      77.34       |
| ResNet50-W3A4-qat-77.34 |  LSQ + distillation     |    The qat model without using convert_fx function, which can be used to evaluate   |     [tar](https://adlik-models.oss-cn-beijing.aliyuncs.com/resnet50best-student.pth.tar)   |  77.34 |

## Object detection

| Model         | Compression strategy           | Introduction                                                 | Pretrain weight                                              | box AP |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| yolov5s | distillation                | Based on the method of "Object Detection at 200 Frames Per Second"                           | [pth](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.pt) &#124; [onnx](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.onnx)            |  -   |
| yolov5s | quantization                | INT8 quantization uses OpenVINO's POT tool                                                   | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.bin)                 |  -   |
| yolov5s | distillation + quantization | In order not to lose accuracy, distillation and INT8 quantization can be used simultaneously | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.bin) |  -   |
| yolov5m | normal training |Use [the official website code](https://github.com/ultralytics/yolov5/blob/master/export.py) to convert the model of pytorch framework to the  model of tensorrt_plan runtime (GPU:Tesla T4) <br /> Use [Adlik tensorrt runtime](https://github.com/Adlik/Adlik#:~:text=registry.cn-beijing.aliyuncs.com/adlik/serving-tensorrt%3Av0.4.0_trt7.2.1.6_cuda11.0) deploy this tensorrt_plan model| [pt](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt) &#124; [tensorrt_plan](https://adlik-models.oss-cn-beijing.aliyuncs.com/ultralytics-yolov5.zip) |  -   |
| Faster R-CNN deit-small | distillation                | Based on the method of "VITKD: Practical Guidelines For ViT Feature Knowledge Distillation" and "Focal and Global Knowledge Distillation for Detectors"                           | [pth](https://adlik-models.oss-cn-beijing.aliyuncs.com/best_bbox_mAP_epoch_24.pth)          |  47.6mAP   |

## Instance segmentation

| Model         | Compression strategy           | Introduction                                                 | Pretrain weight                                              | mask AP |
| ------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| Mask R-CNN deit-small | distillation                | Based on the method of "VITKD: Practical Guidelines For ViT Feature Knowledge Distillation" and "Focal and Global Knowledge Distillation for Detectors"                           | [pth](https://adlik-models.oss-cn-beijing.aliyuncs.com/best_bbox_mAP_epoch_23.pth)             | 42.4mAP |

Table Notes:

- Refer to [model optimizer](https://github.com/Adlik/model_optimizer) for more information.
- Refer to [model_optimizer_tf](https://github.com/Adlik/model_optimizer_tf) for details.
- Refer to [Adlik YOLOv5](https://github.com/Adlik/yolov5) for details.
