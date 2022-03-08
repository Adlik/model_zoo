# Model Optimizer

## Content

- Object detection

## Object detection

|  Model  |    Compression strategy     |                                         Introduction                                         |                                                                                   Pretrain weight                                                                                    |
| ------- | --------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| yolov5s | distillation                | Based on the method of "Object Detection at 200 Frames Per Second"                           | [pth](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.pt) &#124; [onnx](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-39.3.onnx)            |
| yolov5s | quantization                | INT8 quantization uses OpenVINO's POT tool                                                   | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-int8-mixed.bin)                 |
| yolov5s | distillation + quantization | In order not to lose accuracy, distillation and INT8 quantization can be used simultaneously | [xml](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.xml) &#124; [bin](https://adlik-yolov5.oss-cn-beijing.aliyuncs.com/yolov5s-distill-int8-mixed.bin) |

Table Notes:

* Refer to [Adlik YOLOv5](https://github.com/Adlik/yolov5) for details.