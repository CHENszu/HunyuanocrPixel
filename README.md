# HunyuanocrPixel介绍
由于项目需要，我们需要以JSON格式返回OCR识别的**文字**以及对应边框的**像素坐标**，我们尝试了[PaddleOCR-VL](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL) [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) [DeepseekOCR](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR) mineru  monkeyocr dots.ocr nanonet.  
## 1 遇到的困难
由于我们的数据较为复杂：1.PDF里面的图片一些是倾斜的；2.同一表格里面的文本会换行：  
<img width="65" height="127" alt="图片" src="https://github.com/user-attachments/assets/802dd289-a13d-4f65-9ae1-fde951f696d1" />  
经过多个模型的测试，我们发现PaddleOCR-VL的识别效果是最好的，并且对于内存的要求是最低的，如果您只需要文本信息，可以通过docker一行命令进行[部署](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#22-python)：  
  
sudo docker run     -it     --rm     --gpus all     --network host     ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest     paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm  
  
但是这个大模型并不会返回表格内部文字的像素坐标，而是将表格作为一个整体进行处理：  
<img width="791" height="246" alt="图片" src="https://github.com/user-attachments/assets/771dcc5e-6c7f-469a-af34-59a6e1aa3021" />  
  
HunyuanOCR识别中基本上都是**按行**进行处理（经过测试HunyuanOCR效果最优，这里就不介绍后续的模型了），但是那样就出现一个问题，同一个表格内的文字被识别为两部分：  
<img width="143" height="73" alt="图片" src="https://github.com/user-attachments/assets/eec89a45-368a-4f10-80a8-fb7d5f6eb814" />  
我们发现HunyuanOCR不仅能够通过自定义提示词，而且像素坐标是按照一个个文本进行返回，并且如果图片是倾斜的，其也能较为优秀的识别出来。  
## 2 针对于同一表格内 文字换行 的解决方案
我们在这里采用[IoU](https://cloud.tencent.com/developer/article/1446244)合并的方法进行，如果文本的边框在表格边框内占据80%以上的面积，即可认为该文字属于这个表格。  
那么接下来我们的任务就是如何去解析图片中的表格位置，我们在这里推荐通义实验室的[Cycle-CenterNet](https://www.modelscope.cn/models/iic/cv_dla34_table-structure-recognition_cycle-centernet)有线表格识别和[LORE](https://www.modelscope.cn/models/iic/cv_resnet-transformer_table-structure-recognition_lore)无限表格识别，当然我们也可以采用OpenCV来进行框线检测，不过针对不同的数据需要调整超参，不具备普适性。  
在这里给大家展示一下区别：
![Uploading 图片.png…]()

<img width="437" height="493" alt="图片" src="https://github.com/user-attachments/assets/16989e9b-d9a6-4f83-bb2d-c1154ac9571d" />
我们将这个集成项目写了一个前端，大家在部署好HunyuanOCR和Cycle-CenterNet以及LORE就可以解决表格内文本换行的问题，同时带有文字和像素坐标的JSON：  
<img width="1361" height="895" alt="图片" src="https://github.com/user-attachments/assets/9d823b48-c8a9-4a3e-b59a-194ae5c7ff2b" />  
不过我们在测试过程中也发现了HunyuanOCR[坐标异常](https://github.com/Tencent-Hunyuan/HunyuanOCR/issues/65)的情况，不过对于一般的数据都还是没问题的。
## 3 代码重点介绍
+ HunyuanOCR在识别的时候会对图片进行变换，因此我们需要[逆变换](https://github.com/Tencent-Hunyuan/HunyuanOCR/issues/40)才能得到正确的坐标；
+ 配置好环境后运行start.sh系统就正常运行了！
