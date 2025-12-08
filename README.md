# HunyuanocrPixel介绍
由于项目需要，我们需要以JSON格式返回OCR识别的**文字**以及对应边框的**像素坐标**，我们尝试了[PaddleOCR-VL](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL) [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) [DeepseekOCR](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR) mineru  monkeyocr dots.ocr nanonet.  
由于我们的数据较为复杂：1.PDF里面的图片一些是歪的；2.表格里面的文本会换行：  
<img width="65" height="127" alt="图片" src="https://github.com/user-attachments/assets/802dd289-a13d-4f65-9ae1-fde951f696d1" />  
经过多个模型的测试，我们发现PaddleOCR-VL的识别效果是最好的，并且对于内存的要求是最低的，我们可以通过docker一行命令进行[部署](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#22-python)：  
sudo docker run     -it     --rm     --gpus all     --network host     ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest     paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm  
但是这个大模型并不会返回表格内部文字的像素坐标，而是将表格作为一个整体进行处理：  
<img width="791" height="246" alt="图片" src="https://github.com/user-attachments/assets/771dcc5e-6c7f-469a-af34-59a6e1aa3021" />  
  
后续模型在OCR识别中基本上都是**按行**进行处理（HunyuanOCR效果最优），但是那样就出现一个问题，同一个表格内的文字被识别为两部分：  
<img width="143" height="73" alt="图片" src="https://github.com/user-attachments/assets/eec89a45-368a-4f10-80a8-fb7d5f6eb814" />  
我们发现HunyuanOCR不仅能够通过自定义提示词，而且像素坐标是按照一个个文本进行返回，并且如果图片是倾斜的，其也能较为优秀的识别出来。
## 针对于同一表格内 文字换行 的解决方案
我们在这里采用[IoU](https://cloud.tencent.com/developer/article/1446244)合并的方法进行，如果文本的边框在表格边框内占据80%以上的面积，即可认为该文字属于这个表格。  
那么接下来我们的任务就是如何去解析图片中的表格位置，我们在这里推荐通义实验室的[Cycle-CenterNet](https://www.modelscope.cn/models/iic/cv_dla34_table-structure-recognition_cycle-centernet)有线表格识别和[LORE](https://www.modelscope.cn/models/iic/cv_resnet-transformer_table-structure-recognition_lore)无限表格识别，当然我们也可以采用OpenCV来进行框线检测，不过针对不同的数据需要调整超参，不具备普适性。  


