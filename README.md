# HunyuanocrPixel介绍
由于项目需要，我们需要以JSON格式返回OCR识别的**文字**以及对应边框的**像素坐标**，我们尝试了[PaddleOCR-VL](https://www.modelscope.cn/models/PaddlePaddle/PaddleOCR-VL) [HunyuanOCR](https://github.com/Tencent-Hunyuan/HunyuanOCR) [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) [DeepseekOCR](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR) mineru  monkeyocr dots.ocr nanonet.  
由于我们的数据较为复杂：1.PDF里面的图片一些是歪的；2.表格里面的文本会换行：
<img width="65" height="127" alt="图片" src="https://github.com/user-attachments/assets/802dd289-a13d-4f65-9ae1-fde951f696d1" />

