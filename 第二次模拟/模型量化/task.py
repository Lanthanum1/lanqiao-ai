#quantize-start
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

def quantize_model(model_path, quantized_model_path):
    #TODO

    


def prediction_label(test_sentence, model_path):
    # TODO





def main():
    # 量化模型
    quantize_model('/home/project/model.h5', '/home/project/quantized_model.tflite')
    # 测试示例
    test_sentence = "一个 公益广告 ： 爸爸 得 了 老年痴呆  儿子 带 他 去 吃饭  盘子 里面 剩下 两个 饺子  爸爸 直接 用手 抓起 饺子 放进 了 口袋  儿子 愣住 了  爸爸 说  我 儿子 最爱 吃 这个 了  最后 广告 出字 ： 他 忘记 了 一切  但 从未 忘记 爱 你    「 转 」"
    print(prediction_label(test_sentence, '/home/project/quantized_model.tflite'))

if __name__ == "__main__":  
    main()
#quantize-end