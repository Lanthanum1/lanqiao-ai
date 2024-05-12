# quantize-start
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


def quantize_model(model_path, quantized_model_path):
    # 整个函数的流程是：加载模型 -> 创建转换器 -> 配置转换器 -> 添加代表性数据集 -> 转换模型 -> 保存量化模型。
    # TODO
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # 这表示模型应该使用 TensorFlow Lite 的内置操作集。这些操作是专门为 TensorFlow Lite 优化的，可以在不支持完整 TensorFlow 的设备上运行，通常包括基本的数学运算、激活函数等。使用这个操作集，你可以确保模型能在许多低功耗设备上运行，因为它们通常只支持这些内置操作。
        tf.lite.OpsSet.SELECT_TF_OPS,
        #  这表示模型还可以使用一组特定的 TensorFlow 操作，这些操作通常不在 TensorFlow Lite 的基础操作集中。选择这些操作集允许将某些原生 TensorFlow 操作转换为可以在 TFLite 中运行的形式，从而支持更复杂的模型。但是，这通常意味着需要在目标设备上支持更多的库或硬件加速器。
    ]
    
    converter.target_spec.supported_types = [tf.float16] # 参数量化为 16 位浮点数。
    
    # 添加代表性数据集，为了进行动态量子化，转换器需要一个代表性数据集来模拟模型在实际运行时可能遇到的各种输入。
    representative_dataset = []
    
    # 这里生成了100个随机的二进制向量（0或1），形状为(1, 100)，并将其添加到列表representative_dataset中。这确保转换器能够观察到不同类型的输入，从而生成更准确的量化模型。
    for i in range(100):
        input_data = np.random.randint(0, 2, (1, 100)).astype(np.float32)
        # 生成一个形状为 (1, 100) 的数组，数组中的每个元素都是从0到1（不包括2）之间随机选择的整数。由于范围是0和1，实际上生成的是全为0或1的二进制数据。然后，通过 .astype(np.float32) 将这些整数转换为32位浮点数。这种类型的输入数据可能是为了模拟二值化的输入特征，比如某些机器学习任务中处理的图像像素值（黑或白）或简单的分类特征。  
        representative_dataset.append(input_data)
    converter.representative_dataset = representative_dataset
    quantized_tflite_model = converter.convert()

    open(quantized_model_path, "wb").write(quantized_tflite_model)


def prediction_label(test_sentence, model_path):
    # 整个函数的流程是：加载词汇索引 -> 初始化Tokenizer -> 准备测试数据 -> 初始化和运行TensorFlow Lite模型 -> 处理预测结果 -> 返回预测标签。
    # TODO
    with open("word_index.json", "r") as f:
        word_index = json.load(f)
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index 
    # Tokenizer对象用于将文本转换为数字序列，这是模型输入所必需的。
    
    test_sentence = [test_sentence]
    # 将单个测试句子放入一个列表中，因为texts_to_sequences方法期望一个字符串列表。
    
    test_seq = tokenizer.texts_to_sequences(test_sentence)
    # 测试句子转换为数字序列。
    
    test_seq = pad_sequences(test_seq, maxlen=100)
    # 模型预测前需将句子处理成长度 100 的序列。（根据题目的提示）

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    # 准备模型的输入和输出张量。

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], test_seq.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    prediction_label = (prediction > 0.5).astype(np.int_)
    # 预测结果通常是一个概率值，这里假设大于0.5表示正类，小于等于0.5表示负类。将预测结果转换为0或1的整数形式。

    return prediction_label[0][0]
    # 由于我们只有一个测试样本，返回结果是prediction_label二维数组的第一个元素的第一个元素，即最终的预测标签。


def main():
    # 量化模型
    quantize_model("./model.h5", "./quantized_model.tflite")
    # 测试示例
    test_sentence = "一个 公益广告 ： 爸爸 得 了 老年痴呆  儿子 带 他 去 吃饭  盘子 里面 剩下 两个 饺子  爸爸 直接 用手 抓起 饺子 放进 了 口袋  儿子 愣住 了  爸爸 说  我 儿子 最爱 吃 这个 了  最后 广告 出字 ： 他 忘记 了 一切  但 从未 忘记 爱 你    「 转 」"
    print(prediction_label(test_sentence, "./quantized_model.tflite"))


if __name__ == "__main__":
    main()
# quantize-end
