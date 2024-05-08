# task-start
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=512, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    # 类实例化了以下三个组件：
    # self.embedding: 一个 nn.Embedding 层，用于将输入的单词索引转换为固定长度的向量表示。
    # self.rnn: 一个双向 LSTM 层，用于处理输入的嵌入向量序列，捕捉上下文信息。
    # self.fc: 一个全连接层（线性层），nn.Linear，将 LSTM 输出映射到类别概率。

    def forward(self, text):

        embedded = self.embedding(text)
        # 将输入的单词索引通过嵌入层转化为向量表示。

        packed_output, (hidden, cell) = self.rnn(embedded)
        # LSTM 层处理嵌入向量序列，packed_output 包含打包后的输出，hidden 和 cell 分别是 LSTM 的最后一步隐藏状态和细胞状态。

        output = self.fc(hidden.squeeze(0))
        # 使用全连接层对 LSTM 的最终隐藏状态进行分类。squeeze(0) 是为了从形状 (1, hidden_dim) 转换为 (hidden_dim,)，因为全连接层期望一个平坦的输入。

        return output  # 返回模型的输出，这通常是一个形状为 (batch_size, num_classes) 的张量，表示每个样本属于每个类别的概率。


def convert():
    # 加载PyTorch模型
    model = TextClassifier()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    # input_shape = (256, 1)
    # 创建一个样本输入张量。尺寸应匹配模型期望的输入。
    # x = torch.randn(1, *input_shape)
    x = torch.ones([1,1], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    # 导出模型
    torch.onnx.export(model,               # 模型实例
                      x,                   # 模型输入的样本张量
                      "text_classifier.onnx",  # 保存的模型名
                      export_params=True,  # 导出模型中的参数和缓存
                      opset_version=11,    # 使用的 ONNX 版本
                      do_constant_folding=True,  # 是否执行常量折叠优化
                      input_names = ['input'],   # 模型输入节点的名字
                      output_names = ['output'], # 模型输出节点的名字
                      dynamic_axes={'input': {0: 'length'}, 'output': {0: 'length'}}) # 动态轴的设置


# def inference(model_path, input):
#     # 初始化ONNX运行时
#     ort_session = ort.InferenceSession(model_path)

#     # ONNX模型期望建模输入的形状与名字
#     ort_inputs = {ort_session.get_inputs()[0].name: np.array(input, dtype=np.int64).reshape(1, -1)}
#     ort_outs = ort_session.run(None, ort_inputs)

#     # 返回结果
#     result = ort_outs[0].tolist()
#     return result

def inference(model_path, input_sequence):
    # 将输入序列转换为形状为 (len(input_sequence), 1) 的张量
    input_tensor = torch.tensor(input_sequence).unsqueeze(1)

    # 将PyTorch张量转换为NumPy数组，因为ONNX运行时需要NumPy数组作为输入
    numpy_input = input_tensor.numpy()

    # 初始化ONNX运行时
    ort_session = ort.InferenceSession(model_path)

    # ONNX模型期望建模输入的形状与名字
    ort_inputs = {ort_session.get_inputs()[0].name: numpy_input}

    # ONNX运行时执行推理
    ort_outs = ort_session.run(None, ort_inputs)

    # 返回结果
    result = ort_outs[0].tolist()
    return result 



def main():
    convert()
    result = inference("./text_classifier.onnx", [101, 304, 993, 108, 102])
    # print("-" * 10)
    print(result)


if __name__ == "__main__":
    # model = TextClassifier() 
    # model.load_state_dict(torch.load('model.pt'))
    # model.eval()
    
    
    # output = model(torch.ones([256, 1], dtype=torch.long))
    # print(output.detach().numpy().tolist())
    
    
    # input_sequence = [101, 304, 993, 108, 102]
    # input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(1)
    # output = model(input_tensor)
    # print(output.detach().numpy().tolist())

    main()
# task-end
