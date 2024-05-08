# task-start
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.autograd import Variable


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
        packed_output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def convert():
    # TODO
    model = TextClassifier()
    dummy_input = Variable(torch.LongTensor([101, 304, 993, 1008, 102]))
    torch.onnx.export(
        model, dummy_input, "text_classifier.onnx", export_params=True, opset_version=11
    )


def inference(model_path, input):
    # TODO
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_data = np.array(input, dtype=np.int64).reshape(1, -1)
    output = session.run(None, {input_name: input_data})
    return output[0]


def main():
    convert()
    result = inference("./text_classifier.onnx", [101, 304, 993, 108, 102])
    print(result)


if __name__ == "__main__":
    main()
# task-end
