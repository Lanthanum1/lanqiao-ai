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
    # TODO
    model = TextClassifier()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    # 定义输入张量（示例： batch_size=2, seq_len=5）
    input_tensor = torch.randint(0, 1000, (1, 5)).long()

    torch.onnx.export(model, input_tensor, "text_classifier.onnx")


def inference(model_path, input):
    # TODO
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_data = np.array(input, dtype=np.int64).reshape(1, -1)
    output = session.run(None, {input_name: input_data})
    result = output[0].flatten().tolist()
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
    main()
# task-end
