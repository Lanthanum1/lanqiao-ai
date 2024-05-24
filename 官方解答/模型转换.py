#task-start
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

    def forward(self, text):

        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


def convert():
    # TODO
    model = TextClassifier()
    model.load_state_dict(torch.load('/home/project/model.pt'))
    model.eval()
    
    x = torch.ones([128, 1], dtype=torch.long)
    # 这个128实际上是输入的batch_size，可以任意设置，但是要和模型输入的batch_size一致，所以其实换成256，一样是可以过的
    torch.onnx.export(model=model, args=x, f='text_classifier.onnx', input_names=['input'],dynamic_axes={'input': {0: 'batch_size'}})
    # dynamic_axes: 定义输入或输出张量的动态轴。在这里，input的第0个维度（batch_size）被标记为动态，意味着在推理时可以接受不同大小的批次。
    
    


def inference(model_path, input):
    # TODO
    session = ort.InferenceSession(model_path)
    input_data = np.array(input).reshape(-1, 1)
    result = session.run(None, {'input': input_data})
    # 使用session.run方法执行模型推理。第一个参数None表示不请求任何特定的输出节点，因为ONNX模型的输出节点默认已知。第二个参数是一个字典，键是模型输入的名字（在ONNX模型中定义），值是对应的输入数据。在这里，我们只有一个输入，名为'input'，其值是经过预处理的input_data。
    return result[0].tolist()


def main():
    convert()
    result = inference('/home/project/text_classifier.onnx', [101, 304, 993, 108,102])
    print(result)


if __name__ == '__main__':
    main()
#task-end