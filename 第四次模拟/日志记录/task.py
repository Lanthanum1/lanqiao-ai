# task-start
import logging
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import warnings

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=1000,
        embed_dim=128,
        nhead=4,
        num_encoder_layers=2,
        num_classes=2,
    ):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 用于将词汇索引映射到固定维度的向量
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_encoder_layers,
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text):
        # 前向传播过程，输入文本经过嵌入层转换为向量，然后通过Transformer编码器处理，最后通过全连接层得到分类结果。
        embedded = self.embedding(text)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[0])
        return output


def get_data_loaders():
    data, labels = pickle.load(open("text_classify_training_data.pkl", "rb"))

    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 设置了batch_size=32，意味着每次迭代时模型会同时处理32个样本，这对于加速训练和梯度稳定是有益的；并且设置了shuffle=True，这样在每次训练迭代开始前，训练数据的顺序会被随机打乱，有助于模型学习更加泛化。
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader


def train(model, iterator, criterion, optimizer):
    model.train()
    total_loss = 0

    for text, label in iterator:
        optimizer.zero_grad()
        # 清空模型参数的梯度。这是必要的，因为PyTorch会累积梯度，不清零的话，新批次的梯度会累加到旧批次的梯度上。

        outputs = model(text.transpose(0, 1))
        # text.transpose(0, 1)意味着将张量text的第0维和第1维进行互换。这样是为了确保文本数据的形状符合模型的输入要求，通常是因为模型期望序列数据的形状为(sequence_length, batch_size)，而DataLoader默认提供的是(batch_size, sequence_length)。

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        # 根据计算出的梯度更新模型参数。这一步实际上执行了梯度下降或其他优化算法，以减小损失函数的值。

        total_loss += loss.item()
        # 累加损失的标量值（.item()用于从张量中提取单个数值）。

    return total_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    tp, fp, fn = 0, 0 , 0
    tps, fps, fns = 0, 0, 0
    with torch.no_grad():
        # .no_grad():不跟踪梯度，因为这不是训练过程，不需要更新权重。
        for text, label in iterator:
            outputs = model(text.transpose(0, 1))
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            # 计算每个批次的真正例、假正例和假负例
            tp += ((predicted == 1) & (label == 1)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()

            # 累加整个验证集的真正例、假正例和假负例
            tps += tp
            fps += fp
            fns += fn

    accuracy = correct / total * 100
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1 = 2 * precision * recall / (precision + recall)

    return total_loss / len(iterator), accuracy, precision, recall, f1


def run():
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    model = TextClassifier()
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # lr:learning rate，即学习率，是优化算法中的一个重要参数，控制模型参数更新的速度。

    NUM_EPOCHS = 10
    for epoch in trange(NUM_EPOCHS):
        # trange函数提供了带有进度条的迭代器，使得训练过程更易追踪。

        train_loss = train(model, train_loader, criterion, optimizer)
        # 在每个epoch内，调用train函数训练模型，返回训练过程的平均损失。

        if (epoch + 1) * 3 - 2 >= 0:  
            logger.info(f'Epoch: {epoch + 1}')  
        
        val_loss, val_accuracy, precision, recall, f1 = evaluate(
            model, val_loader, criterion
        )
        
        if (epoch + 1) * 3 - 1 >= 0:  
            logger.info(f'Train Loss: {train_loss:.3f}')

        if (epoch + 1) * 3 >= 0:
            logger.info(f'Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy:.3f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')
            
    
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    run()
# task-end
