#task-start
import logging
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(filename = "training.log", level=logging.INFO, format = "")
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, nhead=4, num_encoder_layers=2, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[0])
        return output

def get_data_loaders():
    data, labels = pickle.load(open('text_classify_training_data.pkl', 'rb'))

    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader

def train(model, iterator, criterion, optimizer):
    model.train()
    total_loss = 0

    for text, label in iterator:
        optimizer.zero_grad()
        outputs = model(text.transpose(0, 1))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    correct_precisions = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for text, label in iterator:
            outputs = model(text.transpose(0, 1))
            # 在outputs = model(text.transpose(0, 1))这一行，text是一个形状为(batch_size, sequence_length)的Tensor，其中batch_size是批次大小，sequence_length是每个样本的序列长度。transpose(0, 1)操作是用来交换这两个维度，将text的形状从(batch_size, sequence_length)转换为(sequence_length, batch_size)。这样做是因为模型期望输入的顺序是(sequence_length, batch_size)，以便它能够按时间步长处理每个批次的序列。
            # 简单来说，transpose(0, 1)的作用是：
            # 选择第一个维度（batch_size）并将其移动到第二位置。
            # 同时选择第二个维度（sequence_length）并将其移动到第一位置。
            
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # 获取预测类别，_表示丢弃最大值对应的索引，我们只关心最大值（即预测类别）。
            
            correct_precisions += (predicted == label).sum().item()
            # .item(): 将求和结果从张量转换为标量（一个单独的数字）
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            # extend()方法则是用于将另一个可迭代对象的所有元素添加到列表的末尾，逐个地添    加，而不是作为一个整体。
            # 如果传入的是一个列表，extend()会把那个列表的每个元素分别添加到原列表中。
    accuracy = correct_precisions / len(iterator.dataset)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=all_labels, y_pred=all_preds, average='binary')

    return total_loss / len(iterator), accuracy, precision, recall, f1

def run():
    model = TextClassifier()
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 10
    for epoch in trange(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion)  
        logging.info(f"Epoch: {epoch+1}")
        logging.info(f"Train Loss: {train_loss:.3f}")
        logging.info(f'Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy*100:.3f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')

    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    run()
#task-end