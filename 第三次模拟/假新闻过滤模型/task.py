import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt


def create_dict():
    word = {}
    with open('news_train.txt', encoding='UTF-8') as f:
        sentences = f.readlines()
        f.close()

        for sentence in sentences:
            sentence = sentence.split(' ')
            for w in sentence:
                if w not in word.keys():
                    word[w] = len(word)

    return sentences, word


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(30000, 128)
        self.rnn = nn.LSTM(128, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        packed_output, (hidden, cell) = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output


def get_token(sentence, word_dict):
    global max_length
    token = []
    sentence = sentence.split(' ')
    for w in sentence:
        if w in word_dict.keys():
            token.append(word_dict[w])
        else:
            token.append(len(word_dict))
    return np.array(token)


class NewsDataset(Dataset):
    def __init__(self, sentences, label=None, type_='train', word_dict=None):
        super().__init__()
        self.sentences = sentences
        self.label = label
        self.type = type_
        self.word_dict = word_dict

    def __getitem__(self, item):
        if self.type == 'train':
            return get_token(self.sentences[item], self.word_dict), self.label[item]
        return get_token(self.sentences[item], self.word_dict)

    def __len__(self):
        return len(self.sentences)


def train(model, dataloader):
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []

    for e in range(epochs):
        epoch_loss = 0
        for i, (features, labels) in enumerate(dataloader):
            features = features.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / (i + 1))
        print(f'Epoch {e + 1}/{epochs}, Loss: {epoch_loss / (i + 1)}')