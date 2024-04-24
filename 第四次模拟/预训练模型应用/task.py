# 需要修改
import os
import json
import zipfile
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: 数据加载与处理

def load_data(file_path: str) -> List[Dict[str, List[float]]]:
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('/home/project/')
        
    data = []
    for file_name in os.listdir('/home/project/'):
        if file_name.endswith('.json'):
            with open(os.path.join('/home/project/', file_name), 'r') as f:
                data.extend(json.load(f))
                
    return data

def merge_features(data: List[Dict[str, List[float]]]) -> (np.ndarray, np.ndarray):
    features = np.zeros((len(data), 6144))
    labels = np.zeros(len(data))
        
    for i, sample in enumerate(data):
        labels[i] = sample['label']
        features[i, :2048] = sample['feature'][:2048]  # resnet 特征
        features[i, 2048:4096] = sample['feature'][2048:4096]  # inception 特征
        features[i, 4096:] = sample['feature'][4096:]  # xception 特征

    return features, labels

train_data = load_data('/home/project/train_feature.zip')
train_features, train_labels = merge_features(train_data)

test_data = load_data('/home/project/test_feature.zip')

# Step 2: 模型构建与训练

model = LogisticRegression(max_iter=1000)
model.fit(train_features, train_labels)

# Step 3: 模型评估
train_pred_labels = model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_pred_labels)
print("Training accuracy:", train_accuracy)

# Ensure the training accuracy meets the requirement of at least 95%
assert train_accuracy >= 0.95, "Training accuracy should be at least 95%"

# Step 4: 测试数据预测
test_features = np.zeros((len(test_data), 6144))

for i, sample in enumerate(test_data):
    test_features[i, :] = sample['feature']

test_pred_labels = model.predict(test_features)

# Step 5: 结果输出
with open('/home/project/result.csv', 'w') as f:
    f.write('id,label\n')
    for i, sample in enumerate(test_data):
        f.write(f"{sample['id']},{int(test_pred_labels[i])}\n")