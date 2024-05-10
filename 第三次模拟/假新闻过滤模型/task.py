import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline




# 数据预处理
def preprocess_data(text_file, label_file):
    texts = []
    with open(text_file, 'r') as f_text:
        for line in f_text:
            texts.append(line.strip())
    if text_file == "news_train.txt": # 预处理训练数据才需要有标签
        labels = []
        with open(label_file, 'r') as f_label:
            for line in f_label:
                labels.append(int(line.strip()))

        return texts, labels
    else:
        return texts

# 训练模型
# def train_model(X_train, y_train):
#     vectorizer = CountVectorizer()
#     X_train_vec = vectorizer.fit_transform(X_train)
#     # 词袋模型（Bag-of-Words Model, BoW）是一种在自然语言处理中用于表示文本的统计模型。它忽略了文本中词的顺序和语法结构，只关注词的出现与否以及出现的频率。在BoW模型中，一个文本被看作是一个词的集合，每个词被视为独立的特征，而文本的整体意义是由这些特征的组合来表示的。
    
#     classifier = LogisticRegression(penalty='l2', solver='liblinear')  # 使用L2正则化的逻辑回归
#     classifier.fit(X_train_vec, y_train)
#     return classifier, X_train_vec, vectorizer
def train_and_validate_model(X_train, y_train, X_valid, y_valid):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    classifier = LogisticRegression(penalty='l2', solver='liblinear')
    classifier.fit(X_train_vec, y_train)

    y_pred_train = classifier.predict(X_train_vec)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy: {accuracy_train}")

    y_pred_valid = classifier.predict(X_valid_vec)
    accuracy_valid = accuracy_score(y_valid, y_pred_valid)
    print(f"Validation Accuracy: {accuracy_valid}")

    return classifier, vectorizer


# 预测并保存结果
def predict_and_save(model, vectorizer, test_file, output_file):
    test_texts = []
    with open(test_file, 'r') as f_test:
        for line in f_test:
            test_texts.append(line.strip())

    X_test_vec = vectorizer.transform(test_texts)
    predictions = model.predict(X_test_vec)

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(str(int(pred)) + '\n')
    

if __name__ == "__main__":
    X_train, y_train = preprocess_data('news_train.txt', 'label_newstrain.txt')
    X_test = preprocess_data('news_test.txt', None)[0]

    # model, X_train_vec, vectorizer = train_model(X_train, y_train)

    # # 验证准确率
    # y_pred_train = model.predict(X_train_vec)
    # accuracy_train = accuracy_score(y_train, y_pred_train)
    # # print(f"Training Accuracy: {accuracy_train}")
    
    # 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

    model, vectorizer = train_and_validate_model(X_train, y_train, X_valid, y_valid)

    # 预测并保存结果
    predict_and_save(model, vectorizer, 'news_test.txt', 'pred_test.txt')