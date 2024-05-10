import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def load_data():
    with open("dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

        train_data = np.asarray(data[b'data'])
        train_labels = data[b'labels']

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = to_categorical(train_labels)
    return train_data, train_labels

def build_model_and_train():
    train_images, train_labels = load_data()

    # 初始化一个顺序模型
    model = Sequential()

    # 添加卷积层
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))

    # 添加更多的卷积层
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 展平层
    model.add(Flatten())

    # 添加全连接层
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))  # 10 分类

    # 编译模型，使用分类交叉熵作为损失函数
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, 
              epochs=100,  # 可以设置一个较大的迭代次数，以确保模型能够过拟合
              batch_size=32, 
              verbose=2)  # 显示训练进度

    # 检查训练损失和准确率
    train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
    if train_loss < 1e-3 and train_acc == 100.0:
        print("模型训练完成，损失小于 1e-3，准确率达到 100%")

        # 保存模型
        model.save('image_classify.h5')
        print("模型已保存为 'image_classify.h5'")
    else:
        print("模型未能达到指定的性能指标")

build_model_and_train()