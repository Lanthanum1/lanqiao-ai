#task-start
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data():

    with open("dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

        train_data = np.asarray(data[b'data'][:10])
        train_labels = data[b'labels'][:10]

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = to_categorical(train_labels)
    return train_data, train_labels


def build_model_and_train():
    train_images, train_labels = load_data()
    model = Sequential()

    # TODO
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # 添加了一个卷积层（Conv2D）。它有32个滤波器，每个滤波器的大小为3x3，激活函数为ReLU。input_shape参数指定了输入图像的尺寸，这里是32x32像素的RGB图像。
    model.add(MaxPooling2D((2, 2)))
    # 添加了一个最大池化层（MaxPooling2D）。池化窗口的大小为2x2，这有助于减小特征图的尺寸并减少计算量。
    model.add(Flatten())
    # 添加了一个展平层（Flatten）。这将前一层的多维输出转换为一维，为全连接层做准备。
    model.add(Dense(10, activation='softmax'))
    # 添加了一个全连接层（Dense），具有10个节点，对应于可能的10个类别，激活函数为softmax，用于分类任务。
    model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=['accuracy'])
    # 编译模型，选择了Adam优化器，损失函数为分类交叉熵（categorical_crossentropy），用于多类别的分类问题，以及评估指标为准确率。
    model.fit(train_images, train_labels, epochs=100)
    model.save("image_classify.h5")
    


build_model_and_train()
#task-end