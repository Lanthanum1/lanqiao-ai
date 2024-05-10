# task-start
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def load_data():

    with open("dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

        train_data = np.asarray(data[b"data"][:10])
        train_labels = data[b"labels"][:10]

    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = to_categorical(train_labels)  # 将标签转换为 one-hot 编码
    return train_data, train_labels


def build_model_and_train():
    train_images, train_labels = load_data()

    # 定义一个简单的卷积神经网络模型
    model = Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(32, 32, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3200, activation="relu"),
            # tf.keras.layers.Dropout(0.0), # 用于正则化，防止过拟合。
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # 编译模型，使用 categorical_crossentropy 作为损失函数
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # 设置足够的训练轮数以达到过拟合
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=10, restore_best_weights=True
    # )
    # 监控验证集损失，如果在 10 个周期内没有改善，则停止训练并恢复最好权重。

    checkpoint = tf.keras.callbacks.ModelCheckpoint("image_classify.h5")
    # 回调保存模型的最佳版本

    # 训练模型，使用验证集检查过拟合
    history = model.fit(
        train_images,
        train_labels,
        epochs=50,
        batch_size=320,
        validation_split=0,
        callbacks=[checkpoint],
    )

    # 检查训练损失和准确率，直到满足条件
    if history.history["loss"][-1] < 1e-3 and history.history["accuracy"][-1] == 1.0:
        print("模型训练完成，已保存到 image_classify.h5")
    else:
        print("模型未达到训练要求")


build_model_and_train()
# task-end
