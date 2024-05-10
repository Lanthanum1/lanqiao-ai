#task-start
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


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


build_model_and_train()
#task-end