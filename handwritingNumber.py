import keras.datasets.mnist as mnist
import keras
from keras import layers
import pandas as pd
import numpy as np


def recognizeNumber(num):
    (train_image, train_label), (test_image, test_label) = mnist.load_data()
    # plt.imshow(train_image[0])
    # print(train_label[0])
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(train_image, train_label, epochs=50, batch_size=512, validation_data=(test_image, test_label))
    model.evaluate(test_image, test_label)
    predNum = np.argmax(model.predict(test_image[:num]), axis=1)
    actualNum = test_label[0:num]
    return predNum, actualNum


if __name__ == '__main__':
    pred, actual = recognizeNumber(10)
    print('predict number is {}, actual number is{}'.format(pred, actual))
