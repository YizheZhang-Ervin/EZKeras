import keras
from keras import layers
import pandas as pd


def regression(x, y, predictx):
    model = keras.Sequential()
    model.add(layers.Dense(1, input_dim=3))
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=200)
    return model.predict(predictx)


if __name__ == '__main__':
    x1 = [0, 1, 2, 3, 4, 5, 6]
    x2 = [3, 4, 5, 6, 7, 8, 9]
    x3 = [9, 8, 7, 6, 5, 4, 3]
    x_train = pd.DataFrame([x1, x2, x3]).T
    y_train = [11, 22, 33, 44, 55, 66, 77]
    x_predict = pd.DataFrame([[3, 4, 5]])
    y_predict = regression(x_train, y_train, x_predict)
    print('---------------------------', y_predict)
