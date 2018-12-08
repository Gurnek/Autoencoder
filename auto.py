import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_train = x_train.reshape(-1, 28, 28, 1)

def noise(img):
    import random
    img = np.copy(img)
    for i in range(100):
        x = random.randint(0, 27)
        y = random.randint(0, 27)
        img[x, y] += 1
    
    np.clip(img, 0, 1)
    return img

x_train_noisy = np.empty(x_train.shape)
for i in range(len(x_train)):
    x_train_noisy[i] = noise(x_train[i])


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adadelta', loss='binary_crossentropy')

model.fit(x_train_noisy, x_train, epochs=30)

model.save('auto.model')