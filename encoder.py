import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

model = keras.models.load_model('auto.model')

def noise(img):
    import random
    img = np.copy(img)
    for i in range(150):
        x = random.randint(0, 27)
        y = random.randint(0, 27)
        img[x, y] += 1
    
    np.clip(img, 0, 1)
    return img

x_test_noisy = np.empty(x_test.shape)
for i in range(len(x_test)):
    x_test_noisy[i] = noise(x_test[i])

def plot_mult(rows, cols, imgs):
    fig, axeslist = plt.subplots(ncols=cols, nrows=rows)
    for ind, title in enumerate(imgs):
        axeslist.ravel()[ind].imshow(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

images = [x_test[1001], x_test_noisy[1001]]
noisy = x_test_noisy
noisy = noisy.reshape(-1, 28, 28, 1)
img = model.predict(noisy)[1001].reshape(28, 28)
images.append(img)
plot_mult(1, 3, images)