import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def normalize(sample):
    print("start normalize")
    mean = np.mean(sample, axis=0)
    std = np.max(sample, axis=0)-np.min(sample, axis=0)
    for i in range(sample.shape[1]):
        if std[i] != 0:
            sample[:,i] = (sample[:,i] - mean[i])/std[i]
        else:
            sample[:,i] = 0
    return sample

def show_image(image, shape, label="", cmp=None):
    img = np.reshape(image,shape)
    plt.imshow(img,cmap=cmp, interpolation='none')
    plt.title(label)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoidgrad(x):
    return sigmoid(x) * (1 - sigmoid(x))

def checkgradient(grad, num_grad):
    print('Function Gradient', 'Numerical Gradient')
    for i in range(len(grad)):
        print(grad[i], num_grad[i])

    diff = np.linalg.norm(num_grad-grad)/np.linalg.norm(num_grad+grad)
    print('Relative Difference: ')
    print(diff)




