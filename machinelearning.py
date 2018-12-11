import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from loaddata import *
from function import *
from nncost import *
from test import *
from scipy import optimize
from nntrain import *
from learningcurve import *
train, test = loaddata(2) #input 1 load 1000 data, else num load full data
train_y = train[:,0].astype('int8')
train_x = train[:,1:].astype('float64')
train_y = pd.get_dummies(train_y).as_matrix()
train_x, vad_x, train_y, vad_y = train_test_split(train_x, train_y, test_size=0.3, random_state=0)
#plt.subplot(1, 2, 1)
#ni = np.random.randint(0,train_x.shape[0],1)[0]
#show_image(train_x[ni],(28,28),  cmp="gray")
train_x = normalize(train_x)
test = normalize(test)
vad_x = normalize(vad_x)
#plt.subplot(1, 2, 2)
#show_image(train_x[ni],(28,28),  cmp="gray")
#plt.show()
input_layer_size  = train_x.shape[1]
hidden_layer1_size = 50
hidden_layer2_size = 30
num_labels = train_y.shape[1]
lambda2 = 3.4
nn_params = initalheta(input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels)
#J = costfunction(nn_params.copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2)
res,steps= nntrain(nn_params.copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2)
#res,steps= nntrainmini(nn_params.copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2,1000)
res = np.array(res)
acc = predict(res,train_x,train_y,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels)
vadacc = predict(res,vad_x,vad_y,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels)
plt.plot(np.array(steps[10:]))
plt.show()
#learningcurve(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,vad_x,vad_y,lambda2)
#validationcurve(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,vad_x,vad_y,lambda2)
