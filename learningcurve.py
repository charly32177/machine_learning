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

def learningcurve(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,vad_x,vad_y,lambda2):
    error_train = []
    error_vad = []
    for i in range(20,100,2):
        res,steps= nntrain(nn_params.copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[0:i,:],train_y[0:i,:],lambda2)
        error_train.append(costfunction(np.array(res[0]).copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[0:i,:],train_y[0:i,:],lambda2))
        error_vad.append(costfunction(np.array(res[0]).copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,vad_x,vad_y,lambda2))
    
    plt.plot(np.arange(20,100,2),error_train, label='error_train')
    plt.plot(np.arange(20,100,2),error_vad, label='error_vad')
    plt.legend()
    plt.show()

def validationcurve(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,vad_x,vad_y,lambda2):
    lambda2 = np.arange(2,4,0.2)
    error_train = []
    error_vad = []
    for i in range(lambda2.shape[0]):
        res,steps= nntrain(nn_params.copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[0:100,:] ,train_y[0:100,:] ,lambda2[i])
        error_train.append(costfunction(np.array(res[0]).copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[0:100,:] ,train_y[0:100,:] ,lambda2[i]))
        error_vad.append(costfunction(np.array(res[0]).copy(),input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,vad_x[0:100,:] ,vad_y[0:100,:] ,lambda2[i]))

    plt.plot(lambda2,error_train, label='error_train')
    plt.plot(lambda2,error_vad, label='error_vad')
    plt.legend()
    plt.show()
