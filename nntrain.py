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

def nntrain(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2):
    def save_step(k):
        nonlocal steps
        steps.append(costfunction(k,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2))
    steps = []

    res= optimize.fmin_cg(costfunction,fprime=gradtheta, x0=nn_params.copy(),args=(input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2),callback=save_step,epsilon=1.5e-4)#[xopt,fopt, func_calls, grad_calls,warnflag]
    return res,steps

def predict(res,X,y,L1,L2,L3,L4):
    [Theta1,Theta2,Theta3] =reshapeTheta(res,L1,L2,L3,L4)
    h_theta,a1,a2,a3 = Fh_theta(Theta1,Theta2,Theta3,X)
    print(np.mean(np.argmax(h_theta,axis=1) == np.argmax(y,axis=1)))
    return np.mean(np.argmax(h_theta,axis=1) == np.argmax(y,axis=1))

def nntrainmini(nn_params,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x,train_y,lambda2,batch):
    steps = []
    res = []
    update_res = nn_params.copy()
    for i in range(0,train_x.shape[0],batch):
        def save_step(k):
            nonlocal steps
            steps.append(costfunction(k,input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[i:i+batch,:],train_y[i:i+batch,:],lambda2))
        
        print(i/train_x.shape[0]*100,'%')
        res= optimize.fmin_cg(costfunction,fprime=gradtheta, x0=update_res,args=(input_layer_size,hidden_layer1_size,hidden_layer2_size,num_labels,train_x[i:i+batch,:],train_y[i:i+batch,:],lambda2),callback=save_step,epsilon=1.5e-4)#[xopt,fopt, func_calls, grad_calls,warnflag]
        update_res = res
    
    return res,steps
