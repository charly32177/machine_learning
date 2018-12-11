import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from function import *

def initalhetaT():
    nn =np.arange(1, 19, dtype=int)/10
    X = np.cos( np.array([[1, 2],[3, 4],[5, 6]]) )
    y = np.array([[0,0,0,1] ,[0,1,0,0], [0,0,1,0]])
    return nn,X,y

def Fh_thetaT(Theta1,Theta2,X):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=1, axis=1)
    a2 = sigmoid(a1@Theta1.T) #Theta1 = L2 * n+1 a1 = m * n+1 a2 = L2 * m
    a2 = np.insert(a2, 0, values=1, axis=1) #L2+1 * m
    h_theta = sigmoid(a2@Theta2.T) #Theta2 = L3 * L2+1 a3 = L3*m
    return h_theta, a1, a2

def costfunctionT(nn_params,L1,L2,L3,X,y,lambda2):
    #forward
    m = X.shape[0]
    [Theta1,Theta2] =reshapeThetaT(nn_params,L1,L2,L3)
    print(Theta1,'Theta1')
    print(Theta2,'Theta2')

    h_theta,a1,a2 = Fh_thetaT(Theta1,Theta2,X)
    print(h_theta,'h_theta')
    theta_sum_square = np.sum(Theta1**2) + np.sum(Theta2**2) - (np.sum(Theta1[:,0]**2)+np.sum(Theta2[:,0]**2) )
    J = (-1/m)* np.sum( y * np.log(h_theta) + (1-y)*np.log(1-h_theta)  ) + (lambda2/(2*m)*theta_sum_square)
    return J

def gradthetaT(nn_params,L1,L2,L3,X,y,lambda2):
    m = X.shape[0]
    [Theta1,Theta2] =reshapeThetaT(nn_params,L1,L2,L3)
    h_theta ,a1,a2 = Fh_thetaT(Theta1,Theta2,X)
    delta3 = (h_theta - y) # m * L3        "theta2" L3 * (L2+1) "theta1" L2 * (L1+1)
    print(sigmoidgrad(a1@Theta1.T),'Z2')
    delta2 = delta3@Theta2 * np.insert(sigmoidgrad(a1@Theta1.T), 0, values=1, axis=1)
    # delta2 = m * L3 @ L3 * L2+1 = (m * L2+1) (a1@ Theta1.T = m*L2)
    delta2 = delta2[:,1:] # m*L2
    print(delta2,'delta2')
    print(delta3,'delta3')
    Theta1_grad = ((1/m) * ( delta2.T@a1 + (lambda2*np.insert(Theta1[:,1:], 0, values=0, axis=1))  )).T.ravel()   #m*L2 @ m*L1+1
    Theta2_grad = ((1/m) * ( delta3.T@a2 + (lambda2*np.insert(Theta2[:,1:], 0, values=0, axis=1))  )).T.ravel()
    Theta1_gradall = np.concatenate((Theta1_grad,Theta2_grad), axis=None)
    return Theta1_gradall
# m * (L3+1)
def NumgradthetaT(nn_params,L1,L2,L3,X,y,lambda2):
    e = 1e-4
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    print(numgrad,'numgrad')
    print(perturb,'perturb')
    print(nn_params,'nn_params')
    for i in range(nn_params.shape[0]):
        perturb[i] = e
        print(nn_params+perturb,'nn_params+perturb')
        loss1 = costfunctionT(nn_params+perturb,L1,L2,L3,X,y,lambda2)
        loss2 = costfunctionT(nn_params-perturb,L1,L2,L3,X,y,lambda2)
        numgrad[i] = (loss1 - loss2) / (2*e)
        perturb[i] = 0
    
    return numgrad


#([ones(1,m);sigmoidGradient(Theta1*a1')])'
def reshapeThetaT(nn_params,L1,L2,L3):
    [Theta1,Theta2] =nn_params[0:(L2*(L1+1))].reshape((L1+1),L2).T,nn_params[ (L2*(L1+1)) : (L2*(L1+1))+(L3*(L2+1))].reshape((L2+1),L3).T
    return [Theta1,Theta2]


[nn,X,y] = initalhetaT()
print(costfunctionT(nn.copy(),2,2,4,X,y,0),'cost')
grad = gradthetaT(nn.copy(),2,2,4,X,y,0)
num_grad = NumgradthetaT(nn.copy(),2,2,4,X,y,0)
checkgradient(grad,num_grad)

