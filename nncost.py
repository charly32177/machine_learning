import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from function import *

def initalheta(L1,L2,L3,L4):
    epsilon_init1 = 6**0.5/(L1**0.5+L2**0.5)
    epsilon_init2 = 6**0.5/(L2**0.5+L3**0.5)
    epsilon_init3 = 6**0.5/(L3**0.5+L4**0.5)
    Theta1 = (np.random.randn(L2,L1+1)*2*epsilon_init1-epsilon_init1).T.ravel()
    Theta2 = (np.random.randn(L3,L2+1)*2*epsilon_init2-epsilon_init2).T.ravel()
    Theta3 = (np.random.randn(L4,L3+1)*2*epsilon_init3-epsilon_init3).T.ravel()
    Thetaall = np.concatenate((Theta1,Theta2,Theta3), axis=None)
    return Thetaall

def Fh_theta(Theta1,Theta2,Theta3,X):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=1, axis=1) # m * (L1+1)
    a2 = sigmoid(a1@Theta1.T) #Theta1 = L2 * n+1 a1 = m * L1+1 a2 =  m * L2
    a2 = np.insert(a2, 0, values=1, axis=1) # m *L2+1
    a3 = sigmoid(a2@Theta2.T) #Theta2 = L3 * L2+1 a3 = m * L3
    a3 = np.insert(a3, 0, values=1, axis=1) # a3 = m * L3+1
    h_theta = sigmoid( a3@Theta3.T ) #Theta3 = L4 * L3+1 (L4 * m)
    return h_theta, a1 , a2, a3

def costfunction(nn_params,L1,L2,L3,L4,X,y,lambda2):
    #forward
    m = X.shape[0]
    [Theta1,Theta2,Theta3] =reshapeTheta(nn_params,L1,L2,L3,L4)
    h_theta,a1,a2,a3 = Fh_theta(Theta1,Theta2,Theta3,X)
    theta_sum_square = np.sum(Theta1**2) + np.sum(Theta2**2) + np.sum(Theta3**2) - (np.sum(Theta1[:,0]**2) + np.sum(Theta2[:,0]**2) + np.sum(Theta3[:,0]**2))
    J = (-1/m)* np.sum( y * np.log(h_theta) + (1-y)*np.log(1-h_theta)  ) + (lambda2/(2*m)*theta_sum_square)
    return J

def gradtheta(nn_params,L1,L2,L3,L4,X,y,lambda2):
    m = X.shape[0]
    [Theta1,Theta2,Theta3] =reshapeTheta(nn_params,L1,L2,L3,L4)
    h_theta ,a1,a2,a3 = Fh_theta(Theta1,Theta2,Theta3,X)
    delta4 = (h_theta - y) # m * L4          "theta3" L4 * (L3+1)
    delta3 = delta4@Theta3 * np.insert(sigmoidgrad(a2@Theta2.T), 0, values=1, axis=1) # delta3 = m * L4 @ L4 * L3+1 = (m * L3+1) (Theta2 L3 * L2+1 @ a2 = (L2+1)*m)
    delta3 = delta3[:,1:] # m*L3
    delta2 = delta3@Theta2 * np.insert(sigmoidgrad(a1@Theta1.T), 0, values=1, axis=1) # Theta2=L3 * L2+1 = m*(L2+1) (Theta1 @ a1 = L2+1*m)
    delta2 = delta2[:,1:] #m*L2
    Theta1_grad = ((1/m) * ( delta2.T@a1 + (lambda2*np.insert(Theta1[:,1:], 0, values=0, axis=1) )  )).T.ravel()   #m*L2 @ m*L1+1
    Theta2_grad = ((1/m) * ( delta3.T@a2 + (lambda2*np.insert(Theta2[:,1:], 0, values=0, axis=1))  )).T.ravel()
    Theta3_grad = ((1/m) * ( delta4.T@a3 + (lambda2*np.insert(Theta3[:,1:], 0, values=0, axis=1))  )).T.ravel()
    Theta1_gradall = np.concatenate((Theta1_grad,Theta2_grad,Theta3_grad), axis=None)
    return Theta1_gradall
# m * (L3+1)
def Numgradtheta(nn_params,L1,L2,L3,L4,X,y,lambda2):
    e = 1e-4
    numgrad = np.zeros(nn_params.shape)
    perturb = np.zeros(nn_params.shape)
    for i in range(nn_params.shape[0]):
        perturb[i] = e
        loss1 = costfunction(nn_params+perturb,L1,L2,L3,L4,X,y,lambda2)
        loss2 = costfunction(nn_params-perturb,L1,L2,L3,L4,X,y,lambda2)
        numgrad[i] = (loss1 - loss2) / (2*e)
        perturb[i] = 0

    return numgrad


#([ones(1,m);sigmoidGradient(Theta1*a1')])'
def reshapeTheta(nn_params,L1,L2,L3,L4):
    [Theta1,Theta2,Theta3] =nn_params[0:(L2*(L1+1))].reshape((L1+1),L2).T,nn_params[ (L2*(L1+1)) : (L2*(L1+1))+(L3*(L2+1))].reshape((L2+1),L3).T,nn_params[ (L2*(L1+1))+(L3*(L2+1)):].reshape((L3+1),L4).T
    return [Theta1,Theta2,Theta3]




