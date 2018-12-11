import numpy as np
import pandas as pd
def loaddata(index = 1):
    print("start load data")
    if index == 1:
        dtrain = pd.read_csv("./train_th.csv")
        dtest = pd.read_csv("./test_th.csv")
    else:
        dtrain = pd.read_csv("./train.csv")
        dtest = pd.read_csv("./test.csv")

    train = dtrain.as_matrix()
    test = dtest.as_matrix()
    print(train.shape)
    print("end load data")
    return (train,test)

