import numpy as np
import pandas as pd
from loaddata import *

train, test = loaddata(2)

trains = train[:5001,:]
test = test[:5001,:]
df = pd.DataFrame(trains)
df.to_csv("train_th.csv",header=None, index=None)
dft = pd.DataFrame(test)
dft.to_csv("test_th.csv",header=None, index=None)

