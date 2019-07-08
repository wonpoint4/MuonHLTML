import sys
import numpy as np
import pandas as pd

def maskTarget(x, y):
    y = np.reshape(y, (-1,1))
    stacked = np.hstack((y,x))
    df = pd.DataFrame(data=stacked)
    mask = df[0] == 1
    dfSig = df[mask]
    dfBkg = df[~mask]
    dfSig = dfSig.drop([0],axis=1)
    dfBkg = dfBkg.drop([0],axis=1)

    return dfSig.values, dfBkg.values
