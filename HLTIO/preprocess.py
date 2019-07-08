import sys
import numpy as np
from sklearn.model_selection import train_test_split

def vconcat(sig,bkg):
    y_sig = np.full(sig.shape[0], 1)
    y_bkg = np.full(bkg.shape[0], 0)
    x = np.concatenate((sig,bkg), axis=0)
    y = np.concatenate((y_sig,y_bkg))

    return x, y

def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    return x_train, x_test, y_train, y_test
