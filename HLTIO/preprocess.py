import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def vconcat(sig,bkg):
    y_sig = np.full(sig.shape[0], 1)
    y_bkg = np.full(bkg.shape[0], 0)
    x = np.concatenate((sig,bkg), axis=0)
    y = np.concatenate((y_sig,y_bkg))

    return x, y

def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    return x_train, x_test, y_train, y_test

def dfSigBkg(df):
    fake = ( df.iloc[:,[27]]==-1. ).all(axis=1)
    df_sig = df[ ~fake ]
    df_bkg = df[ fake ]
    print(r'nSig: %d, nBkg: %d' % (df_sig.shape[0],df_bkg.shape[0]))

    if df_bkg.shape[0] > 50000: df_bkg = df_bkg.sample(50000)
    if df_sig.shape[0] > 50000: df_sig = df_sig.sample(50000)

    df = pd.concat([df_sig,df_bkg],axis=0)
    fake = ( df.iloc[:,[27]]==-1. ).all(axis=1)
    df = df.iloc[:,0:27]

    return df, fake

def stdTransform(x_train, x_test, sig, bkg):
    Transformer = preprocessing.StandardScaler()
    x_train = Transformer.fit_transform(x_train)
    x_test = Transformer.transform(x_test)
    sig = Transformer.transform(sig)
    bkg = Transformer.transform(bkg)

    return x_train, x_test, sig, bkg
