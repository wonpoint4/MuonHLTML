import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils

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
    fake0 = ( df.iloc[:,[32]]==0. ).all(axis=1)
    fake1 = ( df.iloc[:,[32]]==1. ).all(axis=1)
    sig0 = ( df.iloc[:,[32]]==2. ).all(axis=1)
    sig1 = ( df.iloc[:,[32]]==3. ).all(axis=1)
    df_fake0 = df[fake0]
    df_fake1 = df[fake1]
    df_sig0 = df[sig0]
    df_sig1 = df[sig1]
    print(r'nBkg0: %d, nBkg1: %d, nSig0: %d, nSig1: %d' % (df_fake0.shape[0],df_fake1.shape[0],df_sig0.shape[0],df_sig1.shape[0]))

    if df_fake0.shape[0] > 25000: df_fake0 = df_fake0.sample(25000)
    if df_fake1.shape[0] > 25000: df_fake1 = df_fake1.sample(25000)
    if df_sig0.shape[0] > 25000: df_sig0 = df_sig0.sample(25000)
    if df_sig1.shape[0] > 25000: df_sig1 = df_sig1.sample(25000)

    df = pd.concat([df_fake0,df_fake1,df_sig0,df_sig1],axis=0)

    y_f0 = np.full(df_fake0.shape[0],0,np.int32)
    y_f1 = np.full(df_fake1.shape[0],1,np.int32)
    y_s0 = np.full(df_sig0.shape[0],2,np.int32)
    y_s1 = np.full(df_sig1.shape[0],3,np.int32)
    y = np.concatenate((y_f0,y_f1))
    y = np.concatenate((y,y_s0))
    y = np.concatenate((y,y_s1))
    y = pd.Series(y,name='y')
    print(r'y shape: %d' % (y.shape) )
    df = df.iloc[:,0:32]

    return df, y

def computeClassWgt(y, y_test):
    wgts = utils.class_weight.compute_class_weight('balanced',np.unique(y),y)

    y_wgts = np.full(y.shape[0],1.)
    ytest_wgts = np.full(y_test.shape[0],1.)
    for i,v in enumerate(wgts):
        y_wgts = np.multiply(y_wgts,np.where(y==i,v,1.))
        ytest_wgts = np.multiply(ytest_wgts,np.where(y_test==i,v,1.))

    return y_wgts, ytest_wgts, wgts

def getNclass(df):
    #notBuilt = df[df['matchedTPsize']==-99999.]
    #combi = df[df['matchedTPsize']==0.]
    #simMatched = df[df['matchedTPsize']>0.].copy()
    #muMatched = df[ (df['bestMatchTP_pdgId']==13.) | (df['bestMatchTP_pdgId']==-13.) ]

    #simMatched.drop(muMatched.index.values, inplace=True)

    #return notBuilt, combi, simMatched, muMatched
    #return notBuilt+combi+simMatched, muMatched

    sig = df[ (df['bestMatchTP_pdgId']==13.) | (df['bestMatchTP_pdgId']==-13.) ]
    df.drop(sig.index.values, inplace=True)
    return df, sig

def filterClass(df):
    df.drop(
        [
            'mva0',
            'mva1',
            'mva2',
            'mva3',
            'truePU',
            'dir',
            'tsos_detId',
            'tsos_pt',
            'tsos_eta',
            'tsos_phi',
            'tsos_glob_x',
            'tsos_glob_y',
            'tsos_glob_z',
            'tsos_pt_val',
            'tsos_hasErr',
            'tsos_charge',
            #'tsos_qbp',
            #'tsos_dxdz',
            #'tsos_dydz',
            'tsos_x',
            'tsos_y',
            'tsos_px',
            'tsos_py',
            'tsos_pz',
            'dR_minDRL1SeedP',
            'dPhi_minDRL1SeedP',
            'dR_minDPhiL1SeedX',
            'dPhi_minDPhiL1SeedX',
            #'dR_minDRL1SeedP_AtVtx',
            #'dPhi_minDRL1SeedP_AtVtx',
            'dR_minDPhiL1SeedX_AtVtx',
            'dPhi_minDPhiL1SeedX_AtVtx',
            'dR_minDRL2SeedP',
            'dPhi_minDRL2SeedP',
            'dR_minDPhiL2SeedX',
            'dPhi_minDPhiL2SeedX',
            'dR_L1TkMuSeedP',
            'dPhi_L1TkMuSeedP',
            'bestMatchTP_pdgId',
            'matchedTPsize',
            'gen_pt',
            'gen_eta',
            'gen_phi',
            #'tsos_err0',
            'tsos_err1',
            #'tsos_err2',
            'tsos_err3',
            'tsos_err4',
            #'tsos_err5',
            'tsos_err6',
            'tsos_err7',
            'tsos_err8',
            'tsos_err9',
            'tsos_err10',
            'tsos_err11',
            'tsos_err12',
            'tsos_err13',
            'tsos_err14',
            ], 
        axis=1, inplace=True
    )

    return df

def stdTransform(x_train, x_test):
    Transformer = preprocessing.StandardScaler()
    x_train     = Transformer.fit_transform(x_train)
    x_test      = Transformer.transform(x_test)
    mean        = Transformer.mean_
    std         = Transformer.scale_

    return x_train, x_test, mean, std

def stdTransformFixed(x_train, x_test, stdTransPar):
    means = np.asarray( stdTransPar[0] )
    stds  = np.asarray( stdTransPar[1] )
    x_dummy_m = (means - stds ).tolist()
    x_dummy_p = (means + stds ).tolist()
    x_dummy =  np.asarray( [x_dummy_m, x_dummy_p] )

    Transformer = preprocessing.StandardScaler()
    Transformer.fit(x_dummy)
    x_train = Transformer.transform(x_train)
    x_test = Transformer.transform(x_test)

    return x_train, x_test
