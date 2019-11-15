import sys
import os, errno
import numpy as np
import ROOT
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path
import math
from sklearn.manifold import TSNE
from HLTvis import vis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import xgboost as xgb


def is_outlier(points, thresh=3.5):

    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def readSeedNtupleBase(path, seedtype):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("seedNtupler/"+seedtype)

    try:
        if not(os.path.isdir('./seed_plots')):
            os.makedirs(os.path.join('./seed_plots'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    try:
        if not(os.path.isdir('./'+'seed_npz')):
            os.makedirs(os.path.join('./'+'seed_npz'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    sig = []
    bkg1 = []
    bkg2 = []
    bkg3 = []

    # bkg quality gets bad as number increases

    n_bkg1 = 0
    n_bkg2 = 0
    n_bkg3 = 0

    n_cut = 30000
    n_cut_sample = 10000

    for iseed in t:

        arr = []

        if iseed.matchedTPsize < 0 : n_bkg3 += 1
        if iseed.matchedTPsize == 0 : n_bkg2 += 1
        if iseed.matchedTPsize == 1 and abs(iseed.bestMatchTP_pdgId) == 13 : n_bkg1 += 1
        if n_bkg3 > n_cut and iseed.matchedTPsize < 0 : continue
        if n_bkg2 > n_cut and iseed.matchedTPsize == 0 : continue
        if n_bkg1 > n_cut and iseed.matchedTPsize == 1 and abs(iseed.bestMatchTP_pdgId) == 13: continue

        # arr.append(np.asarray(iseed.dir,np.int32))
        # arr.append(np.asarray(iseed.tsos_detId,np.uint32))
        arr.append(np.asarray(iseed.tsos_pt,np.float32))
        # arr.append(np.asarray(iseed.tsos_hasErr,np.int32))
        arr.append(np.asarray(iseed.tsos_err0,np.float32))
        arr.append(np.asarray(iseed.tsos_err1,np.float32))
        arr.append(np.asarray(iseed.tsos_err2,np.float32))
        arr.append(np.asarray(iseed.tsos_err3,np.float32))
        arr.append(np.asarray(iseed.tsos_err4,np.float32))
        arr.append(np.asarray(iseed.tsos_err5,np.float32))
        arr.append(np.asarray(iseed.tsos_err6,np.float32))
        arr.append(np.asarray(iseed.tsos_err7,np.float32))
        arr.append(np.asarray(iseed.tsos_err8,np.float32))
        arr.append(np.asarray(iseed.tsos_err9,np.float32))
        arr.append(np.asarray(iseed.tsos_err10,np.float32))
        arr.append(np.asarray(iseed.tsos_err11,np.float32))
        arr.append(np.asarray(iseed.tsos_err12,np.float32))
        arr.append(np.asarray(iseed.tsos_err13,np.float32))
        arr.append(np.asarray(iseed.tsos_err14,np.float32))
        arr.append(np.asarray(iseed.tsos_x,np.float32))
        arr.append(np.asarray(iseed.tsos_y,np.float32))
        arr.append(np.asarray(iseed.tsos_dxdz,np.float32))
        arr.append(np.asarray(iseed.tsos_dydz,np.float32))
        arr.append(np.asarray(iseed.tsos_px,np.float32))
        arr.append(np.asarray(iseed.tsos_py,np.float32))
        arr.append(np.asarray(iseed.tsos_qbp,np.float32))
        # arr.append(np.asarray(iseed.tsos_charge,np.int32))
        arr.append(np.asarray(iseed.dR_minDRL1SeedP,np.float32))
        arr.append(np.asarray(iseed.dPhi_minDRL1SeedP,np.float32))
        arr.append(np.asarray(iseed.dR_minDPhiL1SeedX,np.float32))
        arr.append(np.asarray(iseed.dPhi_minDPhiL1SeedX,np.float32))

        seed = np.hstack(arr)
        if iseed.matchedTPsize == 1 and abs(iseed.bestMatchTP_pdgId)==13: sig.append(seed)
        if iseed.matchedTPsize == 1 and abs(iseed.bestMatchTP_pdgId)!=13: bkg1.append(seed)
        if iseed.matchedTPsize == 0                                     : bkg2.append(seed)
        if iseed.matchedTPsize < 0                                      : bkg3.append(seed)

    sig = np.vstack(sig)
    bkg1 = np.vstack(bkg1) # FIXME : if #bkg is 0 concatenation will fail
    bkg2 = np.vstack(bkg2) # FIXME : if #bkg is 0 concatenation will fail
    bkg3 = np.vstack(bkg3)

    sig = pd.DataFrame(sig)
    bkg1 = pd.DataFrame(bkg1)
    bkg2 = pd.DataFrame(bkg2)
    bkg3 = pd.DataFrame(bkg3)

    if sig.shape[0]  > n_cut_sample: sig = sig.sample(n_cut_sample)
    if bkg1.shape[0] > n_cut_sample: bkg1 = bkg1.sample(n_cut_sample)
    if bkg2.shape[0] > n_cut_sample: bkg2 = bkg2.sample(n_cut_sample)
    if bkg3.shape[0] > n_cut_sample: bkg3 = bkg3.sample(n_cut_sample)

    np.savez('./seed_npz/'+seedtype, sig=sig, bkg1=bkg1, bkg2=bkg2, bkg3=bkg3)

    print(sig.shape[0])
    print(bkg1.shape[0])
    print(bkg2.shape[0])
    print(bkg3.shape[0])

    return sig, bkg1, bkg2, bkg3

def readSeedNtuple(path, seedtype):

    checkfile = Path('./seed_npz/'+seedtype+'.npz')
    try:
        checkfile.resolve()
    except FileNotFoundError:
        print('here1')

        sig, bkg1, bkg2, bkg3 = readSeedNtupleBase(path, seedtype)
        return sig, bkg1, bkg2, bkg3

    else:
        print('here2')
        np_load = np.load('./seed_npz/'+seedtype+'.npz')
        sig = np_load['sig']
        bkg1 = np_load['bkg1']
        bkg2 = np_load['bkg2']
        bkg3 = np_load['bkg3']
        return sig, bkg1, bkg2, bkg3

def draw_varplots(path, seedtype):

    sig, bkg1, bkg2, bkg3 = readSeedNtuple(path, seedtype)    

    for i in range(27):

        idx = str(i)
        vis.histoverlay(sig[i][~is_outlier(sig[i])], bkg1[i][~is_outlier(bkg1[i])], bkg2[i][~is_outlier(bkg2[i])], bkg3[i][~is_outlier(bkg3[i])], seedtype, idx)

    return None

def do_tsne(path, seedtype):

    try:
        if not(os.path.isdir('./seed_csv')):
            os.makedirs(os.path.join('./seed_csv'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    sig, bkg1, bkg2, bkg3 = readSeedNtuple(path, seedtype)

    sig = pd.DataFrame(sig)
    bkg1 = pd.DataFrame(bkg1)
    bkg2 = pd.DataFrame(bkg2)
    bkg3 = pd.DataFrame(bkg3)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
    tsne_sig = tsne.fit_transform(sig)
    tsne_bkg1 = tsne.fit_transform(bkg1)
    tsne_bkg2 = tsne.fit_transform(bkg2)
    tsne_bkg3 = tsne.fit_transform(bkg3)

    sig['tsne-x'] = tsne_sig[:,0]
    sig['tsne-y'] = tsne_sig[:,1]
    bkg1['tsne-x'] = tsne_bkg1[:,0]
    bkg1['tsne-y'] = tsne_bkg1[:,1]
    bkg2['tsne-x'] = tsne_bkg2[:,0]
    bkg2['tsne-y'] = tsne_bkg2[:,1]
    bkg3['tsne-x'] = tsne_bkg3[:,0]
    bkg3['tsne-y'] = tsne_bkg3[:,1]

    sig.to_csv('./seed_csv/'+seedtype+'sig.csv',index=None,header=True)
    bkg1.to_csv('./seed_csv/'+seedtype+'bkg1.csv',index=None,header=True)
    bkg2.to_csv('./seed_csv/'+seedtype+'bkg2.csv',index=None,header=True)
    bkg3.to_csv('./seed_csv/'+seedtype+'bkg3.csv',index=None,header=True)

    vis.hist2dSig(sig[['tsne-x','tsne-y']].values,'t-sne'+seedtype+'Sig')
    vis.hist2dBkg1(bkg1[['tsne-x','tsne-y']].values,'t-sne'+seedtype+'Bkg1')
    vis.hist2dBkg2(bkg2[['tsne-x','tsne-y']].values,'t-sne'+seedtype+'Bkg2')
    vis.hist2dBkg3(bkg3[['tsne-x','tsne-y']].values,'t-sne'+seedtype+'Bkg3')

    return None

def do_XGB(path, seedtype):

    sig, bkg1, bkg2, bkg3 = readSeedNtuple(path, seedtype)

    y_sig = np.full((sig.shape[0],1), 0)
    y_bkg1 = np.full((bkg1.shape[0],1), 1)
    y_bkg2 = np.full((bkg2.shape[0],1), 2)
    y_bkg3 = np.full((bkg3.shape[0],1), 3)

    x = np.concatenate((sig, bkg1, bkg2, bkg3), axis=0)
    y = np.concatenate((y_sig, y_bkg1, y_bkg2, y_bkg3), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

    Transformer = preprocessing.StandardScaler()
    x_train = Transformer.fit_transform(x_train)
    x_test = Transformer.transform(x_test)
    sig = Transformer.transform(sig)
    bkg1 = Transformer.transform(bkg1)
    bkg2 = Transformer.transform(bkg2)
    bkg3 = Transformer.transform(bkg3)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    dsig = xgb.DMatrix(sig, label=y_sig)
    dbkg1 = xgb.DMatrix(bkg1, label=y_bkg1)
    dbkg2 = xgb.DMatrix(bkg2, label=y_bkg2)
    dbkg3 = xgb.DMatrix(bkg3, label=y_bkg3)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth':4, 'eta':0.1, 'gamma':10, 'objective':'multi:softmax', 'num_class': 4, 'subsample':0.5, 'eval_metric':'mlogloss','lambda':2.5}
    param['tree_method'] = 'exact'

    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    dSigPredict = bst.predict(dsig, ntree_limit=bst.best_ntree_limit)
    dBkg1Predict = bst.predict(dbkg1, ntree_limit=bst.best_ntree_limit)
    dBkg2Predict = bst.predict(dbkg2, ntree_limit=bst.best_ntree_limit)
    dBkg3Predict = bst.predict(dbkg3, ntree_limit=bst.best_ntree_limit)

    vis.drawScore(dSigPredict, dBkg1Predict, dBkg2Predict, dBkg3Predict, seedtype)

    return None

# usage : python3 SeedNtuple_analyzer.py "NThltIter0" >&NThltIter0.log&
do_XGB("./hadd_PU200_10k.root", sys.argv[1]) 
# do_tsne("./hadd_PU200_10k.root", sys.argv[1])
# draw_varplots("./hadd_PU200_10k.root", sys.argv[1])

