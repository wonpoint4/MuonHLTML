import sys
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from HLTvis import postprocess
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# gpu_id = '0' #sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

def doXGB(seed,seedname):
    print(seedname)

    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1])
    x_train, x_test = preprocess.stdTransform(x_train, x_test)
    y_wgtsTrain, y_wgtsTest = preprocess.computeClassWgt(y_train, y_test)

    print(r'C0: %d, C1: %d, C2: %d, C3: %d' % ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train)
    dtest = xgb.DMatrix(x_test, weight=y_wgtsTest, label=y_test)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {'max_depth':6, 'eta':0.1, 'gamma':10, 'objective':'multi:softmax', 'num_class': 4, 'subsample':0.5, 'eval_metric':'mlogloss','lambda':2.5}
    param['min_child_weight'] = np.sum(y_wgtsTrain)/1000.
    param['tree_method'] = 'exact'

    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    dTrainPredict = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
    dTestPredict = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    confMat = postprocess.confMat(dTestPredict,y_test)
    vis.drawConfMat(confMat,seedname+'_testConfMat')

    confMatTrain = postprocess.confMat(dTrainPredict,y_train)
    vis.drawConfMat(confMatTrain,seedname+'_trainConfMat')

    return

seedsOI = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIterL3OI',0.,99999.)
doXGB(seedsOI,'OI')
seedsIOL2quad = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter0',0.,99999.)
doXGB(seedsIOL2quad,'IOL2quad')
seedsIOL2tri = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter2',0.,99999.)
doXGB(seedsIOL2tri,'IOL2tri')
seedsIOL2doub = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter3',0.,99999.)
doXGB(seedsIOL2doub,'IOL2doub')
seedsIOL1quad = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter0FromL1',0.,99999.)
doXGB(seedsIOL1quad,'IOL1quad')
seedsIOL1tri = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter2FromL1',0.,99999.)
doXGB(seedsIOL1tri,'IOL1tri')
seedsIOL1doub = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/NThltIter3FromL1',0.,99999.)
doXGB(seedsIOL1doub,'IOL1doub')
