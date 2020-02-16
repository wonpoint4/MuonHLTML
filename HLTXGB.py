import sys
import multiprocessing
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

    print(seedname + r' C0: %d, C1: %d, C2: %d, C3: %d' % ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train)
    dtest = xgb.DMatrix(x_test, weight=y_wgtsTest, label=y_test)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {'max_depth':6, 'eta':0.1, 'gamma':10, 'objective':'multi:softmax', 'num_class': 4, 'subsample':0.5, 'eval_metric':'mlogloss','lambda':2.5}
    param['min_child_weight'] = np.sum(y_wgtsTrain)/1000.
    param['tree_method'] = 'exact'
    param['nthread'] = 1

    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    dTrainPredict = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
    dTestPredict = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    confMat = postprocess.confMat(dTestPredict,y_test)
    vis.drawConfMat(confMat,seedname+'_testConfMat')

    confMatTrain = postprocess.confMat(dTrainPredict,y_train)
    vis.drawConfMat(confMatTrain,seedname+'_trainConfMat')

    return

def run(seedname):
    seed = IO.readMinSeeds('/home/common/MuGunPU200_seedNtuple/ntuple_*.root','seedNtupler/'+seedname,0.,99999.)
    doXGB(seed,seedname)

seedlist = ['NThltIterL3OI','NThltIter0','NThltIter2','NThltIter3','NThltIter0FromL1','NThltIter2FromL1','NThltIter3FromL1']

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=7)
    pool.map(run,seedlist)
    pool.close()
    pool.join()

print('Finished')
