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
    x_train, dfs = preprocess.stdTransform(x_train, [x_test])
    y_wgtsTrain = preprocess.computeClassWgt(y_train)
    y_wgtsTest = preprocess.computeClassWgt(y_test)

    # classWeight = float(fake.sum(axis=0))/float((~fake).sum(axis=0))
    print(r'C0: %d, C1: %d, C2: %d, C3: %d' % ( (seed[1]==0).sum(), ,(seed[1]==1).sum() ,(seed[1]==2).sum() ,(seed[1]==3).sum() ) )

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train)
    dtest = xgb.DMatrix(dfs[0], weight=y_wgtsTest, label=y_test)
    # dsig = xgb.DMatrix(sig, label=np.full((sig.shape[0],1), 1))
    # dbkg = xgb.DMatrix(bkg, label=np.full((bkg.shape[0],1), 0))
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth':6, 'eta':0.1, 'gamma':10, 'objective':'multi:softmax', 'num_class': 4, 'subsample':0.5, 'eval_metric':'mlogloss','lambda':2.5}
    param['min_child_weight'] = np.sum(y_wgtsTrain)/100.
    # param['scale_pos_weight'] = classWeight
    param['tree_method'] = 'exact'

    num_round = 500
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    dTrainPredict = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
    dTestPredict = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    confMat = postprocess.confMat(dTestPredict,y_test)
    vis.drawConfMat(confMat,seedname+'_testConfMat')

    confMatTrain = postprocess.confMat(dTrainPredict,y_train)
    vis.drawConfMat(confMatTrain,seedname+'_trainConfMat')
    # dSigPredict = bst.predict(dsig, ntree_limit=bst.best_ntree_limit)
    # dBkgPredict = bst.predict(dbkg, ntree_limit=bst.best_ntree_limit)

    # fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test = postprocess.calROC(dTrainPredict, dTestPredict, y_train, y_test)

    # vis.drawScore(dSigPredict, dBkgPredict, 'BDTscore_'+seedname)
    # vis.drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, 'ROC_'+seedname)
    #
    # fig = plt.figure(figsize=(6,4))
    # xgb.plot_importance(bst)
    # plt.savefig('./plot/importance_'+seedname+'.png', dpi=300, bbox_inches='tight')

    return

# seeds = IO.readSeedNp("./data/ntuple_SingleMuon2018C_Run319941_NMu1_Pt27to1000000000_PU40to60_RAWAOD.root")
#
# doXGB(seeds[0],"iterL3OISeedsFromL2Muons")
# doXGB(seeds[1],"iter0IterL3MuonPixelSeedsFromPixelTracks")
# doXGB(seeds[2],"iter2IterL3MuonPixelSeeds")
# doXGB(seeds[3],"iter3IterL3MuonPixelSeeds")
# doXGB(seeds[4],"iter0IterL3FromL1MuonPixelSeedsFromPixelTracks")
# doXGB(seeds[5],"iter2IterL3FromL1MuonPixelSeeds")
# doXGB(seeds[6],"iter3IterL3FromL1MuonPixelSeeds")

# IO.readSeedTree('./data/ntuple_PU200_0.root','seedNtupler/NThltIterL3OI')
seedsOI = IO.readMinSeeds('./data/ntuple_PU200*.root','seedNtupler/NThltIterL3OI')
doXGB(seedsOI,'OI')
