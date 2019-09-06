import sys
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from HLTvis import postprocess
import xgboost as xgb

import os
gpu_id = 0 #sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

def doXGB(seed,seedname):
    seed = pd.DataFrame(seed)
    seed, fake = preprocess.dfSigBkg(seed)

    x_train, x_test, y_train, y_test = preprocess.split(seed, ~fake)
    x_train, x_test = preprocess.stdTransform(x_train, x_test)
    classWeight = float(fake.sum(axis=0))/float(~fake.sum(axis=0))

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = {'max_depth':5, 'eta':0.1, 'gamma':10, 'min_child_weight':10, 'silent':0, 'objective':'binary:logistic', 'subsample':0.5, 'eval_metric':'auc'}
    param['scale_pos_weight'] = classWeight
    param['tree_method'] = 'gpu_exact'

    num_round = 1000
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

    dTrainPredict = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
    dTestPredict = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

    fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test = postprocess.calROC(dTrainPredict, dTestPredict, y_train, y_test)

    vis.drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, 'ROC_'+seedname)

    return

seeds = IO.readSeed("./data/ntuple_HLTPhysics2018C_Run319941_RAWAOD.root")

doTSNE(seeds[0],"iterL3OISeedsFromL2Muons")
doTSNE(seeds[1],"iter0IterL3MuonPixelSeedsFromPixelTracks")
doTSNE(seeds[2],"iter2IterL3MuonPixelSeeds")
doTSNE(seeds[3],"iter3IterL3MuonPixelSeeds")
doTSNE(seeds[4],"iter0IterL3FromL1MuonPixelSeedsFromPixelTracks")
doTSNE(seeds[5],"iter2IterL3FromL1MuonPixelSeeds")
doTSNE(seeds[6],"iter3IterL3FromL1MuonPixelSeeds")
