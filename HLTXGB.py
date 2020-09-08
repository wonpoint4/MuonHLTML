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

def doXGB(seed,seedname,runname,doLoad,stdTransPar=None):
    colname = list(seed[0].columns)
    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1])

    if stdTransPar==None:
        x_train, x_test, x_mean, x_std = preprocess.stdTransform(x_train, x_test)
        with open("scalefiles/%s_%s_scale.txt" % (runname, seedname), "w") as f_scale:
            f_scale.write( "%s_%s_ScaleMean = %s\n" % (runname, seedname, str(x_mean.tolist())) )
            f_scale.write( "%s_%s_ScaleStd  = %s\n" % (runname, seedname, str(x_std.tolist())) )
            f_scale.close()
    else:
        x_train, x_test = preprocess.stdTransformFixed(x_train, x_test, stdTransPar)

    y_wgtsTrain, y_wgtsTest, wgts = preprocess.computeClassWgt(y_train, y_test)

    print(seedname+"|"+runname + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
        ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train, feature_names=colname)
    dtest  = xgb.DMatrix(x_test,  weight=y_wgtsTest,  label=y_test,  feature_names=colname)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {
        'max_depth':6,
        'eta':0.1,
        'gamma':10,
        'objective':'multi:softprob',
        'num_class': 4,
        'subsample':0.5,
        'eval_metric':'mlogloss',
        'lambda':2.5
    }
    param['min_child_weight'] = np.sum(y_wgtsTrain)/50.
    param['tree_method'] = 'exact'
    param['nthread'] = 4

    num_round = 500

    bst = xgb.Booster(param)

    if doLoad:
        bst.load_model('model/'+runname+'_'+seedname+'.model')
    else:
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50, verbose_eval=100)
        bst.save_model('model/'+runname+'_'+seedname+'.model')

    dTrainPredict = bst.predict(dtrain)
    dTestPredict = bst.predict(dtest)

    labelTrain = postprocess.softmaxLabel(dTrainPredict)
    labelTest = postprocess.softmaxLabel(dTestPredict)

    for cat in range(4):
        if ( np.asarray(y_train==cat,dtype=np.int).sum() < 2 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 2 ): continue

        fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
            dTrainPredict[:,cat],
            dTestPredict[:,cat],
            np.asarray(y_train==cat,dtype=np.int),
            np.asarray(y_test==cat, dtype=np.int)
        )
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, runname+'_'+seedname+r'_ROC1_cat%d' % cat)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, runname+'_'+seedname+r'_ROC2_cat%d' % cat)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  runname+'_'+seedname+r'_Thr1_cat%d' % cat)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  runname+'_'+seedname+r'_Thr2_cat%d' % cat)

    confMat, confMatAbs = postprocess.confMat(y_test,labelTest)
    vis.drawConfMat(confMat,   runname+'_'+seedname+'_testConfMatNorm')
    vis.drawConfMat(confMatAbs,runname+'_'+seedname+'_testConfMat', doNorm = False)

    confMatTrain, confMatTrainAbs = postprocess.confMat(y_train,labelTrain)
    vis.drawConfMat(confMatTrain,   runname+'_'+seedname+'_trainConfMatNorm')
    vis.drawConfMat(confMatTrainAbs,runname+'_'+seedname+'_trainConfMat', doNorm = False)

    if not doLoad:
        gain = bst.get_score( importance_type='gain')
        cover = bst.get_score(importance_type='cover')
        vis.drawImportance(gain,cover,colname,runname+'_'+seedname+'_importance')

    return

def run_quick(seedname):
    doLoad = False

    ntuple_path = '/home/msoh/MuonHLTML_forTest/data/ntuple_1-620.root'

    runname = 'PU200Barrel'
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,True)
    doXGB(seed,seedname,runname,doLoad)

    runname = 'PU200Endcap'
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,False)
    doXGB(seed,seedname,runname,doLoad)

def run(seedname, runname):
    doLoad = False
    isB = ('Barrel' in runname)

    # ntuple_path = '/home/msoh/MuonHLTML_forTest/data/ntuple_1-620.root'
    ntuple_path = '/home/common/DY_seedNtuple_v20200510/ntuple_*.root'

    print("\n\nStart: %s|%s" % (seedname, runname))
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,isB)
    doXGB(seed,seedname,runname,doLoad)


# seedlist = ['NThltIterL3OI','NThltIter0','NThltIter2','NThltIter3','NThltIter0FromL1','NThltIter2FromL1','NThltIter3FromL1']
seedlist = ['NThltIter0FromL1']
runlist  = ['PU200Barrel','PU200Endcap']
seed_run_list = [ (seed, run) for run in runlist for seed in seedlist ]

if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    # run_quick('NThltIter2FromL1')
    pool = multiprocessing.Pool(processes=14)
    pool.starmap(run,seed_run_list)
    pool.close()
    pool.join()

print('Finished')
