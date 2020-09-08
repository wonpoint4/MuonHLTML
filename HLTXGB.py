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

def doXGB(version, seed, seedname, tag, doLoad, stdTransPar=None):
    plotdir = 'plot_'+version
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    colname = list(seed[0].columns)
    print(colname)

    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1])

    if stdTransPar==None:
        x_train, x_test, x_mean, x_std = preprocess.stdTransform(x_train, x_test)
        if not os.path.isdir('scalefiles_'+version):
            os.makedirs('scalefiles_'+version)
        with open("scalefiles_%s/%s_%s_scale.txt" % (version, tag, seedname), "w") as f_scale:
            f_scale.write( "%s_%s_%s_ScaleMean = %s\n" % (version, tag, seedname, str(x_mean.tolist())) )
            f_scale.write( "%s_%s_%s_ScaleStd  = %s\n" % (version, tag, seedname, str(x_std.tolist())) )
            f_scale.close()
    else:
        x_train, x_test = preprocess.stdTransformFixed(x_train, x_test, stdTransPar)

    y_wgtsTrain, y_wgtsTest, wgts = preprocess.computeClassWgt(y_train, y_test)

    print(seedname+"|"+tag + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
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
        bst.load_model('model_'+version+'/'+version+'_'+tag+'_'+seedname+'.model')
    else:
        if not os.path.isdir('model_'+version):
            os.makedirs('model_'+version)
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50, verbose_eval=100)
        bst.save_model('model_'+version+'/'+version+'_'+tag+'_'+seedname+'.model')

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
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, tag+'_'+seedname+r'_ROC1_cat%d' % cat, plotdir)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, tag+'_'+seedname+r'_ROC2_cat%d' % cat, plotdir)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  tag+'_'+seedname+r'_Thr1_cat%d' % cat, plotdir)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  tag+'_'+seedname+r'_Thr2_cat%d' % cat, plotdir)

    confMat, confMatAbs = postprocess.confMat(y_test,labelTest)
    vis.drawConfMat(confMat,   tag+'_'+seedname+'_testConfMatNorm', plotdir)
    vis.drawConfMat(confMatAbs,tag+'_'+seedname+'_testConfMat', plotdir, doNorm = False)

    confMatTrain, confMatTrainAbs = postprocess.confMat(y_train,labelTrain)
    vis.drawConfMat(confMatTrain,   tag+'_'+seedname+'_trainConfMatNorm', plotdir)
    vis.drawConfMat(confMatTrainAbs,tag+'_'+seedname+'_trainConfMat', plotdir, doNorm = False)

    if not doLoad:
        gain = bst.get_score( importance_type='gain')
        cover = bst.get_score(importance_type='cover')
        vis.drawImportance(gain,cover,colname,tag+'_'+seedname+'_importance', plotdir)

    return

def run_quick(seedname):
    doLoad = False

    ntuple_path = '/home/msoh/MuonHLTML_Run3/data/ntuple_81.root'

    tag = 'TESTBarrel'
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,True)
    doXGB('vTEST',seed,seedname,tag,doLoad)

    tag = 'TESTEndcap'
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,False)
    doXGB('vTEST',seed,seedname,tag,doLoad)

def run(version, seedname, tag):
    doLoad = False
    isB = ('Barrel' in tag)

    ntuple_path = '/home/msoh/MuonHLTML_Run3/data/ntuple_81.root'
    # ntuple_path = '/home/common/DY_seedNtuple_v20200510/ntuple_*.root'

    print("\n\nStart: %s|%s" % (seedname, tag))
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,isB)
    doXGB(version, seed, seedname, tag, doLoad)


VER = 'Run3v0'
seedlist = ['NThltIterL3OI','NThltIter0','NThltIter2','NThltIter3','NThltIter0FromL1','NThltIter2FromL1','NThltIter3FromL1']
seedlist = ['NThltIter2FromL1']
taglist  = ['Barrel','Endcap']
seed_run_list = [ (VER, seed, tag) for tag in taglist for seed in seedlist ]

if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    run_quick('NThltIter2FromL1')

    # pool = multiprocessing.Pool(processes=14)
    # pool.starmap(run,seed_run_list)
    # pool.close()
    # pool.join()

print('Finished')
