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
import importlib
import time
import os
# gpu_id = '0' #sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

#def doXGB(version, seed, seedname, tag, doLoad, stdTransPar=None):
def doXGB(version, seed, seedname, tag, eta, gamma, lamda, subsample, doLoad, stdTransPar=None):
    #plotdir = 'plot_'+version
    #version = version + "_10depth_"+str(eta)+"_"+str(gamma)+"_"+str(lamda)+"_"+str(subsample)
    version = version + "_"+str(eta)+"depth_"+str(gamma)+"_"+str(lamda)+"_"+str(subsample)
    plotdir = 'gridsearch_New2_reproduce/plot_'+version
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    colname = list(seed[0].columns)
    print(colname)
    #print(seedname+"|"+tag + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
    #    ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )
    print(version+" : "+seedname+"|"+tag + r' C0: %d, C1: %d' %( (seed[1]==0).sum(), (seed[1]==1).sum() ) )

    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1])

    if doLoad and stdTransPar==None:
        print("doLoad is True but stdTransPar==None --> return")
        return

    if stdTransPar==None:
        x_train, x_test, x_mean, x_std = preprocess.stdTransform(x_train, x_test)
        #with open("scalefiles/%s_%s_%s_scale.txt" % (version, tag, seedname), "w") as f_scale:
        with open("gridsearch_New2_reproduce/scalefiles/%s_%s_%s_scale.txt" % (version, tag, seedname), "w") as f_scale:
            f_scale.write( "%s_%s_%s_ScaleMean = %s\n" % (version, tag, seedname, str(x_mean.tolist())) )
            f_scale.write( "%s_%s_%s_ScaleStd  = %s\n" % (version, tag, seedname, str(x_std.tolist())) )
            f_scale.close()
    else:
        x_train, x_test = preprocess.stdTransformFixed(x_train, x_test, stdTransPar)

    y_wgtsTrain, y_wgtsTest, wgts = preprocess.computeClassWgt(y_train, y_test)

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train, feature_names=colname)
    dtest  = xgb.DMatrix(x_test,  weight=y_wgtsTest,  label=y_test,  feature_names=colname)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    param = {
        'max_depth':10,#6,
        'eta':0.1,#0.1,
        'gamma':10,#10,
        'objective':'binary:logistic',
        'eval_metric':'logloss',
	#'eval_metric':'error@0.34',
        #'max_depth':6,
        #'eta':0.1,
        #'gamma':10,
        #'objective':'multi:softprob',
        #'num_class': 4,
        #'eval_metric':'mlogloss',
        'subsample':0.5,
        'lambda':2.5
    }
    param['min_child_weight'] = np.sum(y_wgtsTrain)/50.
    param['tree_method'] = 'exact'
    param['nthread'] = 4

    num_round = 2000#500

    bst = xgb.Booster(param)

    if doLoad:
        bst.load_model('model/'+version+'_'+tag+'_'+seedname+'.model')
    else:
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=150, verbose_eval=100)
        param = { 'eta':0.03,'gamma':gamma,'eval_metric' : f'error@{lamda}','subsample':0.9, 'alpha':1, 'lambda':100,'max_depth':eta, 'min_child_weight':subsample }
        print("Change eval_metric, and training once more")
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=100)
        bst.save_model('model/'+version+'_'+tag+'_'+seedname+'.model')

    dTrainPredict    = bst.predict(dtrain)
    dTestPredict     = bst.predict(dtest)

    dTrainPredictRaw = bst.predict(dtrain, output_margin=True)
    dTestPredictRaw  = bst.predict(dtest,  output_margin=True)

    #labelTrain       = postprocess.softmaxLabel(dTrainPredict)
    #labelTest        = postprocess.softmaxLabel(dTestPredict)
    labelTrain       = postprocess.binaryLabel(dTrainPredict)
    labelTest        = postprocess.binaryLabel(dTestPredict)
    #print(dTestPredict)
    #print(labelTest)
    # -- ROC -- #
    #for cat in range(4):
    for cat in range(1,2):
        #if ( np.asarray(y_train==cat,dtype=np.int).sum() < 2 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 2 ): continue
        if ( np.asarray(y_train==cat,dtype=np.int).sum() < 1 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 1 ): continue

        fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
            #dTrainPredict[:,cat],
            #dTestPredict[:,cat],
            dTrainPredict,
            dTestPredict,
            np.asarray(y_train==cat,dtype=np.int),
            np.asarray(y_test==cat, dtype=np.int)
        )
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_logROC_cat%d' % cat, plotdir)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_linROC_cat%d' % cat, plotdir)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_logThr_cat%d' % cat, plotdir)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_linThr_cat%d' % cat, plotdir)

        #fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
        #    postprocess.sigmoid( dTrainPredictRaw[:,cat] ),
        #    postprocess.sigmoid( dTestPredictRaw[:,cat] ),
        #    np.asarray(y_train==cat,dtype=np.int),
        #    np.asarray(y_test==cat, dtype=np.int)
        #)
        #vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_logROCSigm_cat%d' % cat, plotdir)
        #vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_linROCSigm_cat%d' % cat, plotdir)
        #vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_logThrSigm_cat%d' % cat, plotdir)
        #vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_linThrSigm_cat%d' % cat, plotdir)
    # -- ROC -- #

    # -- Confusion matrix -- #
    confMat, confMatAbs = postprocess.confMat(y_test,labelTest)
    vis.drawConfMat(confMat,   version+'_'+tag+'_'+seedname+'_testConfMatNorm', plotdir)
    vis.drawConfMat(confMatAbs,version+'_'+tag+'_'+seedname+'_testConfMat', plotdir, doNorm = False)

    confMatTrain, confMatTrainAbs = postprocess.confMat(y_train,labelTrain)
    vis.drawConfMat(confMatTrain,   version+'_'+tag+'_'+seedname+'_trainConfMatNorm', plotdir)
    vis.drawConfMat(confMatTrainAbs,version+'_'+tag+'_'+seedname+'_trainConfMat', plotdir, doNorm = False)
    # -- #

    # -- Score -- #
    '''
    TrainScoreCat3 = dTrainPredict[:,3]
    TestScoreCat3  = dTestPredict[:,3]

    TrainScoreCat3Sig = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScore(TrainScoreCat3Sig, TrainScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_trainScore_cat3', plotdir)

    TestScoreCat3Sig = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScore(TestScoreCat3Sig, TestScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_testScore_cat3', plotdir)

    TrainScoreCat3 = postprocess.sigmoid( dTrainPredictRaw[:,3] )
    TestScoreCat3  = postprocess.sigmoid( dTestPredictRaw[:,3] )

    TrainScoreCat3Sig = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScore(TrainScoreCat3Sig, TrainScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_trainScoreSigm_cat3', plotdir)

    TestScoreCat3Sig = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScore(TestScoreCat3Sig, TestScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_testScoreSigm_cat3', plotdir)

    TrainScoreCat3 = dTrainPredictRaw[:,3]
    TestScoreCat3  = dTestPredictRaw[:,3]

    TrainScoreCat3Sig = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScoreRaw(TrainScoreCat3Sig, TrainScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_trainScoreRaw_cat3', plotdir)

    TestScoreCat3Sig = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScoreRaw(TestScoreCat3Sig, TestScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_testScoreRaw_cat3', plotdir)
    '''
    TrainScoreCat = dTrainPredict
    TestScoreCat  = dTestPredict

    TrainScoreCatSig = np.array( [ score for i, score in enumerate(TrainScoreCat) if y_train[i]==1 ] )
    TrainScoreCatBkg = np.array( [ score for i, score in enumerate(TrainScoreCat) if y_train[i]!=1 ] )
    vis.drawScore(TrainScoreCatSig, TrainScoreCatBkg, version+'_'+tag+'_'+seedname+r'_trainScore', plotdir)

    TestScoreCatSig = np.array( [ score for i, score in enumerate(TestScoreCat) if y_test[i]==1 ] )
    TestScoreCatBkg = np.array( [ score for i, score in enumerate(TestScoreCat) if y_test[i]!=1 ] )
    vis.drawScore(TestScoreCatSig, TestScoreCatBkg, version+'_'+tag+'_'+seedname+r'_testScore', plotdir)

    #TrainScoreCat = postprocess.sigmoid( dTrainPredictRaw[:,1] )
    #TestScoreCat  = postprocess.sigmoid( dTestPredictRaw[:,1] )

    #TrainScoreCat3Sig = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    #TrainScoreCat3Bkg = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    #vis.drawScore(TrainScoreCat3Sig, TrainScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_trainScoreSigm_cat3', plotdir)

    #TestScoreCat3Sig = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    #TestScoreCat3Bkg = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    #vis.drawScore(TestScoreCat3Sig, TestScoreCat3Bkg, version+'_'+tag+'_'+seedname+r'_testScoreSigm_cat3', plotdir)

    TrainScoreCat = dTrainPredictRaw
    TestScoreCat  = dTestPredictRaw

    TrainScoreCatSig = np.array( [ score for i, score in enumerate(TrainScoreCat) if y_train[i]==1 ] )
    TrainScoreCatBkg = np.array( [ score for i, score in enumerate(TrainScoreCat) if y_train[i]!=1 ] )
    vis.drawScoreRaw(TrainScoreCatSig, TrainScoreCatBkg, version+'_'+tag+'_'+seedname+r'_trainScoreRaw', plotdir)

    TestScoreCatSig = np.array( [ score for i, score in enumerate(TestScoreCat) if y_test[i]==1 ] )
    TestScoreCatBkg = np.array( [ score for i, score in enumerate(TestScoreCat) if y_test[i]!=1 ] )
    vis.drawScoreRaw(TestScoreCatSig, TestScoreCatBkg, version+'_'+tag+'_'+seedname+r'_testScoreRaw', plotdir)
    # -- #

    # -- Importance -- #
    if not doLoad:
        gain = bst.get_score( importance_type='gain')
        cover = bst.get_score(importance_type='cover')
        vis.drawImportance(gain,cover,colname,version+'_'+tag+'_'+seedname+'_importance', plotdir)
    # -- #

    return

def run_quick(seedname):
    doLoad = False

    ntuple_path = '/home/wjun/MuonHLTML/data/v2_RAW_L2Recover/ntuple_Run3v3_WP0p00_1.root'

    tag = 'TESTBarrel'
    print("\n\nStart: %s|%s" % (seedname, tag))
    stdTrans = None
    if doLoad:
        scalefile = importlib.import_module("scalefiles."+tag+"_"+seedname+"_scale")
        scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleMean")
        scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleStd")
        stdTrans = [ scaleMean, scaleStd ]
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,True)
    doXGB('vTEST',seed,seedname,tag,doLoad,stdTrans)
    '''
    tag = 'TESTEndcap'
    print("\n\nStart: %s|%s" % (seedname, tag))
    stdTrans = None
    if doLoad:
        scalefile = importlib.import_module("scalefiles."+tag+"_"+seedname+"_scale")
        scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleMean")
        scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleStd")
        stdTrans = [ scaleMean, scaleStd ]
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,False)
    doXGB('vTEST',seed,seedname,tag,doLoad)
    '''
#def run(version, seedname, tag):
def run(version, seedname, tag, eta, gamma, lammda, subsample):
    version = version + "_"+str(eta)+"_"+str(gamma)+"_"+str(lammda)+"_"+str(subsample)
    doLoad = False
    isB = ('Barrel' in tag)

    ntuple_path = '/home/wjun/MuonHLTML/data/v2_RAW_L2Recover/ntuple_*1.root'
    # ntuple_path = '/home/common/DY_seedNtuple_v20200510/ntuple_*.root'

    stdTrans = None
    if doLoad:
        scalefile = importlib.import_module("scalefiles."+tag+"_"+seedname+"_scale")
        scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleMean")
        scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleStd")
        stdTrans = [ scaleMean, scaleStd ]

    print("\n\nStart: %s|%s" % (seedname, tag))
    seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedname, 0.,99999.,isB)
    #doXGB(version, seed, seedname, tag, doLoad, stdTrans)
    doXGB(version, seed, seedname, tag, eta, gamma, lammda, subsample, doLoad, stdTrans)

startTime = time.time()
print('\nStarts at ', time.strftime('%c', time.localtime(startTime)))

VER = 'Run3v6'
#seedlist = ['NThltIterL3OI','NThltIter0','NThltIter2','NThltIter3','NThltIter0FromL1','NThltIter2FromL1','NThltIter3FromL1']
#seedlist = ['NThltIter2', 'NThltIter2FromL1']
seedtypelist = ['NThltIter2']#,'NThltIter2FromL1']
taglist  = ['Barrel', 'Endcap']

etalist = [6,10,12,13,14]
gammalist = [0.5, 1, 2]#0.001,0.01,0.1,1]
lambdalist = [0.001,0.003,0.01,0.03,0.1]
subsamplelist = [0.1, 1, 5, 10]
#seed_run_list = [ (VER, seed, tag) for tag in taglist for seed in seedlist ]
#seed_run_list = [ (VER, seedtype, tag, eta, gamma, lamda, subsample) for tag in taglist for seedtype in seedtypelist for eta in etalist for gamma in gammalist for lamda in lambdalist for subsample in subsamplelist ]

ntuple_path = '/home/wjun/MuonHLTML/data/v2_RAW_L2Recover/ntuple_*.root'

for seedtype in seedtypelist:
    for tag in taglist:
        doLoad = False
        isB = ('Barrel' in tag)

        stdTrans = None
        if doLoad:
            scalefile = importlib.import_module("scalefiles."+tag+"_"+seedtype+"_scale")
            scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedtype+"_ScaleMean")
            scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedtype+"_ScaleStd")
            stdTrans = [ scaleMean, scaleStd ]
        
        print("Start: %s|%s" % (seedtype, tag))
        seed = IO.readMinSeeds(ntuple_path, 'seedNtupler/'+seedtype, 0.,99999.,isB)

        seed_run_list = [ (VER, seed, seedtype, tag, eta, gamma, lamda, subsample, doLoad, stdTrans) for eta in etalist for gamma in gammalist for lamda in lambdalist for subsample in subsamplelist ]

        if __name__ == '__main__':
            from warnings import simplefilter
            simplefilter(action='ignore', category=FutureWarning)

            pool = multiprocessing.Pool(processes=20)
            pool.starmap(doXGB,seed_run_list)
    
pool.close()
pool.join()

'''
if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    #run_quick('NThltIter2FromL1')

    pool = multiprocessing.Pool(processes=14)
    pool.starmap(run,seed_run_list)
    pool.close()
    pool.join()
'''
endTime=time.time()
print('Finished at '+time.strftime('%c',time.localtime(endTime)))
print('Time took', endTime-startTime,'seconds.')
