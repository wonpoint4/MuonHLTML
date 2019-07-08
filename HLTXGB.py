import sys
import numpy as np
import time
import xgboost as xgb

# Training
print("Training ...")
start = time.time()

dtrain = xgb.DMatrix('./data/testTrain.svm')
dtest = xgb.DMatrix('./data/testTest.svm')
evallist = [(dtest, 'eval'), (dtrain, 'train')]

param = {'max_depth':3, 'eta':0.1, 'gamma':10, 'min_child_weight':1e2, 'silent':0,
'objective':'binary:logistic', 'subsample':0.5, 'eval_metric':'aucpr', 'tree_method':'exact'}
num_round = 100

bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)
# bst.save_model('./model/testXGB.model')

dTrainPredict = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)
dTestPredict = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

end = time.time()

print(r'Elapsed time = %4.2f' % (end-start))
print('Finished')
