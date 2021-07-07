import sys
import glob
import numpy as np
import ROOT
from HLTIO import preprocess
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path
import math
import pandas as pd
import json
import time
import pickle

# IO (Require ROOT version > 6.14)
def dR(eta1, phi1, eta2, phi2):
    dr = math.sqrt((eta1-eta2)*(eta1-eta2) + (phi1-phi2)*(phi1-phi2))
    return dr

def setEtaPhi(x, y, z):
    perp = math.sqrt(x*x + y*y)
    eta = np.arcsinh(z/perp)
    phi = np.arccos(x/perp)
    return eta, phi

def dphi(phi1, phi2):
    tmpdphi = math.fabs(phi1-phi2)
    if tmpdphi >= math.pi:
        tmpdphi = 2*math.pi - tmpdphi
    return tmpdphi

def Read(path,varlist):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("tree")

    mtx = t.AsMatrix(varlist)

    return mtx

def treeToDf(tree):
    npArr, cols = tree.AsMatrix(return_labels=True)
    df = pd.DataFrame(data=npArr, columns=cols)

    return df

def readSeedTree(path,treePath,minpt,maxpt,isB):
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    tree = f.Get(treePath)
    df = treeToDf(tree)

    df = df[ df['gen_pt'] < maxpt ]
    df = df[ df['gen_pt'] > minpt ]
    if isB:
        df = df[ ( (df['tsos_eta'] < 1.2) & (df['tsos_eta'] > -1.2) ) ]
    else:
        df = df[ ( (df['tsos_eta'] > 1.2) | (df['tsos_eta'] < -1.2) ) ]

    return preprocess.getNclass(df)

def readMinSeeds(dir,treePath,minpt,maxpt,isB):
    if(isB) : tag = 'Barrel'
    else : tag = 'Endcap'
    filelist = glob.glob(dir)
    ## If 'dir' argument has pkl, then instead of reading root, read pre-defined pickle (much faster)
    if '/home/wjun/MuonHLTML/data/'+tag+'_'+treePath.replace('/','_')+'.pkl' in filelist :
        f = open('data/'+tag+'_'+treePath.replace('/','_')+'.pkl', "rb")
        seedData = pickle.load(f)
        print('data/'+tag+'_'+treePath.replace('/','_')+'.pkl is loaded')
        f.close()
        return seedData

    full = pd.DataFrame()
    y = np.array([]).reshape(0,)
    n = np.array([0,0])
    cut = 1e6

    nfile = 0
    startTime = time.time()
    for path in filelist:
        print('Processing %dth file %s ...' % (nfile, path) )
        if np.all( n >= cut ):
            continue

        bkg, sig = readSeedTree(path,treePath,minpt,maxpt,isB)
        subset = pd.DataFrame()
        n_ = np.array([0,0])#,0,0])
        y_ = np.array([]).reshape(0,)
        if n[0] < cut:
            subset = subset.append(bkg,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(bkg.shape[0],0) ) )
            n_[0] = bkg.shape[0]
        if n[1] < cut:
            subset = subset.append(sig,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(sig.shape[0],1) ) )
            n_[1] = sig.shape[0]

        full = full.append(subset, ignore_index=True)
        n += n_
        y = np.hstack( (y,y_) )

        if "L1" in treePath:
            isL1 = True
        else:
            isL1 = False
        full = preprocess.filterClass(full, isL1)

        nfile = nfile+1

    endTime = time.time()
    print(tag+' : '+treePath + ' | %d/%d files | (notBuilt + combi + simMatched, muMatched) = (%d, %d) seeds added, This took %d seconds' %(nfile, len(filelist), n[0], n[1], (endTime-startTime)))
    f = open('data/'+tag+'_'+treePath.replace('/','_')+'.pkl', "wb")
    pickle.dump((full, y), f)
    f.close()
    print('data/'+tag+'_'+treePath.replace('/','_')+'.pkl is saved')
    return full, y

def dumpsvm(x, y, filename):
    dump_svmlight_file(x, y, filename, zero_based=True)

    return

def loadsvm(filepath):
    x, y = load_svmlight_file(filepath)
    x = x.toarray()

    return x, y

def maketest(mu,sigma,name):
    testfile = ROOT.TFile("./data/test"+name+".root","RECREATE")
    tree = ROOT.TTree("tree","test")
    v1 = np.empty((1), dtype="float32")
    v2 = np.empty((1), dtype="float32")
    v3 = np.empty((1), dtype="float32")
    v4 = np.empty((1), dtype="float32")
    v5 = np.empty((1), dtype="float32")
    tree.Branch("v1",v1,"v1/F")
    tree.Branch("v2",v2,"v2/F")
    tree.Branch("v3",v3,"v3/F")
    tree.Branch("v4",v4,"v4/F")
    tree.Branch("v5",v5,"v5/F")

    for i in range(10000):
        v1[0] = np.random.normal(mu,sigma,1)
        v2[0] = np.random.normal(mu,sigma,1)
        v3[0] = np.random.normal(mu,sigma,1)
        v4[0] = np.random.normal(mu,sigma,1)
        v5[0] = np.random.normal(mu,sigma,1)
        tree.Fill()
    testfile.Write()
    testfile.Close()

    return

def print_params(string, param_json):
    print(string)
    params = json.loads(param_json)
    print(json.dumps(params, indent=4, sort_keys=True))

    return
