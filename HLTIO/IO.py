import sys
import numpy as np
import ROOT
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse

# IO (Require ROOT version > 6.14)

def Read(path,varlist):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("tree")

    mtx = t.AsMatrix(varlist)

    return mtx

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
