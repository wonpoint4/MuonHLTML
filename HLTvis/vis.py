import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from HLTvis import postprocess

def scatter2d(x, y, plotname):
    sig, bkg = postprocess.maskTarget(x, y)
    plt.figure(figsize=(6,4))
    plt.scatter(bkg[:,0],bkg[:,1],c='r',label='bkg',alpha=0.1)
    plt.scatter(sig[:,0],sig[:,1],c='b',label='sig',alpha=0.5)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return

def scatter2dSB(sig, bkg, plotname):
    plt.figure(figsize=(6,4))
    plt.scatter(bkg[:,0],bkg[:,1],c='r',label='bkg',alpha=0.1)
    plt.scatter(sig[:,0],sig[:,1],c='b',label='sig',alpha=0.5)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return

def hist2d(idx,data, plotname):
    if idx==0: cm=plt.cm.Reds
    if idx==1: cm=plt.cm.Oranges
    if idx==2: cm=plt.cm.Greens
    if idx==3: cm=plt.cm.Blues
    plt.figure(figsize=(6,4))
    plt.hist2d(data[:,0], data[:,1], bins=100, cmap=cm)
    plt.colorbar()
    plt.draw()
    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.close()

    return

def hist2dOverlay(dataSig0,dataSig1,dataBkg0,dataBkg1,plotname):
    plt.figure(figsize=(6,4))
    cmapBlue = plt.cm.Blues(np.arange(plt.cm.Blues.N))
    cmapBlue[:,-1] = np.linspace(0, 0.8, plt.cm.Blues.N)
    cmapBlue = ListedColormap(cmapBlue)
    cmapRed = plt.cm.Reds(np.arange(plt.cm.Reds.N))
    cmapRed[:,-1] = np.linspace(0, 0.8, plt.cm.Reds.N)
    cmapRed = ListedColormap(cmapRed)
    cmapOrange = plt.cm.Oranges(np.arange(plt.cm.Oranges.N))
    cmapOrange[:,-1] = np.linspace(0, 0.8, plt.cm.Oranges.N)
    cmapOrange = ListedColormap(cmapOrange)
    cmapGreen = plt.cm.Greens(np.arange(plt.cm.Greens.N))
    cmapGreen[:,-1] = np.linspace(0, 0.8, plt.cm.Greens.N)
    cmapGreen = ListedColormap(cmapGreen)
    plt.hist2d(dataBkg0[:,0], dataBkg0[:,1], bins=100, cmap=cmapRed, normed=True)
    plt.hist2d(dataBkg1[:,0], dataBkg1[:,1], bins=100, cmap=cmapOrange, normed=True)
    # plt.colorbar()
    plt.hist2d(dataSig0[:,0], dataSig0[:,1], bins=100, cmap=cmapGreen, normed=True)
    plt.hist2d(dataSig1[:,0], dataSig1[:,1], bins=100, cmap=cmapBlue, normed=True)
    # plt.colorbar()
    plt.draw()
    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')
    plt.close()

    return

def drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, plotname):
    plt.figure(figsize=(6,4))
    plt.plot(fpr_Train, 1-tpr_Train, color='r', label='Train ROC (AUC = %.4f)' % AUC_Train)
    plt.plot(fpr_Test, 1-tpr_Test, color='b', label='Test ROC (AUC = %.4f)' % AUC_Test)
    plt.xlabel('False Positive Rate')
    plt.ylabel('1 - True Positive Rate')
    plt.yscale('log')
    plt.ylim(ymin=1e-3, ymax=1.0)
    plt.title(plotname)
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return

def drawScore(dSigPredict, dBkgPredict, plotname):
    plt.figure(figsize=(6,4))
    plt.hist(dSigPredict, 100, normed=True, alpha=0.5, label='Sig', range=(0,1), color='b')
    plt.hist(dBkgPredict, 100, normed=True, alpha=0.5, label='Bkg', range=(0,1), color='r')
    plt.grid()
    # plt.yscale('log')
    plt.title(plotname)
    plt.xlabel('Output')
    plt.ylabel('seeds(normed)/0.01')
    plt.legend(loc='upper right')

    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return
