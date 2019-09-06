import sys
import numpy as np
import matplotlib.pyplot as plt
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

def drawROC(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, plotname):
    plt.plot(fpr_Train, tpr_Train, color='r', label='Train ROC (AUC = %.4f)' % AUC_Train)
    plt.plot(fpr_Test, tpr_Test, color='b', label='Test ROC (AUC = %.4f)' % AUC_Test)
    plt.xscale('log')
    plt.xlim(xmin=1e-4, xmax=1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, '+plotname)
    plt.legend(loc='lower right')
    plt.grid()

    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return
