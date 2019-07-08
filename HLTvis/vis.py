import sys
import numpy as np
import matplotlib.pyplot as plt
from HLTvis import postprocess

def scatter2d(x, y, plotname):
    sig, bkg = postprocess.maskTarget(x, y)
    plt.figure(figsize=(6,4))
    plt.scatter(sig[:,0],sig[:,1],c='b',label='sig',alpha=0.1)
    plt.scatter(bkg[:,0],bkg[:,1],c='r',label='bkg',alpha=0.1)
    plt.legend()
    plt.grid()
    plt.draw()
    plt.savefig('./plot/'+plotname+'.png',dpi=300, bbox_inches='tight')

    return
