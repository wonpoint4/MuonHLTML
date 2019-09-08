import sys
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from sklearn.manifold import TSNE
from pathlib import Path

def doTSNE(seed,seedname,filename):
    checkfile = Path('data/t-sne_'+filename+'_'+seedname+'.csv')
    try:
        checkfile.resolve()
    except FileNotFoundError:
        seed = pd.DataFrame(seed)
        seed, fake = preprocess.dfSigBkg(seed)
        seed.drop(seed.columns[[1,3,26]],axis=1,inplace=True)

        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
        tsne_result = tsne.fit_transform(seed)
        seed['tsne-x'] = tsne_result[:,0]
        seed['tsne-y'] = tsne_result[:,1]

        seed.to_csv(checkfile,index=None,header=True)
    else:
        seed = pd.read_csv(checkfile)

    # vis.scatter2dSB(seed[~fake][['tsne-x','tsne-y']].values, seed[fake][['tsne-x','tsne-y']].values, 't-sne_'+seedname)
    vis.hist2dSig(seed[~fake][['tsne-x','tsne-y']].values,'t-sneSig_'+filename+'_'+seedname)
    vis.hist2dBkg(seed[fake][['tsne-x','tsne-y']].values,'t-sneBkg_'+filename+'_'+seedname)
    vis.hist2dOverlay(seed[~fake][['tsne-x','tsne-y']].values,seed[fake][['tsne-x','tsne-y']].values,'t-sneOverlay_'+filename+'_'+seedname)

    return

# seeds = IO.readSeed("./data/ntuple_PU50.root")
filename = 'ntuple_SingleMuon2018C_Run319941_NMu1_Pt27to1000000000_PU40to60_RAWAOD'
seeds = IO.readSeedNp("./data/"+filename+".root")

doTSNE(seeds[0],"iterL3OISeedsFromL2Muons",filename)
doTSNE(seeds[1],"iter0IterL3MuonPixelSeedsFromPixelTracks",filename)
doTSNE(seeds[2],"iter2IterL3MuonPixelSeeds",filename)
doTSNE(seeds[3],"iter3IterL3MuonPixelSeeds",filename)
doTSNE(seeds[4],"iter0IterL3FromL1MuonPixelSeedsFromPixelTracks",filename)
doTSNE(seeds[5],"iter2IterL3FromL1MuonPixelSeeds",filename)
doTSNE(seeds[6],"iter3IterL3FromL1MuonPixelSeeds",filename)




# x, y = IO.loadsvm('./data/testTrain.svm')
#
# pca = PCA(n_components=2)
# pca.fit(x)
# x = pca.transform(x)
#
# vis.scatter2d(x,y,'PCA')
