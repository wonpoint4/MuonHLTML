import sys
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from sklearn.manifold import TSNE

def doTSNE(seed,seedname):
    seed = pd.DataFrame(seed)
    seed, fake = preprocess.dfSigBkg(seed)
    seed.drop(seed.columns[[1,3,26]],axis=1,inplace=True)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
    tsne_result = tsne.fit_transform(seed)
    seed['tsne-x'] = tsne_result[:,0]
    seed['tsne-y'] = tsne_result[:,1]

    vis.scatter2dSB(seed[~fake][['tsne-x','tsne-y']].values, seed[fake][['tsne-x','tsne-y']].values, 't-sne_'+seedname)
    vis.hist2d(seed[~fake][['tsne-x','tsne-y']].values,'t-sneSig_'+seedname)
    vis.hist2d(seed[fake][['tsne-x','tsne-y']].values,'t-sneBkg_'+seedname)

    return

# seeds = IO.readSeed("./data/ntuple_PU50.root")
seeds = IO.readSeed("./data/ntuple_SingleMuon2018C_Run319941_NMu1_Pt27to1000000000_PU40to60_RAWAOD.root")

doTSNE(seeds[0],"iterL3OISeedsFromL2Muons")
doTSNE(seeds[1],"iter0IterL3MuonPixelSeedsFromPixelTracks")
doTSNE(seeds[2],"iter2IterL3MuonPixelSeeds")
doTSNE(seeds[3],"iter3IterL3MuonPixelSeeds")
doTSNE(seeds[4],"iter0IterL3FromL1MuonPixelSeedsFromPixelTracks")
doTSNE(seeds[5],"iter2IterL3FromL1MuonPixelSeeds")
doTSNE(seeds[6],"iter3IterL3FromL1MuonPixelSeeds")




# x, y = IO.loadsvm('./data/testTrain.svm')
#
# pca = PCA(n_components=2)
# pca.fit(x)
# x = pca.transform(x)
#
# vis.scatter2d(x,y,'PCA')
