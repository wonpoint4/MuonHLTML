import sys
import numpy as np
import ROOT
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path

# IO (Require ROOT version > 6.14)

def Read(path,varlist):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("tree")

    mtx = t.AsMatrix(varlist)

    return mtx

def readSeed(path):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("ntupler/ntuple")

    iterL3OISeedsFromL2Muons = []
    iter0IterL3MuonPixelSeedsFromPixelTracks = []
    iter2IterL3MuonPixelSeeds = []
    iter3IterL3MuonPixelSeeds = []
    iter0IterL3FromL1MuonPixelSeedsFromPixelTracks = []
    iter2IterL3FromL1MuonPixelSeeds = []
    iter3IterL3FromL1MuonPixelSeeds = []

    for evt in t:
        if evt.nhltIterL3OISeedsFromL2Muons > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_dir,np.int32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIterL3OISeedsFromL2Muons_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIterL3OISeedsFromL2Muons = np.hstack(arr)
            iterL3OISeedsFromL2Muons.append(hltIterL3OISeedsFromL2Muons)

        if evt.nhltIter0IterL3MuonPixelSeedsFromPixelTracks > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_dir,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter0IterL3MuonPixelSeedsFromPixelTracks = np.hstack(arr)
            iter0IterL3MuonPixelSeedsFromPixelTracks.append(hltIter0IterL3MuonPixelSeedsFromPixelTracks)

        if evt.nhltIter2IterL3MuonPixelSeeds > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_dir,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter2IterL3MuonPixelSeeds = np.hstack(arr)
            iter2IterL3MuonPixelSeeds.append(hltIter2IterL3MuonPixelSeeds)

        if evt.nhltIter3IterL3MuonPixelSeeds > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_dir,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter3IterL3MuonPixelSeeds = np.hstack(arr)
            iter3IterL3MuonPixelSeeds.append(hltIter3IterL3MuonPixelSeeds)

        if evt.nhltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dir,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks = np.hstack(arr)
            iter0IterL3FromL1MuonPixelSeedsFromPixelTracks.append(hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks)

        if evt.nhltIter2IterL3FromL1MuonPixelSeeds > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_dir,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter2IterL3FromL1MuonPixelSeeds = np.hstack(arr)
            iter2IterL3FromL1MuonPixelSeeds.append(hltIter2IterL3FromL1MuonPixelSeeds)

        if evt.nhltIter3IterL3FromL1MuonPixelSeeds > 0 :
            arr = []
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_dir,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_detId,np.uint32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_pt,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_hasErr,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err1,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err2,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err3,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err4,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err5,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err6,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err7,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err8,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err9,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err10,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err11,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err12,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err13,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err14,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_x,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_y,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_dxdz,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_dydz,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_px,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_py,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_qbp,np.float32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_charge,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_iterL3Matched,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_iterL3Ref,np.int32))
            arr.append(np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tmpL3Ref,np.int32))

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter3IterL3FromL1MuonPixelSeeds = np.hstack(arr)
            iter3IterL3FromL1MuonPixelSeeds.append(hltIter3IterL3FromL1MuonPixelSeeds)

    iterL3OISeedsFromL2Muons = np.vstack(iterL3OISeedsFromL2Muons)
    iter0IterL3MuonPixelSeedsFromPixelTracks = np.vstack(iter0IterL3MuonPixelSeedsFromPixelTracks)
    iter2IterL3MuonPixelSeeds = np.vstack(iter2IterL3MuonPixelSeeds)
    iter3IterL3MuonPixelSeeds = np.vstack(iter3IterL3MuonPixelSeeds)
    iter0IterL3FromL1MuonPixelSeedsFromPixelTracks = np.vstack(iter0IterL3FromL1MuonPixelSeedsFromPixelTracks)
    iter2IterL3FromL1MuonPixelSeeds = np.vstack(iter2IterL3FromL1MuonPixelSeeds)
    iter3IterL3FromL1MuonPixelSeeds = np.vstack(iter3IterL3FromL1MuonPixelSeeds)

    print(iterL3OISeedsFromL2Muons.shape)
    print(iter0IterL3MuonPixelSeedsFromPixelTracks.shape)
    print(iter2IterL3MuonPixelSeeds.shape)
    print(iter3IterL3MuonPixelSeeds.shape)
    print(iter0IterL3FromL1MuonPixelSeedsFromPixelTracks.shape)
    print(iter2IterL3FromL1MuonPixelSeeds.shape)
    print(iter3IterL3FromL1MuonPixelSeeds.shape)

    filename = path.replace('.root','')
    np.save(filename+'iterL3OISeedsFromL2Muons',iterL3OISeedsFromL2Muons)
    np.save(filename+'iter0IterL3MuonPixelSeedsFromPixelTracks',iter0IterL3MuonPixelSeedsFromPixelTracks)
    np.save(filename+'iter2IterL3MuonPixelSeeds',iter2IterL3MuonPixelSeeds)
    np.save(filename+'iter3IterL3MuonPixelSeeds',iter3IterL3MuonPixelSeeds)
    np.save(filename+'iter0IterL3FromL1MuonPixelSeedsFromPixelTracks',iter0IterL3FromL1MuonPixelSeedsFromPixelTracks)
    np.save(filename+'iter2IterL3FromL1MuonPixelSeeds',iter2IterL3FromL1MuonPixelSeeds)
    np.save(filename+'iter3IterL3FromL1MuonPixelSeeds',iter3IterL3FromL1MuonPixelSeeds)

    return iterL3OISeedsFromL2Muons, iter0IterL3MuonPixelSeedsFromPixelTracks, iter2IterL3MuonPixelSeeds, iter3IterL3MuonPixelSeeds, \
    iter0IterL3FromL1MuonPixelSeedsFromPixelTracks, iter2IterL3FromL1MuonPixelSeeds, iter3IterL3FromL1MuonPixelSeeds

def readSeedNp(path):
    filename = path.replace('.root','')
    checkfile = Path(filename+'iterL3OISeedsFromL2Muons.npy')
    try:
        checkfile.resolve()
    except FileNotFoundError:
        seeds = readSeed(path)
        
        return seeds
    else:
        iterL3OISeedsFromL2Muons = np.load(filename+'iterL3OISeedsFromL2Muons.npy')
        iter0IterL3MuonPixelSeedsFromPixelTracks = np.load(filename+'iter0IterL3MuonPixelSeedsFromPixelTracks.npy')
        iter2IterL3MuonPixelSeeds = np.load(filename+'iter2IterL3MuonPixelSeeds.npy')
        iter3IterL3MuonPixelSeeds = np.load(filename+'iter3IterL3MuonPixelSeeds.npy')
        iter0IterL3FromL1MuonPixelSeedsFromPixelTracks = np.load(filename+'iter0IterL3FromL1MuonPixelSeedsFromPixelTracks.npy')
        iter2IterL3FromL1MuonPixelSeeds = np.load(filename+'iter2IterL3FromL1MuonPixelSeeds.npy')
        iter3IterL3FromL1MuonPixelSeeds = np.load(filename+'iter3IterL3FromL1MuonPixelSeeds.npy')

        return iterL3OISeedsFromL2Muons, iter0IterL3MuonPixelSeedsFromPixelTracks, iter2IterL3MuonPixelSeeds, iter3IterL3MuonPixelSeeds, \
        iter0IterL3FromL1MuonPixelSeedsFromPixelTracks, iter2IterL3FromL1MuonPixelSeeds, iter3IterL3FromL1MuonPixelSeeds

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
