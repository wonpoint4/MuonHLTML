import sys
import numpy as np
import ROOT
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path
import math

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
        # map to TP index
        hltIterL3OIMuonTrack_TPmap = []
        if evt.nhltIterL3OIMuonTrack >0:
            for itrack in range(evt.nhltIterL3OIMuonTrack):
                if evt.hltIterL3OIMuonTrack_matchedTPsize[itrack]!=0:
                    hltIterL3OIMuonTrack_TPmap.append(itrack)
        hltIter0IterL3MuonTrack_TPmap = []
        if evt.nhltIter0IterL3MuonTrack >0:
            for itrack in range(evt.nhltIter0IterL3MuonTrack):
                if evt.hltIter0IterL3MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter0IterL3MuonTrack_TPmap.append(itrack)
        hltIter2IterL3MuonTrack_TPmap = []
        if evt.nhltIter2IterL3MuonTrack >0:
            for itrack in range(evt.nhltIter2IterL3MuonTrack):
                if evt.hltIter2IterL3MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter2IterL3MuonTrack_TPmap.append(itrack)
        hltIter3IterL3MuonTrack_TPmap = []
        if evt.nhltIter3IterL3MuonTrack >0:
            for itrack in range(evt.nhltIter3IterL3MuonTrack):
                if evt.hltIter3IterL3MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter3IterL3MuonTrack_TPmap.append(itrack)
        hltIter0IterL3FromL1MuonTrack_TPmap = []
        if evt.nhltIter0IterL3FromL1MuonTrack >0:
            for itrack in range(evt.nhltIter0IterL3FromL1MuonTrack):
                if evt.hltIter0IterL3FromL1MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter0IterL3FromL1MuonTrack_TPmap.append(itrack)
        hltIter2IterL3FromL1MuonTrack_TPmap = []
        if evt.nhltIter2IterL3FromL1MuonTrack >0:
            for itrack in range(evt.nhltIter2IterL3FromL1MuonTrack):
                if evt.hltIter2IterL3FromL1MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter2IterL3FromL1MuonTrack_TPmap.append(itrack)
        hltIter3IterL3FromL1MuonTrack_TPmap = []
        if evt.nhltIter3IterL3FromL1MuonTrack >0:
            for itrack in range(evt.nhltIter3IterL3FromL1MuonTrack):
                if evt.hltIter3IterL3FromL1MuonTrack_matchedTPsize[itrack]!=0:
                    hltIter3IterL3FromL1MuonTrack_TPmap.append(itrack)

        if evt.nhltIterL3OISeedsFromL2Muons > 0 :
            # add dR, dPhi
            hltIterL3OISeedsFromL2Muons_dR = np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIterL3OISeedsFromL2Muons_dPhi = np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIterL3OISeedsFromL2Muons_SigBkgtag = np.asarray(evt.hltIterL3OISeedsFromL2Muons_tsos_err0,np.float32).copy() #dummy array, value will be replaced  

            for iseed in range(evt.nhltIterL3OISeedsFromL2Muons):

                seed_eta = evt.hltIterL3OISeedsFromL2Muons_tsos_eta[iseed]
                seed_phi = evt.hltIterL3OISeedsFromL2Muons_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIterL3OISeedsFromL2Muons_tsos_glob_x[iseed], evt.hltIterL3OISeedsFromL2Muons_tsos_glob_y[iseed], evt.hltIterL3OISeedsFromL2Muons_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIterL3OISeedsFromL2Muons_dR[iseed] = theDR
                hltIterL3OISeedsFromL2Muons_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIterL3OISeedsFromL2Muons_tmpL3Ref[iseed] == -1:
                    hltIterL3OISeedsFromL2Muons_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIterL3OISeedsFromL2Muons_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIterL3OISeedsFromL2Muons_tmpL3Ref[iseed]
                    if evt.hltIterL3OIMuonTrack_matchedTPsize[trkindex] == 0:
                        hltIterL3OISeedsFromL2Muons_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIterL3OIMuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIterL3OIMuonTrack_bestMatchTP_pdgId[hltIterL3OIMuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIterL3OISeedsFromL2Muons_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIterL3OIMuonTrack_bestMatchTP_pdgId[hltIterL3OIMuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIterL3OISeedsFromL2Muons_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIterL3OISeedsFromL2Muons_dR)
            arr.append(hltIterL3OISeedsFromL2Muons_dPhi)
            arr.append(hltIterL3OISeedsFromL2Muons_SigBkgtag)

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIterL3OISeedsFromL2Muons = np.hstack(arr)
            iterL3OISeedsFromL2Muons.append(hltIterL3OISeedsFromL2Muons)

        if evt.nhltIter0IterL3MuonPixelSeedsFromPixelTracks > 0 :
            # add dR, dPhi
            hltIter0IterL3MuonPixelSeedsFromPixelTracks_dR = np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter0IterL3MuonPixelSeedsFromPixelTracks_dPhi = np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag = np.asarray(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  

            for iseed in range(evt.nhltIter0IterL3MuonPixelSeedsFromPixelTracks):

                seed_eta = evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_eta[iseed]
                seed_phi = evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_glob_x[iseed], evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_glob_y[iseed], evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter0IterL3MuonPixelSeedsFromPixelTracks_dR[iseed] = theDR
                hltIter0IterL3MuonPixelSeedsFromPixelTracks_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed] == -1:
                    hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter0IterL3MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed]
                    if evt.hltIter0IterL3MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter0IterL3MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter0IterL3MuonTrack_bestMatchTP_pdgId[hltIter0IterL3MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter0IterL3MuonTrack_bestMatchTP_pdgId[hltIter0IterL3MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter0IterL3MuonPixelSeedsFromPixelTracks_dR)
            arr.append(hltIter0IterL3MuonPixelSeedsFromPixelTracks_dPhi)
            arr.append(hltIter0IterL3MuonPixelSeedsFromPixelTracks_SigBkgtag)


            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter0IterL3MuonPixelSeedsFromPixelTracks = np.hstack(arr)
            iter0IterL3MuonPixelSeedsFromPixelTracks.append(hltIter0IterL3MuonPixelSeedsFromPixelTracks)

        if evt.nhltIter2IterL3MuonPixelSeeds > 0 :
            # add dR, dPhi
            hltIter2IterL3MuonPixelSeeds_dR = np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter2IterL3MuonPixelSeeds_dPhi = np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter2IterL3MuonPixelSeeds_SigBkgtag = np.asarray(evt.hltIter2IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            for iseed in range(evt.nhltIter2IterL3MuonPixelSeeds):

                seed_eta = evt.hltIter2IterL3MuonPixelSeeds_tsos_eta[iseed]
                seed_phi = evt.hltIter2IterL3MuonPixelSeeds_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter2IterL3MuonPixelSeeds_tsos_glob_x[iseed], evt.hltIter2IterL3MuonPixelSeeds_tsos_glob_y[iseed], evt.hltIter2IterL3MuonPixelSeeds_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter2IterL3MuonPixelSeeds_dR[iseed] = theDR
                hltIter2IterL3MuonPixelSeeds_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter2IterL3MuonPixelSeeds_tmpL3Ref[iseed] == -1:
                    hltIter2IterL3MuonPixelSeeds_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter2IterL3MuonPixelSeeds_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter2IterL3MuonPixelSeeds_tmpL3Ref[iseed]
                    if evt.hltIter2IterL3MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter2IterL3MuonPixelSeeds_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter2IterL3MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter2IterL3MuonTrack_bestMatchTP_pdgId[hltIter2IterL3MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter2IterL3MuonPixelSeeds_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter2IterL3MuonTrack_bestMatchTP_pdgId[hltIter2IterL3MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter2IterL3MuonPixelSeeds_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter2IterL3MuonPixelSeeds_dR)
            arr.append(hltIter2IterL3MuonPixelSeeds_dPhi)   
            arr.append(hltIter2IterL3MuonPixelSeeds_SigBkgtag)

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter2IterL3MuonPixelSeeds = np.hstack(arr)
            iter2IterL3MuonPixelSeeds.append(hltIter2IterL3MuonPixelSeeds)

        if evt.nhltIter3IterL3MuonPixelSeeds > 0 :
            # add dR, dPhi
            hltIter3IterL3MuonPixelSeeds_dR = np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter3IterL3MuonPixelSeeds_dPhi = np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter3IterL3MuonPixelSeeds_SigBkgtag = np.asarray(evt.hltIter3IterL3MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            for iseed in range(evt.nhltIter3IterL3MuonPixelSeeds):

                seed_eta = evt.hltIter3IterL3MuonPixelSeeds_tsos_eta[iseed]
                seed_phi = evt.hltIter3IterL3MuonPixelSeeds_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter3IterL3MuonPixelSeeds_tsos_glob_x[iseed], evt.hltIter3IterL3MuonPixelSeeds_tsos_glob_y[iseed], evt.hltIter3IterL3MuonPixelSeeds_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter3IterL3MuonPixelSeeds_dR[iseed] = theDR
                hltIter3IterL3MuonPixelSeeds_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter3IterL3MuonPixelSeeds_tmpL3Ref[iseed] == -1:
                    hltIter3IterL3MuonPixelSeeds_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter3IterL3MuonPixelSeeds_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter3IterL3MuonPixelSeeds_tmpL3Ref[iseed]
                    if evt.hltIter3IterL3MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter3IterL3MuonPixelSeeds_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter3IterL3MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter3IterL3MuonTrack_bestMatchTP_pdgId[hltIter3IterL3MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter3IterL3MuonPixelSeeds_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter3IterL3MuonTrack_bestMatchTP_pdgId[hltIter3IterL3MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter3IterL3MuonPixelSeeds_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter3IterL3MuonPixelSeeds_dR)
            arr.append(hltIter3IterL3MuonPixelSeeds_dPhi)
            arr.append(hltIter3IterL3MuonPixelSeeds_SigBkgtag)

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter3IterL3MuonPixelSeeds = np.hstack(arr)
            iter3IterL3MuonPixelSeeds.append(hltIter3IterL3MuonPixelSeeds)

        if evt.nhltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks > 0 :
            # add dR, dPhi
            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dR = np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dPhi = np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag = np.asarray(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            for iseed in range(evt.nhltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks):

                seed_eta = evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_eta[iseed]
                seed_phi = evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_glob_x[iseed], evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_glob_y[iseed], evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dR[iseed] = theDR
                hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed] == -1:
                    hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_tmpL3Ref[iseed]
                    if evt.hltIter0IterL3FromL1MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter0IterL3FromL1MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter0IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter0IterL3FromL1MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter0IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter0IterL3FromL1MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dR)
            arr.append(hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_dPhi)
            arr.append(hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks_SigBkgtag)

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks = np.hstack(arr)
            iter0IterL3FromL1MuonPixelSeedsFromPixelTracks.append(hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks)

        if evt.nhltIter2IterL3FromL1MuonPixelSeeds > 0 :
            # add dR, dPhi
            hltIter2IterL3FromL1MuonPixelSeeds_dR = np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter2IterL3FromL1MuonPixelSeeds_dPhi = np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag = np.asarray(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            for iseed in range(evt.nhltIter2IterL3FromL1MuonPixelSeeds):

                seed_eta = evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_eta[iseed]
                seed_phi = evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_glob_x[iseed], evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_glob_y[iseed], evt.hltIter2IterL3FromL1MuonPixelSeeds_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter2IterL3FromL1MuonPixelSeeds_dR[iseed] = theDR
                hltIter2IterL3FromL1MuonPixelSeeds_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter2IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed] == -1:
                    hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter2IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter2IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed]
                    if evt.hltIter2IterL3FromL1MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter2IterL3FromL1MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter2IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter2IterL3FromL1MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter2IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter2IterL3FromL1MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter2IterL3FromL1MuonPixelSeeds_dR)
            arr.append(hltIter2IterL3FromL1MuonPixelSeeds_dPhi)
            arr.append(hltIter2IterL3FromL1MuonPixelSeeds_SigBkgtag)

            for i in [1,3,26,27,28,29]: arr[i].astype(np.float32)
            for i in range(len(arr)): arr[i] = np.reshape(arr[i], (-1,1))

            hltIter2IterL3FromL1MuonPixelSeeds = np.hstack(arr)
            iter2IterL3FromL1MuonPixelSeeds.append(hltIter2IterL3FromL1MuonPixelSeeds)

        if evt.nhltIter3IterL3FromL1MuonPixelSeeds > 0 :
            # add dR, dPhi
            hltIter3IterL3FromL1MuonPixelSeeds_dR = np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter3IterL3FromL1MuonPixelSeeds_dPhi = np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag = np.asarray(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_err0,np.float32).copy() #dummy array, value will be replaced  
            for iseed in range(evt.nhltIter3IterL3FromL1MuonPixelSeeds):

                seed_eta = evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_eta[iseed]
                seed_phi = evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_phi[iseed]
                seed_pos_eta, seed_pos_phi = setEtaPhi(evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_glob_x[iseed], evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_glob_y[iseed], evt.hltIter3IterL3FromL1MuonPixelSeeds_tsos_glob_z[iseed])
                
                theDR = 999.
                theDphi = 999.
                if evt.nL1Muon > 0 :
                    for iL1 in range(evt.nL1Muon):
                        dR_tmp = dR(evt.L1Muon_eta[iL1], evt.L1Muon_phi[iL1], seed_eta ,seed_phi)
                        if(dR_tmp < theDR):
                            theDR = dR_tmp
                            theDphi = dphi(seed_pos_phi, evt.L1Muon_phi[iL1])

                hltIter3IterL3FromL1MuonPixelSeeds_dR[iseed] = theDR
                hltIter3IterL3FromL1MuonPixelSeeds_dPhi[iseed] = theDphi

                # sig, bkg tag
                if evt.hltIter3IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed] == -1:
                    hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 0 # bkg seed that doens't have track
                if evt.hltIter3IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed] != -1:
                    trkindex = evt.hltIter3IterL3FromL1MuonPixelSeeds_tmpL3Ref[iseed]
                    if evt.hltIter3IterL3FromL1MuonTrack_matchedTPsize[trkindex] == 0:
                        hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 1 # bkg seed that are not matched to simtrack
                    if evt.hltIter3IterL3FromL1MuonTrack_matchedTPsize[trkindex] != 0:
                        if abs(evt.hltIter3IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter3IterL3FromL1MuonTrack_TPmap.index(trkindex)]) != 13:
                            hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 2 # bkg seed that are not muon
                        if abs(evt.hltIter3IterL3FromL1MuonTrack_bestMatchTP_pdgId[hltIter3IterL3FromL1MuonTrack_TPmap.index(trkindex)]) == 13:
                            hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag[iseed] = 3 # signal

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
            arr.append(hltIter3IterL3FromL1MuonPixelSeeds_dR)
            arr.append(hltIter3IterL3FromL1MuonPixelSeeds_dPhi)
            arr.append(hltIter3IterL3FromL1MuonPixelSeeds_SigBkgtag)

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
