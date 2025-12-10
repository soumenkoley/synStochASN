#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.special import j0
from scipy.special import jv
from joblib import Parallel, delayed
from math import ceil
import time
import os, sys
from pysurf96 import surf96
from modules import gfLoader
from modules import plotUtils
import matplotlib.pyplot as plt

def main():
    # Define the velocity model in meters and m/s, and kg/m^3
    thickness = np.array([10000.0, 0.0]) # units in meters
    vS = np.array([2000.0, 2500.0]) # units in m/s
    vP = np.array([4000, 5000.0]) # units in m/s
    rho = np.array([2465.0, 2606.0]) # units in kg/m^3
    fMin = 2.0; fMax = 8.0; df = 0.05; # units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMaxGF = 5000.0;# maximum horizontal offset upto which displacements will be used
    zMaxGF = 5000.0; # maximum depth upto which displacements will be used
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    tMax = 40; nSamp = 2048;
    # specify the folder where you want to write all input files, should have rw access
    fInpPath = "/data/gravwav/koley/QseisInpN/"
    #fInpPath = "/data/gravwav/koley/SALVUSOut/"
    outDispPath = "/data/gravwav/koley/OutDisp/"
    outDispRea = "/data/gravwav/koley/OutDispRea/"
    components = ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']
    
    nx, ny = 100, 100
    x = np.linspace(-1000, 1000, nx)
    y = np.linspace(-1000, 1000, ny)
    gridXMat, gridYMat = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)
    # make them flat
    gridX = gridXMat.ravel(); gridY = gridYMat.ravel();
    zTarget = 0;
    nRea = 20; # number of realizations
    #source = (2500, 2500)  # source in center
    
    splitFileName = fInpPath + 'splitAll.mat';
    splitMat = loadmat(splitFileName);
    splitAll = splitMat['splitAll'];
    # file names to be added
    
    # make outDispDir
    outDir = makeDepthDispfolder(outDispPath, zTarget);
    
    R1 = 0; R2 = 2500;
    nSrc = 100;
    srcDistri = "internal"
    
    freqOut, idxFreq, df_native = getFreqGrid(tMax, nSamp, fMin, fMax, df);
    nFreq = len(freqOut)

    zList = [0.0,250.0]
    #pA = np.array((0,0)); pB = np.array((0,200));

    """
    for reaNo in range(0,nRea):
        # get the source distribution
        xSrc, ySrc, azSrc, phiSrc, ampSrc =  genAmbSrc(nSrc , mode = srcDistri, R1 = R1, R2 = R2, xMin=-2000.0, xMax=2000.0, yMin=-2000.0,
                                           yMax=2000.0, randomPhase=True, freqDependent=True, nFreq=nFreq)

        getSurfDeepDispPerRea(zList, xSrc, ySrc, azSrc, phiSrc, ampSrc, xMaxGF, splitAll, fMin, fMax, outDispRea,
                    fInpPath, components, reaNo, idxFreq = idxFreq, freqOut=freqOut)

    # read the saved surface and deep displacements
    dispPointAllRea, attnAllRea, freqDisp = assembleSurfDeepDispAllRea(outDispRea, zList, freqOut, nRea)
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,0], 'Surface', 'b', fig=None, axs=None, 
                                         quantity="ASD")
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,1], 'Deep', 'r', fig=figASD, axs=axASD, 
                                         quantity="ASD")

    figAttn, axAttn = plotUtils.plotPSDDeepSurfMulti(freqOut, attnAllRea, 'Surf-Deep', 'b', fig=None, axs=None, 
                                         quantity="Attn")
    # load the Terziet attenuation model
    terzAttn = loadmat('/data/gravwav/koley/TerzModel/attnModel.mat');
    attnZ = terzAttn['attnZ']
    attnE = terzAttn['attnE']
    attnN = terzAttn['attnN']

    axAttn[0].plot(attnZ[:,0],attnZ[:,2],'r')
    axAttn[1].plot(attnZ[:,0],attnE[:,2],'r')
    axAttn[2].plot(attnZ[:,0],attnN[:,2],'r')
    
    """
    
    # call computeFullDisp
    #dispTotal, outDir, freqOut_used = computeFullDisp(zTarget, gridX, gridY, xSrc, ySrc, azSrc, phiSrc, ampSrc, idxFreq, freqOut, fMin, fMax, outDispPath,
    #                splitAll, xMaxGF, fInpPath, components, nCPU=4, nChunk=20000, minVel=100.0)

    #dispPointFull = getFullDispPerRea(outDir,pA[0],pA[1])
    #figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut_used, dispPointFull, 'Surface', 'b', fig=None, axs=None, 
    #                                     quantity="ASD")
    #getSurfDeepDisp(zList, pA, pB, nSrc, srcDistri, R1, R2, gridX, gridY, xMaxGF, splitAll, fMin, fMax, outDispPath,
    #                fInpPath, components, nRea, idxFreq = idxFreq, freqOut=freqOut)

    
    # get the Green's function from database, no need to reload per realization
    xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zTarget, 0.01, xMaxGF, fInpPath, components, minVel=100.0)
    print(xxW.shape, tVec.shape, distVec.shape)

    # just see where the two particular receiver points are
    # define the two points first, lets say (0,0) and (0,200)
    pA = np.array((0,0)); pB = np.array((0, 200));
    recIdxA, distRecA = find_receiver_index(pA[0], pA[1], gridX, gridY);
    recIdxB, distRecB = find_receiver_index(pB[0], pB[1], gridX, gridY);
    print('rec idx A = ' + str(recIdxA) + ' distA = ' + str(distRecA));
    print('rec idx B = ' + str(recIdxB) + ' distB = ' + str(distRecB));

    dispPointFull = np.zeros((nFreq,3))
    ccRealAllZ = np.zeros((nFreq,))
    ccRealAllX = np.zeros((nFreq,))
    ccRealAllY = np.zeros((nFreq,))
    
    for reaNo in range(0,nRea):
        print("Starting realization = " + str(reaNo));
        
        xSrc, ySrc, azSrc, phiSrc, ampSrc =  genAmbSrc(nSrc,mode="internal", R1 = R1, R2 = R2, xMin=-2000.0, xMax=2000.0, yMin=-2000.0,
                                           yMax=2000.0, randomPhase=True, freqDependent=True, nFreq=nFreq)

        print(np.shape(xSrc));
    
        # plot the src and the receivers
        #plotUtils.plotSrcRec(xSrc, ySrc, gridX, gridY, pA);

        #tested
        #checkSurfCC(xxW, tVec, distVec, pA, pB, xSrc, ySrc, azSrc, phiSrc, ampSrc, 1, thickness, vP, vS, rho, fMin, fMax, idxFreq=idxFreq, freqOut=freqOut)
    
        #plotDispSpect(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc, fMin, fMax);

        # tested
        #dispHForce, dispVForce, freqOut = simFixedDepth_memmap(xxW,tVec,distVec,gridX,gridY,xSrc,ySrc,azSrc,phiSrc,ampSrc,
        #                                                       fMin, fMax,20000,outDir,1,idxFreq=idxFreq,freqOut=freqOut)

        #tested
        dispHForce, dispVForce, freqOut = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc,
                                                phiSrc, ampSrc, fMin, fMax, 20000, outDir, 4, idxFreq = idxFreq, freqOut=freqOut)
        dispPointRea = getDispPerRea(outDir,pA[0],pA[1])
        ccRealZ, ccRealX, ccRealY, recDistAB = getCCPerRea(outDir)

        # sum squared for displacement <x>^2
        dispPointFull = dispPointFull + dispPointRea**2
        ccRealAllZ = ccRealAllZ + ccRealZ
        ccRealAllX = ccRealAllX + ccRealX
        ccRealAllY = ccRealAllY + ccRealY

    dispPointFull = np.sqrt(dispPointFull/nRea)
    ccRealAllZ = ccRealAllZ/nRea
    ccRealAllX = ccRealAllX/nRea
    ccRealAllY = ccRealAllY/nRea
        
    # go for plot
    
    # load the CC
    #data = np.load(os.path.join(outDir,"CCtest_worker2.npz"))
    #freqOut = data["freq"]; recDistAB = data["recDistAB"]; 
    #ccRealZ, ccRealX, ccRealY = data["ccRealZ"], data["ccRealX"], data["ccRealY"]
    # get Bessel
    j0R, j0L, fJ = getBessel(thickness,vP,vS,rho,fMin,fMax,recDistAB);
    figCC, axCC = plotUtils.plotCC(ccRealAllZ, ccRealAllX, ccRealAllY, freqOut, j0R = None, j0L=None, fJ = None)
    # get the second order Bessels
    j2R, j2L, fJ = getBesselN(thickness,vP,vS,rho,fMin,fMax,recDistAB,2)
    # add the bessel diff to subplots
    axCC[0].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(kd$")
    axCC[0].legend(loc="upper right")
    
    axCC[1].plot(fJ,j0R+j2R,'r',label=r"Theoretical $J_0(kd)+J_2(kd)$")
    axCC[1].legend(loc="upper right")
    
    #axCC[1].plot(fJ,j0R-j2R,'k',label=r"Theoretical $J_0(kd)-J_2(kd)$")
    #axCC[1].plot(fJ,j0R+j2R,'r',label=r"Theoretical $J_0(kd)+J_2(kd)$")
    axCC[2].plot(fJ,j0R-j2R,'r',label=r"Theoretical $J_0(kd)-J_2(kd)$")
    axCC[2].legend(loc="upper right")
    
    plotUtils.plotDispSpectAllRea(dispPointFull, freqOut)
    
    plotUtils.plotDispSurf(outDir, fTarget=2.5)
    #plotUtils.plotDispSpect(outDir,pA[0],pA[1])
    #distVal, azVal = computeDistAzi(gridX, gridY, source[0], source[1] );
    #print(np.shape(distVal)); print(np.shape(azVal));
    

def computeFullDisp(zTar, gridX, gridY, xSrc, ySrc, azSrc, phiSrc, ampSrc, idxFreq, freqOut, fMin, fMax, outDispPath,
                    splitAll, xMaxGF, fInpPath, components, nCPU=4, nChunk=20000, minVel=100.0):
    """
    Compute the full displacement field (H-force + V-force) at one depth slice zTar
    on the receiver grid defined by (gridX, gridY).

    This function:
      1. Loads interpolated Green's functions for depth zTar.
      2. Runs simFixedDepth_partition_receivers to compute dispHForce, dispVForce
         as memmapped arrays on disk.
      3. Creates a new memmap dispTotal.dat = dispHForce + dispVForce (component-wise).
      4. Returns (dispTotal_memmap, outDir, freqOut).

    Parameters
    ----------
    zTar : float
        Target depth [meters] (positive down).
    gridX, gridY : 1D arrays
        Receiver coordinates (flattened, same length nRec).
    xSrc, ySrc, azSrc, phiSrc, ampSrc :
        Source parameters from genAmbSrc (or similar).
    idxFreq, freqOut :
        Frequency indices and values to use (from getFreqGrid).
    fMin, fMax : float
        Frequency band [Hz].
    outDispPath : str
        Base output directory for displacement files.
    splitAll, xMaxGF, fInpPath, components :
        Green's function / database config.
    nCPU : int
        Number of workers for simFixedDepth_partition_receivers.
    nChunk : int
        Receiver chunk size per worker.
    minVel : float
        Minimum velocity for GF interpolation (passed to gfLoader).

    Returns
    -------
    dispTotal : np.memmap
        Memmapped full displacement array with shape (nFreq, nRec, 3).
    outDir : str
        Directory where files are stored.
    freqOut : ndarray
        Frequencies used.
    """

    # 1) Make output folder for this depth
    outDir = makeDepthDispfolder(outDispPath, zTar)

    # 2) Load Green's functions for this depth
    xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zTar, 0.01, xMaxGF, fInpPath, components, minVel=minVel)

    # 3) Run parallel simulation at this depth
    dispHForce, dispVForce, freqOut_used = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, 
                                                                             xSrc, ySrc, azSrc, phiSrc, ampSrc, 
                                                                             fMin, fMax, nChunk, outDir, n_workers=nCPU,
                                                                             idxFreq=idxFreq, freqOut=freqOut)

    # dispHForce, dispVForce are np.memmap with shape (nFreq, nRec, 3)
    nFreq, nRec, _ = dispHForce.shape

    # 4) Create total displacement memmap and sum in chunks
    totalFile = os.path.join(outDir, "dispTotal.dat")
    dispTotal = np.memmap(totalFile, dtype=np.complex128, mode="w+", shape=(nFreq, nRec, 3))

    # Chunked sum to avoid huge temporary arrays
    chunk_size = nChunk  # adjust if you like
    for start in range(0, nRec, chunk_size):
        end = min(start + chunk_size, nRec)
        # this reads only the slice from each memmap
        dispTotal[:, start:end, :] = (dispHForce[:, start:end, :] + dispVForce[:, start:end, :])

    dispTotal.flush()

    # (Optional) keep receiverGrid.npz consistent — simFixedDepth_partition_receivers
    # already saved xGrid, yGrid, freqOut there.

    return dispTotal, outDir, freqOut_used
    
def assembleSurfDeepDispAllRea(outReaPath,zList, freqOut, nRea):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointAllRea = np.zeros((nFreq,3,zLen))
    attnAllRea = np.zeros((nFreq,3))
    
    for reaNo in range(0,nRea):
        sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        dispPointFull = data["dispPointFull"]
        dispPointAllRea = dispPointAllRea + dispPointFull**2

    dispPointAllRea = np.sqrt(dispPointAllRea/nRea)
    attnAllRea = dispPointAllRea[:,:,0]/dispPointAllRea[:,:,1]
    freqOut = data["freqOut"];
    sName = 'fullDisp.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointAllRea=dispPointAllRea, freqOut=freqOut, attnAllRea=attnAllRea)
    return dispPointAllRea, attnAllRea, freqOut

    
def getSurfDeepDispPerRea(zList, xSrc, ySrc, azSrc, phiSrc, ampSrc, xMaxGF, splitAll, fMin, fMax, outReaPath,
                    fInpPath, components, reaNo, idxFreq = None, freqOut=None, nCPU = 4):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the displacement at point (0,0) on the surface and at depth
    specified by zList
    the grid will be generated within and will be a small one because we are only interested
    at getting the displacement at one point (0,0)
    
    """
    # generate the small grid, private variables not to be changed
    nx, ny = 10, 10
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-50, 50, ny)
    
    gridXMat, gridYMat = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)
    # make them flat
    gridX = gridXMat.ravel(); gridY = gridYMat.ravel();

    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # make outDispDir
        outDir = makeDepthDispfolder(outReaPath, zVal)
        # load the green's funtion for the depth
        xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zVal, 0.01, xMaxGF, fInpPath,
                                                            components, minVel=100.0)
        dispHForce, dispVForce, freqOut = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc,
                                                phiSrc, ampSrc, fMin, fMax, 20000, outDir, n_workers=nCPU, idxFreq = idxFreq, freqOut=freqOut)
        dispPointFull[:,:,zNo] = getDispPerRea(outDir,0.0,0.0)

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut
    
def getSurfDeepDispAllRea(zList, pA, pB, nSrc, srcDistri, R1, R2, gridX, gridY, xMaxGF, splitAll, fMin, fMax, outDispPath,
                    fInpPath, components, nRea, idxFreq = None, freqOut=None, nCPU=4):
    """
    function returns displacement on surface and at a desired depth
    always run on a smmaler grid for quick results, to check how accurate they are to start with
    """
    nFreq = len(freqOut)
    zLen = len(zList)
    dispPointFull = np.zeros((nFreq,3,zLen))
    ccRealAllZ = np.zeros((nFreq,zLen))
    ccRealAllX = np.zeros((nFreq,zLen))
    ccRealAllY = np.zeros((nFreq,zLen))

    xMin = min(gridX); xMax = max(gridX)
    yMin = min(gridY); yMax = max(gridY)
    
    for reaNo in range(0,nRea):
        xSrc, ySrc, azSrc, phiSrc, ampSrc =  genAmbSrc(nSrc , mode = srcDistri, R1 = R1, R2 = R2, xMin=-1000.0, xMax=1000.0, yMin=-2000.0,
                                           yMax=2000.0, randomPhase=True, freqDependent=True, nFreq=nFreq)
        for zNo, zVal in enumerate(zList):
            # make outDispDir
            outDir = makeDepthDispfolder(outDispPath, zVal)
            # load the green's funtion for the depth
            xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zVal, 0.01, xMaxGF, fInpPath,
                                                            components, minVel=100.0)
            dispHForce, dispVForce, freqOut = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc,
                                                phiSrc, ampSrc, fMin, fMax, 20000, outDir, n_workers=nCPU, idxFreq = idxFreq, freqOut=freqOut)
            dispPointRea = getDispPerRea(outDir,pA[0],pA[1])
            ccRealZ, ccRealX, ccRealY, recDistAB = getCCPerRea(outDir)

            # sum squared for displacement <x>^2
            dispPointFull[:,:,zNo] = dispPointFull[:,:,zNo] + dispPointRea**2
            ccRealAllZ[:,zNo] = ccRealAllZ[:,zNo] + ccRealZ
            ccRealAllX[:,zNo] = ccRealAllX[:,zNo] + ccRealX
            ccRealAllY[:,zNo] = ccRealAllY[:,zNo] + ccRealY
            
    dispPointFull = np.sqrt(dispPointFull/nRea)
    ccRealAllZ = ccRealAllZ/nRea
    ccRealAllX = ccRealAllX/nRea
    ccRealAllY = ccRealAllY/nRea

    # plot the correlations
    figCC, axCC = plotUtils.plotCCDeepSurfMulti(freqOut, ccRealAllZ[:,0], ccRealAllX[:,0], ccRealAllY[:,0], 'Surface',
                                      'b', fig=None, axs=None, quantity="real(CC)")
    figCC, axCC = plotUtils.plotCCDeepSurfMulti(freqOut, ccRealAllZ[:,1], ccRealAllX[:,1], ccRealAllY[:,1], 'Deep',
                                      'r', fig=figCC, axs=axCC, quantity="real(CC)")

    # plot the displacements
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointFull[:,:,0], 'Surface', 'b', fig=None, axs=None, 
                                         quantity="ASD")
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointFull[:,:,1], 'Deep', 'r', fig=figASD, axs=axASD, 
                                         quantity="ASD")
    
def getCCPerRea(outDir):
    data = np.load(os.path.join(outDir,"CCtest_worker2.npz"))
    
    freqOut = data["freq"]; recDistAB = data["recDistAB"]; 
    
    ccRealZ, ccRealX, ccRealY = data["ccRealZ"], data["ccRealX"], data["ccRealY"]

    return ccRealZ, ccRealX, ccRealY, recDistAB

def getFullDispPerRea(outDir,xP,yP):
    # first load the all the receievr grid
    data = np.load(os.path.join(outDir, "receiverGrid.npz"));
    
    xGrid = data["xGrid"];
    yGrid = data["yGrid"];
    freqOut = data["freqOut"];
    nFreq = data["nFreq"];
    nRec = len(xGrid);
    
    # load the displacement file
    dispForce = np.memmap(
        os.path.join(outDir,"dispTotal.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    
    rec_index, distRec = find_receiver_index(xP, yP, xGrid, yGrid)  # example receiver index
    dispPointFull = np.abs(dispForce[:, rec_index, :]) # shape (nFreq, 3)
    #dispPointFull = np.abs(dispVForce[:, rec_index, :]) # shape (nFreq, 3)
    return dispPointFull
    
def getDispPerRea(outDir,xP,yP):
    # first load the all the receievr grid
    data = np.load(os.path.join(outDir, "receiverGrid.npz"));
    
    xGrid = data["xGrid"];
    yGrid = data["yGrid"];
    freqOut = data["freqOut"];
    nFreq = data["nFreq"];
    nRec = len(xGrid);
    
    # load the displacement file
    dispHForce = np.memmap(
        os.path.join(outDir,"dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir,"dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    
    rec_index, distRec = find_receiver_index(xP, yP, xGrid, yGrid)  # example receiver index
    dispPointFull = np.abs(dispHForce[:, rec_index, :] + dispVForce[:, rec_index, :]) # shape (nFreq, 3)
    #dispPointFull = np.abs(dispVForce[:, rec_index, :]) # shape (nFreq, 3)
    return dispPointFull

    
def makeDepthDispfolder(outDispPath, zTarget):
    """
    Create (if needed) and return the output folder for a given receiver depth.

    Example:
        base_dir = '/data/.../OutDisp/'
        zTarget = 6.25
        -> returns '/data/.../OutDisp/Depth6p25/'
    """
    # Format depth safely for filenames
    depth_folder = f"Depth{zTarget:.2f}".replace(".", "p")

    # Combine into full path
    outDir = os.path.join(outDispPath, depth_folder)

    # Ensure directory exists
    os.makedirs(outDir, exist_ok=True)

    #print(f"[simDisp] Output directory for z={zTarget:.2f} m - {outDir}")
    return outDir
    
def worker_proc(worker_id, n_workers, xxWFFT, distVec, idx, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc,
                nFreq, chunk_size, outDir, dispHFile, dispVFile, freqOut):
    """
    Worker that computes contributions for a disjoint receiver slice.
    Each worker loops over all sources, but only writes to its assigned receiver indices.
    """

    start_time = time.time()

    nRec = len(xGrid)
    if nRec == 0:
        raise ValueError("simFixedDepth_partition_receivers: nRec == 0 (empty receiver grid)")
    
    # compute receiver slice for this worker (balanced, contiguous)
    base = np.array_split(np.arange(nRec), n_workers)
    rec_inds = base[worker_id]            # this is an array of receiver indices for this worker

    #print(f"[Worker {worker_id}] Assigned receiver indices: {rec_inds[0]}–{rec_inds[-1]} "
    #      f"(total {len(rec_inds)})")

    # Create memmap views for the whole file (each worker only writes its indices)
    dispH = np.memmap(dispHFile, dtype=np.complex128, mode='r+', shape=(nFreq, nRec, 3))
    dispV = np.memmap(dispVFile, dtype=np.complex128, mode='r+', shape=(nFreq, nRec, 3))

    # Helper to group consecutive indices into contiguous ranges
    def index_ranges(idxs):
        if len(idxs) == 0:
            return []
        ranges = []
        start = idxs[0]
        prev = start
        for i in idxs[1:]:
            if i == prev + 1:
                prev = i
                continue
            ranges.append((start, prev + 1))
            start = i
            prev = i
        ranges.append((start, prev + 1))
        return ranges

    rec_ranges = index_ranges(rec_inds.tolist())

    nSrc = len(xSrc)
    xxWFFT_sel = xxWFFT[idx, :, :]  # shape (nFreq, nx, ncomp)

    doCCtest = (worker_id == 2)
    doCCtest = 0;
    if doCCtest:
        # receiver indices for your test points (you can pass them in later as arguments too)
        recIdxA, distRecA = find_receiver_index(0, 0, xGrid, yGrid)
        recIdxB, distRecB = find_receiver_index(0, 200, xGrid, yGrid)
        recDistAB = np.sqrt((xGrid[recIdxA]-xGrid[recIdxB])**2 + (yGrid[recIdxA]-yGrid[recIdxB])**2)
        
        # predefine accumulators
        ccAAZ = np.zeros((nFreq,), dtype=np.complex128)
        ccBBZ = np.zeros((nFreq,), dtype=np.complex128)
        ccABZ = np.zeros((nFreq,), dtype=np.complex128)
        ccAAX = np.zeros((nFreq,), dtype=np.complex128)
        ccBBX = np.zeros((nFreq,), dtype=np.complex128)
        ccABX = np.zeros((nFreq,), dtype=np.complex128)
        ccAAY = np.zeros((nFreq,), dtype=np.complex128)
        ccBBY = np.zeros((nFreq,), dtype=np.complex128)
        ccABY = np.zeros((nFreq,), dtype=np.complex128)

    last_print_time = time.time()

    for srcNo in range(nSrc):
        xS = xSrc[srcNo]; yS = ySrc[srcNo]; azS = azSrc[srcNo]
        #phaseFactor = ampSrc[srcNo]*np.exp(-1j * phiSrc[srcNo]);
        phaseFactor = ampSrc[srcNo, :][:, np.newaxis, np.newaxis] * \
                  np.exp(-1j * phiSrc[srcNo, :][:, np.newaxis, np.newaxis])
        
        for (i0, i1) in rec_ranges:
            for j0 in range(i0, i1, chunk_size):
                j1 = min(j0 + chunk_size, i1)
                dx = xGrid[j0:j1] - xS
                dy = yGrid[j0:j1] - yS
                distGrid = np.sqrt(dx * dx + dy * dy)
                azGridV = np.arctan2(dx, dy)         # for vertical source (no rotation)
                azGridH = azGridV - azS   # for horizontal source rotation
                
                # interpolation indices
                distGrid = np.clip(distGrid, distVec[0], distVec[-1]);
                idx_hi = np.searchsorted(distVec, distGrid, side='right')
                idx_hi = np.clip(idx_hi, 1, len(distVec) - 1)
                idx_lo = idx_hi - 1

                w = (distGrid - distVec[idx_lo]) / (distVec[idx_hi] - distVec[idx_lo])
                w = w[np.newaxis, :, np.newaxis]

                G_lo = xxWFFT_sel[:, idx_lo, :]
                G_hi = xxWFFT_sel[:, idx_hi, :]

                xxInterp = ((1 - w) * G_lo + w * G_hi)*phaseFactor

                sinH = np.sin(azGridH)[np.newaxis, :]
                cosH = np.cos(azGridH)[np.newaxis, :]
                sinV = np.sin(azGridV)[np.newaxis, :]
                cosV = np.cos(azGridV)[np.newaxis, :]
                
                if doCCtest and (recIdxA >= j0 and recIdxA < j1 and recIdxB >= j0 and recIdxB < j1):
                    # local indices inside this chunk
                    iA = recIdxA - j0
                    iB = recIdxB - j0

                    #uHZA = xxInterp[:, iA, 0] + xxInterp[:, iA, 3]
                    #uHXA = xxInterp[:, iA, 1] * sinH[:, iA] + xxInterp[:, iA, 2] * cosH[:, iA] + xxInterp[:, iA, 4] * sinV[:, iA]
                    #uHYA = xxInterp[:, iA, 1] * cosH[:, iA] - xxInterp[:, iA, 2] * sinH[:, iA] + xxInterp[:, iA, 4] * cosV[:, iA]
                    
                    uHZA = xxInterp[:, iA, 3]
                    uHXA = xxInterp[:, iA, 4] * sinV[:, iA]
                    uHYA = xxInterp[:, iA, 4] * cosV[:, iA]
                    
                    #uHZB = xxInterp[:, iB, 0] + xxInterp[:, iB, 3]
                    #uHXB = xxInterp[:, iB, 1] * sinH[:, iB] + xxInterp[:, iB, 2] * cosH[:, iB] + xxInterp[:, iB, 4] * sinV[:, iB]
                    #uHYB = xxInterp[:, iB, 1] * cosH[:, iB] - xxInterp[:, iB, 2] * sinH[:, iB] + xxInterp[:, iB, 4] * cosV[:, iB]

                    uHZB = xxInterp[:, iB, 3]
                    uHXB = xxInterp[:, iB, 4] * sinV[:, iB]
                    uHYB = xxInterp[:, iB, 4] * cosV[:, iB]
                    
                    # accumulate
                    ccAAZ += uHZA * np.conjugate(uHZA)
                    ccBBZ += uHZB * np.conjugate(uHZB)
                    ccABZ += uHZA * np.conjugate(uHZB)

                    ccAAX += uHXA * np.conjugate(uHXA)
                    ccBBX += uHXB * np.conjugate(uHXB)
                    ccABX += uHXA * np.conjugate(uHXB)

                    ccAAY += uHYA * np.conjugate(uHYA)
                    ccBBY += uHYB * np.conjugate(uHYB)
                    ccABY += uHYA * np.conjugate(uHYB)
                    
                dispH[:, j0:j1, 0] += -xxInterp[:, :, 0]
                dispH[:, j0:j1, 1] += xxInterp[:, :, 1] * sinH + xxInterp[:, :, 2] * cosH
                dispH[:, j0:j1, 2] += xxInterp[:, :, 1] * cosH - xxInterp[:, :, 2] * sinH

                dispV[:, j0:j1, 0] += -xxInterp[:, :, 3]
                dispV[:, j0:j1, 1] += xxInterp[:, :, 4] * sinV
                dispV[:, j0:j1, 2] += xxInterp[:, :, 4] * cosV

                dispH.flush()
                dispV.flush()

        # --- progress print every 100 sources ---
        if (srcNo + 1) % 100 == 0 or (srcNo + 1) == nSrc:
            elapsed = time.time() - last_print_time
            total_elapsed = time.time() - start_time
            #print(f"[Worker {worker_id}] Processed {srcNo + 1}/{nSrc} sources "
            #      f"({100*(srcNo+1)/nSrc:.1f}%) | "
            #      f"Elapsed since last: {elapsed:.2f}s | Total: {total_elapsed:.1f}s")
            last_print_time = time.time()

    total_time = time.time() - start_time
    if doCCtest:
        ccRealZ = np.real(ccABZ / np.sqrt(ccAAZ * ccBBZ))
        ccRealX = np.real(ccABX / np.sqrt(ccAAX * ccBBX))
        ccRealY = np.real(ccABY / np.sqrt(ccAAY * ccBBY))
        #freqOut = np.linspace(fMin,fMax,nFreq)
        #outout = "/data/gravwav/koley/QseisInp/OutDisp/";
        np.savez(
            os.path.join(outDir, f"CCtest_worker{worker_id}.npz"),
            freq=freqOut,
            ccRealZ=ccRealZ,
            ccRealX=ccRealX,
            ccRealY=ccRealY,
            recDistAB=recDistAB
        )
    #print(f"[Worker {worker_id}] Finished in {total_time:.2f} s")

    return True


def simFixedDepth_partition_receivers(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc,
                                      fMin, fMax, chunk_size, outDir, n_workers=None, idxFreq=None, freqOut=None):
    """
    Partition receivers across workers (no per-worker big files).
    Each worker loops over all sources but writes only to its receiver slice.
    """

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)

    if nRec == 0:
        print("[simDisp] No receivers - returning empty displacement.")
        # Return empty memmaps or None; whichever your caller expects
        return None, None, freqOut

    # cap workers globally
    n_workers = min(n_workers, nRec)
    if n_workers < 1:
        n_workers = 1
    
    #dt = tVec[1] - tVec[0]
    #freqUse = np.fft.rfftfreq(nt, dt)
    xxWFFT = np.fft.rfft(xxW, axis=0)
    
    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
    #print(freqUse[0:5])
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
    #freqOut = freqUse[idx]
    #nFreq = len(idx)

    #df_native = freqUse[1] - freqUse[0]
    #df_target = df  # or user input
    #step = max(1, int(round(df_target / df_native)))
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    #freqOut = freqUse[idx]
    nFreq = len(idxFreq);

    os.makedirs(outDir, exist_ok=True)
    dispHFile = os.path.join(outDir, "dispHForce.dat")
    dispVFile = os.path.join(outDir, "dispVForce.dat")

    dispH = np.memmap(dispHFile, dtype=np.complex128, mode='w+', shape=(nFreq, nRec, 3))
    dispV = np.memmap(dispVFile, dtype=np.complex128, mode='w+', shape=(nFreq, nRec, 3))
    dispH[:] = 0
    dispV[:] = 0
    dispH.flush(); dispV.flush()

    #print(f"Starting parallel processing with {n_workers} workers...")
    start = time.time()

    Parallel(n_jobs=n_workers, backend='multiprocessing')(
        delayed(worker_proc)(worker_id, n_workers, xxWFFT, distVec, idxFreq, xGrid, yGrid,
                             xSrc, ySrc, azSrc, phiSrc, ampSrc, nFreq, chunk_size, outDir, dispHFile, dispVFile, freqOut)
        for worker_id in range(n_workers)
    )

    #print(f"All workers finished in {time.time() - start:.2f} s")

    dispH_final = np.memmap(dispHFile, dtype=np.complex128, mode='r', shape=(nFreq, nRec, 3))
    dispV_final = np.memmap(dispVFile, dtype=np.complex128, mode='r', shape=(nFreq, nRec, 3))

    np.savez(os.path.join(outDir, "receiverGrid.npz"), xGrid=xGrid, yGrid=yGrid, nFreq=nFreq, freqOut=freqOut)

    return dispH_final, dispV_final, freqOut
    
    
def simFixedDepth_memmap(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc, fMin, fMax,
                         chunk_size, outDir, ifCC, idxFreq=None, freqOut=None):
    """
    Same as previous simFixedDepth but uses np.memmap for outputs.
    """

    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)

    #dt = tVec[1] - tVec[0]
    #freqUse = np.fft.rfftfreq(nt, dt)
    xxWFFT = np.fft.rfft(xxW, axis=0)

    # Keep only the frequency range of interest
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
    #freqOut = freqUse[idx]
    #nFreq = len(idx)
    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
    
    #df_native = freqUse[1] - freqUse[0]
    #df_target = df  # or user input
    #step = max(1, int(round(df_target / df_native)))
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    #freqOut = freqUse[idx]
    nFreq = len(idxFreq);
    
    # Prepare memmap files
    dispHForceFile = os.path.join(outDir, "dispHForce.dat")
    dispVForceFile = os.path.join(outDir, "dispVForce.dat")

    dispHForce = np.memmap(dispHForceFile, dtype=np.complex128,
                           mode="w+", shape=(nFreq, nRec, 3))
    dispVForce = np.memmap(dispVForceFile, dtype=np.complex128,
                           mode="w+", shape=(nFreq, nRec, 3))

    # Precompute distance grid step for interpolation
    distStep = np.diff(distVec)
    distStep = np.concatenate([distStep, distStep[-1:]])

    if(ifCC):
        # including this part to check everything is fine by using cross-correlations
        recIdxA, distRecA = find_receiver_index(0, 0, xGrid, yGrid);
        recIdxB, distRecB = find_receiver_index(0, 600, xGrid, yGrid);
        recDistAB = np.sqrt((xGrid[recIdxA]-xGrid[recIdxB])**2 + (yGrid[recIdxA]-yGrid[recIdxB])**2)
        thickness = np.array([20.0, 25.0, 75.0, 180.0, 700.0, 4000.0, 0.0]) # units in meters
        vS = np.array([300.0, 500.0, 1500.0, 2500.0, 3000.0, 3360.0, 3750.0]) # units in m/s
        vP = np.array([1000, 2000.0, 3000.0, 4500.0, 5000.0, 5800.0, 6000.0]) # units in m/s
        rho = np.array([1743.0, 2073.1, 2294.3, 2539.0, 2606.8, 2720.0, 2800.0]) # units in kg/m^3
        j0R, j0L, fJ = getBessel(thickness,vP,vS,rho,fMin,fMax,recDistAB);
        ccAAZ = np.zeros((nFreq,))
        ccBBZ = np.zeros((nFreq,))
        ccABZ = np.zeros((nFreq,))

        ccAAX = np.zeros((nFreq,))
        ccBBX = np.zeros((nFreq,))
        ccABX = np.zeros((nFreq,))

        ccAAY = np.zeros((nFreq,))
        ccBBY = np.zeros((nFreq,))
        ccABY = np.zeros((nFreq,))
    
    # Loop over sources
    stTime = time.time();
    for srcNo in range(nSrc):
        # display tie taken every 50 sources
        phaseFactor = np.exp(-1j * phiSrc[srcNo]);
        if (srcNo+1) % 50 == 0:
            endTime = time.time();
            #print(f"Source {srcNo+1} took {endTime - stTime:.3f} seconds");
            stTime = time.time();
        
        xS, yS, azS = xSrc[srcNo], ySrc[srcNo], azSrc[srcNo]
        phaseFactor = ampSrc[srcNo, :][:, np.newaxis, np.newaxis] * \
                  np.exp(-1j * phiSrc[srcNo, :][:, np.newaxis, np.newaxis])
        # Process receivers in chunks
        for i0 in range(0, nRec, chunk_size):
            i1 = min(i0 + chunk_size, nRec)

            dx = xGrid[i0:i1] - xS
            dy = yGrid[i0:i1] - yS

            distGrid = np.sqrt(dx**2 + dy**2)
            azGridH = np.arctan2(dx, dy) - azS
            azGridV = np.arctan2(dx, dy)
            
            # Vectorized interpolation
            idx_hi = np.searchsorted(distVec, distGrid, side='right')
            idx_hi = np.clip(idx_hi, 1, len(distVec)-1)
            idx_lo = idx_hi - 1
            w = (distGrid - distVec[idx_lo]) / (distVec[idx_hi] - distVec[idx_lo])
            w = w[np.newaxis, :, np.newaxis]

            G_lo = xxWFFT[idxFreq[:, np.newaxis], idx_lo, :]
            G_hi = xxWFFT[idxFreq[:, np.newaxis], idx_hi, :]
            xxInterp = ((1 - w) * G_lo + w * G_hi)*phaseFactor

            if(ifCC):
                # adding this part to check cross-correlations
                uHZA = xxInterp[:,recIdxA,0];
                uHXA = xxInterp[:, recIdxA, 1] * np.sin(azGridH[recIdxA]) + xxInterp[:, recIdxA, 2] * np.cos(azGridH[recIdxA])  # X
                uHYA = xxInterp[:, recIdxA, 1] * np.cos(azGridH[recIdxA]) - xxInterp[:, recIdxA, 2] * np.sin(azGridH[recIdxA])  # Y
            
                uHZB = xxInterp[:,recIdxB,0];
                uHXB = xxInterp[:, recIdxB, 1] * np.sin(azGridH[recIdxB]) + xxInterp[:, recIdxB, 2] * np.cos(azGridH[recIdxB])  # X
                uHYB = xxInterp[:, recIdxB, 1] * np.cos(azGridH[recIdxB]) - xxInterp[:, recIdxB, 2] * np.sin(azGridH[recIdxB])  # Y

                ccAAZ = ccAAZ + uHZA*np.conjugate(uHZA);
                ccBBZ = ccBBZ + uHZB*np.conjugate(uHZB);
                ccABZ = ccABZ + uHZA*np.conjugate(uHZB);

                ccAAX = ccAAX + uHXA*np.conjugate(uHXA);
                ccBBX = ccBBX + uHXB*np.conjugate(uHXB);
                ccABX = ccABX + uHXA*np.conjugate(uHXB);
        
                ccAAY = ccAAY + uHYA*np.conjugate(uHYA);
                ccBBY = ccBBY + uHYB*np.conjugate(uHYB);
                ccABY = ccABY + uHYA*np.conjugate(uHYB);
            
            sinH = np.sin(azGridH)[np.newaxis, :]
            cosH = np.cos(azGridH)[np.newaxis, :]
            sinV = np.sin(azGridV)[np.newaxis, :]
            cosV = np.cos(azGridV)[np.newaxis, :]

            # Horizontal force
            dispHForce[:, i0:i1, 0] += xxInterp[:, :, 0]
            dispHForce[:, i0:i1, 1] += xxInterp[:, :, 1]*sinH + xxInterp[:, :, 2]*cosH
            dispHForce[:, i0:i1, 2] += xxInterp[:, :, 1]*cosH - xxInterp[:, :, 2]*sinH

            # Vertical force
            dispVForce[:, i0:i1, 0] += xxInterp[:, :, 3]
            dispVForce[:, i0:i1, 1] += xxInterp[:, :, 4]*sinV
            dispVForce[:, i0:i1, 2] += xxInterp[:, :, 4]*cosV

            # Flush after each chunk to ensure data is written
            dispHForce.flush()
            dispVForce.flush()
        
        #endTime = time.time()  # <-- end timing

    if(ifCC):
        # compute the normalized cross-correlation
        ccRealZ = np.real(ccABZ/np.sqrt(ccAAZ*ccBBZ));
        ccRealX = np.real(ccABX/np.sqrt(ccAAX*ccBBX));
        ccRealY = np.real(ccABY/np.sqrt(ccAAY*ccBBY));
        plotUtils.plotCC(ccRealZ, ccRealX, ccRealY, freqOut, j0R, j0L, fJ);
    
    np.savez(os.path.join(outDir, "receiverGrid.npz"), xGrid=xGrid, yGrid=yGrid, nFreq=nFreq,freqOut=freqOut);
    return dispHForce, dispVForce, freqOut

def find_receiver_index(x1, y1, xGrid, yGrid, xVec=None, yVec=None):
    """
    Find the receiver index closest to a given (x1, y1) location.

    Parameters
    ----------
    x1, y1 : float
        Target coordinates of interest.
    xGrid, yGrid : np.ndarray
        Flattened receiver coordinate arrays (1D, same length).
    xVec, yVec : np.ndarray, optional
        1D coordinate vectors used to generate the grid (optional).
        If provided, index computation is analytic and faster.

    Returns
    -------
    rec_index : int
        Index of the nearest receiver in flattened arrays.
    distance : float
        Euclidean distance between target and receiver location.
    """

    # --- Case 1: structured grid provided ---
    if xVec is not None and yVec is not None:
        nx = len(xVec)
        ix = np.argmin(np.abs(xVec - x1))
        iy = np.argmin(np.abs(yVec - y1))
        rec_index = iy * nx + ix
        distance = np.sqrt((xVec[ix] - x1)**2 + (yVec[iy] - y1)**2)
        return rec_index, distance

    # --- Case 2: general case, arbitrary receiver layout ---
    dist = np.sqrt((xGrid - x1)**2 + (yGrid - y1)**2)
    rec_index = np.argmin(dist)
    #print('RecInd = ' + str(rec_index) + ' Dist = ' + str(dist[rec_index]));
    #print('Recpoint x = ' + str(xGrid[rec_index]) + ' y = ' + str(yGrid[rec_index]));
    
    return rec_index, dist[rec_index]

    
def simFixedDepth(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc, fMin, fMax, idxFreq=None, freqOut = None):
    """
    Simulate displacement for many sources (single-depth case) in a fully vectorized way.
    
    Parameters
    ----------
    xxW : np.ndarray
        Green's function matrix of shape (nt, nx, ncomp)
    tVec : np.ndarray
        Time vector
    distVec : np.ndarray
        Distance samples for xxW
    xGrid, yGrid : np.ndarray
        Receiver coordinates (1D arrays of length nRec)
    xSrc, ySrc : np.ndarray
        Source x, y positions (1D arrays of length nSrc)
    
    Returns
    -------
    dispHForce : np.ndarray
        Simulated summed displacement at all receivers for horizontal force
        Shape: (nt, nRec, 3)  [Z, X, Y]
    dispVForce : np.ndarray
        Simulated summed displacement at all receivers for vertical force
        Shape: (nt, nRec, 3)  [Z, X, Y]
    """
    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)
    
    # Flatten receiver grids if they are 2D
    xGridFlat = np.ravel(xGrid)
    yGridFlat = np.ravel(yGrid)

    dt = tVec[1]-tVec[0];
    freqUse = np.fft.rfftfreq(nt,dt);
    # perform FFT of Green's function
    xxWFFT = np.fft.rfft(xxW,axis=0);
    
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0];
    #nFreq = len(idx);
    #freqOut = freqUse[idx];

    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
        
    #df_native = freqUse[1] - freqUse[0]
    #df_target = df  # or user input
    #step = max(1, int(round(df_target / df_native)))
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    #freqOut = freqUse[idx]
    nFreq = len(idxFreq);
    
    dispHForce = np.zeros((nFreq, nRec, 3), dtype = 'complex')
    dispVForce = np.zeros((nFreq, nRec, 3), dtype = 'complex')
    
    for srcNo in range(nSrc):
        # 1) Compute distance and azimuth from current source to all receivers
        phaseFactor = ampSrc[srcNo, :][:, np.newaxis] * \
                  np.exp(-1j * phiSrc[srcNo, :][:, np.newaxis])
        dx = xGridFlat - xSrc[srcNo]
        dy = yGridFlat - ySrc[srcNo]
        distGrid = np.sqrt(dx**2 + dy**2)
        azGridV = np.arctan2(dx, dy)  # QSEIS convention: North = +x
        # now each horizontal source has a azimuth so do that correction
        azGridH = azGridV - azSrc[srcNo];
        
        # 2) Interpolate GF along horizontal distance for all components
        xxInterp = np.zeros((nFreq, nRec, ncomp), dtype = 'complex')
        for icomp in range(ncomp):
            f = interp1d(distVec, xxWFFT[idxFreq, :, icomp], kind='linear', bounds_error=False, fill_value=0.0, axis=1)
            xxInterp[:, :, icomp] = f(distGrid)*phaseFactor
            
        # 3) Rotate and sum contributions
        # Components: [fh-2.tz, fh-2.tr, fh-2.tt, fz-2.tz, fz-2.tr]

        # Horizontal force
        dispHForce[:, :, 0] += xxInterp[:, :, 0]  # Z
        dispHForce[:, :, 1] += xxInterp[:, :, 1] * np.sin(azGridH) + xxInterp[:, :, 2] * np.cos(azGridH)  # X
        dispHForce[:, :, 2] += xxInterp[:, :, 1] * np.cos(azGridH) - xxInterp[:, :, 2] * np.sin(azGridH)  # Y
        
        # Vertical force
        dispVForce[:, :, 0] += xxInterp[:, :, 3]  # Z
        dispVForce[:, :, 1] += xxInterp[:, :, 4] * np.sin(azGridV)  # X
        dispVForce[:, :, 2] += xxInterp[:, :, 4] * np.cos(azGridV) # Y
    
    return dispHForce, dispVForce, freqOut
    
def checkSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc, applyRot, thickness, vP, vS, rho, fMin, fMax, idxFreq=None, freqOut=None):
    """
    simply implements computeSurfCC for the two points specified as xGrid, yGrid
    additionally computes the bessel function for the Rayleigh and Love waes
    a plot is created to check if they match
    must match in all components if applyRot is set to 0,
    only the Z component will match if applyRot is set to 1.
    """
    ccRealZ, ccRealX, ccRealY, freqUse = computeSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc, applyRot, fMin, fMax, idxFreq, freqOut);
    recDist = yGrid[1] - yGrid[0];
    j0R, j0L, fJ = getBessel(thickness,vP,vS,rho,fMin,fMax,recDist)
    
    plotUtils.plotCC(ccRealZ, ccRealX, ccRealY, freqUse, j0R, j0L, fJ);
    
def computeSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phiSrc, ampSrc, applyRot, fMin, fMax, idxFreq=None, freqOut=None):
    """
    Simulate displacement for many sources (single-depth case) in a fully vectorized way
    then compute the correlation between two points on the surface
    this is a serial version of the code and meant to work for just two points
    so xGrid and yGrid have length 2
    in case xGrid and yGrid have much larger length, the first two entries are considered
    
    Parameters
    ----------
    xxW : np.ndarray
        Green's function matrix of shape (nt, nx, ncomp)
    tVec : np.ndarray
        Time vector
    distVec : np.ndarray
        Distance samples for xxW
    xGrid, yGrid : np.ndarray
        Receiver coordinates (ideally of length 2)
    xSrc, ySrc : np.ndarray
        Source x, y positions (1D arrays of length nSrc)
    applyRot can be 0 or 1
    if applyRot is 0, then radial and transverse components are not rotated, in that case
    theoretical bessel functions for Love and Rayleigh waves must match simulation
    if applyRot is 1, then new X and Y components will not match the theoretical Bessel function
    but the Z component will match, same as for applyRot = 0, Z component is invariant
    
    Returns
    -------
    ccRealZ, ccRealX, ccRealY : np.ndarray
        Simulated frequency domain normalized cross-correlation between the two points
        freqUse: frequency vector corresponding to the cross-correlation
    """
    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)
    
    # Flatten receiver grids if they are 2D
    xGridFlat = np.ravel(xGrid)
    yGridFlat = np.ravel(yGrid)

    dt = tVec[1]-tVec[0];
    freqUse = np.fft.rfftfreq(nt,dt);
    nFreq = len(freqUse);
    
    # perform FFT and perform entire operation frequency domain
    xxWFFT = np.fft.rfft(xxW,axis=0);

    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
        
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0];
    #nFreq = len(idx);
    #freqOut = freqUse[idx];
    #df_native = freqUse[1] - freqUse[0]
    #df_target = df  # or user input
    #step = max(1, int(round(df_target / df_native)))
    #idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    #freqOut = freqUse[idx]
    nFreq = len(idxFreq);
    
    ccAAZ = np.zeros((nFreq,))
    ccBBZ = np.zeros((nFreq,))
    ccABZ = np.zeros((nFreq,))

    ccAAX = np.zeros((nFreq,))
    ccBBX = np.zeros((nFreq,))
    ccABX = np.zeros((nFreq,))

    ccAAY = np.zeros((nFreq,))
    ccBBY = np.zeros((nFreq,))
    ccABY = np.zeros((nFreq,))
    
    for srcNo in range(nSrc):
        # 1) Compute distance and azimuth from current source to all receivers
        phaseFactor = ampSrc[srcNo, :][:, np.newaxis] * \
                  np.exp(-1j * phiSrc[srcNo, :][:, np.newaxis])
        dx = xGridFlat - xSrc[srcNo]
        dy = yGridFlat - ySrc[srcNo]
        distGrid = np.sqrt(dx**2 + dy**2)
        azGridV = np.arctan2(dx, dy)  # QSEIS convention: North = +x
        # now each horizontal source has a azimuth so do that correction
        azGridH = azGridV - azSrc[srcNo];
        
        # 2) Interpolate GF along horizontal distance for all components
        xxInterp = np.zeros((nFreq, nRec, ncomp),dtype='complex')
        for icomp in range(ncomp):
            f = interp1d(distVec, xxWFFT[idxFreq, :, icomp], kind='linear', bounds_error=False, fill_value=0.0, axis=1)
            xxInterp[:, :, icomp] = f(distGrid)*phaseFactor
            
        # 3) Rotate and sum contributions
        # Components: [fh-2.tz, fh-2.tr, fh-2.tt, fz-2.tz, fz-2.tr]

        if(applyRot):
            uHZ = xxInterp[:,:,0];
            uHX = xxInterp[:, :, 1] * np.sin(azGridH) + xxInterp[:, :, 2] * np.cos(azGridH)  # X
            uHY = xxInterp[:, :, 1] * np.cos(azGridH) - xxInterp[:, :, 2] * np.sin(azGridH)  # Y
        else:
            uHZ = xxInterp[:,:,0];
            uHX = xxInterp[:, :, 2];
            uHY = xxInterp[:, :, 1];

        # populate the cross-correlation
        ccAAZ = ccAAZ + uHZ[:,0]*np.conjugate(uHZ[:,0]);
        ccBBZ = ccBBZ + uHZ[:,1]*np.conjugate(uHZ[:,1]);
        ccABZ = ccABZ + uHZ[:,0]*np.conjugate(uHZ[:,1]);

        ccAAX = ccAAX + uHX[:,0]*np.conjugate(uHX[:,0]);
        ccBBX = ccBBX + uHX[:,1]*np.conjugate(uHX[:,1]);
        ccABX = ccABX + uHX[:,0]*np.conjugate(uHX[:,1]);
        
        ccAAY = ccAAY + uHY[:,0]*np.conjugate(uHY[:,0]);
        ccBBY = ccBBY + uHY[:,1]*np.conjugate(uHY[:,1]);
        ccABY = ccABY + uHY[:,0]*np.conjugate(uHY[:,1]);

    # compute the normalized cross-correlation
    ccRealZ = np.real(ccABZ/np.sqrt(ccAAZ*ccBBZ));
    ccRealX = np.real(ccABX/np.sqrt(ccAAX*ccBBX));
    ccRealY = np.real(ccABY/np.sqrt(ccAAY*ccBBY));
    
    return ccRealZ, ccRealX, ccRealY, freqOut
    
def getBessel(thickness,vP,vS,rho,fMin,fMax,recDist):
    """
    computes the bessel function J0(2*pi*d*f/v)
    thickness, vP, vS, and rho are used for calculating the love and Rayleigh
    wave phase velocities
    computation is done in the frequency band fMin to fMax
    recDist is the distance between the two receivers in meters
    """
    freqs = np.arange(fMin,fMax,0.1);
    
    periods = 1/freqs;

    vDispRay = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(freqs)
    #print(vDispRay)
    vDispLove = surf96(thickness,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
    #print(vDispLove)
    j0R = j0(2*np.pi*recDist*freqs/vDispRay);
    j0L = j0(2*np.pi*recDist*freqs/vDispLove);

    return j0R, j0L, freqs

def getBesselN(thickness,vP,vS,rho,fMin,fMax,recDist,n):
    """
    computes the bessel function J0(2*pi*d*f/v)
    thickness, vP, vS, and rho are used for calculating the love and Rayleigh
    wave phase velocities
    computation is done in the frequency band fMin to fMax
    recDist is the distance between the two receivers in meters
    """
    freqs = np.arange(fMin,fMax,0.1);
    
    periods = 1/freqs;

    vDispRay = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(freqs)
    #print(vDispRay)
    vDispLove = surf96(thickness,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
    #print(vDispLove)
    jnR = jv(n,2*np.pi*recDist*freqs/vDispRay);
    jnL = jv(n,2*np.pi*recDist*freqs/vDispLove);

    return jnR, jnL, freqs

def genAmbSrc(nSrc,
              mode="ring",
              R1=2000.0, R2=2500.0,
              xMin=-1000.0, xMax=1000.0,
              yMin=-1000.0, yMax=1000.0,
              randomPhase=True,
              freqDependent=True,
                nFreq=None):
    """
    Generate ambient noise sources, each with a random amplitude and phase per frequency bin.
    operates in three modes: ring, internal, inDisk
    ring = ring of sources outside the sensor array defined by inner and out radius of R1 and R2
    internal = sources randomly distributed within the receiver grid
    inDisk = sources distributed within a disk of radius R1 and R2, uniform area distribution
    xMin, xMax, yMin, yMax are the simulation domain applicabe to mode=internal
    randomPhase and frequencyDependent set to True sd default
    nFreq = number of frequency bins to be analyzed (generated from getFreqGrid)
    Returns
    -------
    xSrc, ySrc, azSrc : (nSrc,)
    phiSrc : (nSrc, nFreq)
    ampSrc : (nSrc, nFreq)
    """

    rng = np.random.default_rng()

    # --- positions ---
    if mode == "ring":
        theta = rng.uniform(0, 2*np.pi, nSrc)
        r = rng.uniform(R1, R2, nSrc)
        xSrc, ySrc = r*np.cos(theta), r*np.sin(theta)
    elif mode == "internal":
        xSrc = rng.uniform(xMin, xMax, nSrc)
        ySrc = rng.uniform(yMin, yMax, nSrc)
    elif mode == "inDisk":
        theta = rng.uniform(0, 2*np.pi, nSrc)
        # area-uniform radii
        r = np.sqrt(rng.uniform(R1**2, R2**2, nSrc))
        xSrc = r*np.cos(theta); ySrc = r*np.sin(theta)
    else:
        raise ValueError("mode must be 'ring' or 'internal'")

    azSrc = rng.uniform(0, 2*np.pi, nSrc)

    # --- random phase matrix ---
    if randomPhase:
        phiSrc = 2*np.pi * rng.random((nSrc, nFreq))
    else:
        phiSrc = np.zeros((nSrc, nFreq))

    # --- amplitude matrix ---
    if freqDependent:
        ampSrc = rng.random((nSrc, nFreq))
    else:
        ampSrc = rng.random((nSrc, 1)) * np.ones((nSrc, nFreq))

    return xSrc, ySrc, azSrc, phiSrc, ampSrc


def getFreqGrid(tTot, nSamp, fMin, fMax, df):
    """
    Compute the FFT frequency grid and select bins within [fMin, fMax]
    subsampled to ~df_target.

    Returns
    -------
    freqOut : ndarray
        Selected frequency bins (Hz)
    idx : ndarray
        Indices of those bins in the native FFT grid
    df_native : float
        Native FFT frequency spacing
    """

    dt = tTot / nSamp
    freqUse = np.fft.rfftfreq(nSamp, dt)
    print(freqUse[0:5])
    df_native = freqUse[1] - freqUse[0]
    step = max(1, int(round(df / df_native)))
    idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    freqOut = freqUse[idx]
    return freqOut, idx, df_native
    
def computeDistAzi(xGrid, yGrid, xSrc, ySrc):
    """
    Compute distance and azimuth from a source to all grid points.

    Parameters
    ----------
    xGrid : np.ndarray
        2D array of x-coordinates of the grid (shape: ny x nx)
    yGrid : np.ndarray
        2D array of y-coordinates of the grid (shape: ny x nx)
    xSrc : float
        x-coordinate of the source
    ySrc : float
        y-coordinate of the source

    Returns
    -------
    distVec : np.ndarray
        1D array of distances from source to all grid points (flattened)
    azimuthVec : np.ndarray
        1D array of azimuths (radians) from source to all grid points (flattened)
    """
    dx = xGrid - xSrc
    dy = yGrid - ySrc

    distVec = np.sqrt(dx**2 + dy**2).ravel()        # Flattened distance vector
    # note that this is dx/dy because QSEIS operates like that, North is +x
    azimuthVec = np.arctan2(dx, dy).ravel()        # Flattened azimuth vector

    return distVec, azimuthVec
    
if __name__ == "__main__":
    main()


# In[ ]:




