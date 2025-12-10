#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('Activating plot modiule')
    xP = 0; yP = 0;
    outDir = '/data/gravwav/koley/OutDisp/Depth0p00/'
    plotDispSpect(outDir,xP,yP)
    plotDispSpectMemap(outDir, fTarget=2)
    
def plotPSDDeepSurfMulti(freq, data, depth_label, color, fig=None, axs=None, quantity="ASD"):

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            1, 3, figsize=(12, 3),
            sharey=True,              # <<< THIS FIXES THE OVERLAP
            constrained_layout=True   # better than tight_layout
        )

        component_titles = ["Z Component", "X Component", "Y Component"]
        for i, ax in enumerate(axs):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            if i == 0:                # <<< Only left subplot gets the y-label
                ax.set_ylabel(quantity)
            ax.grid(True)

    for i in range(3):
        axs[i].semilogy(freq, np.abs(data[:, i]), color=color, label=depth_label)
        axs[i].legend(loc="upper right")

    return fig, axs

def plotCCDeepSurfMulti(freq, ccZ, ccX, ccY, depth_label, color, fig=None, axs=None, quantity="real(CC)"):
    """
    Plot 3-component (Z, X, Y) spectra vs frequency on shared subplots.

    Returns
    -------
    fig, axs : updated figure and axes
    """

    # Create figure if not passed
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True,constrained_layout=True)
        component_titles = ["Z Component", "X Component", "Y Component"]
        for i, ax in enumerate(axs):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(quantity)
            ax.grid(True)

    # Plot data on the same axes
    #labels = ["X", "Y", "Z"]
    axs[0].plot(freq, ccZ, color=color, label=depth_label)
    axs[0].legend(loc="upper left")
    axs[1].plot(freq, ccX, color=color, label=depth_label)
    axs[1].legend(loc="upper left")
    axs[2].plot(freq, ccY, color=color, label=depth_label)
    axs[1].legend(loc="upper left")
    #plt.tight_layout();
    return fig, axs

    
def plotCC(ccRealZ, ccRealX, ccRealY, freqOut, j0R=None, j0L=None, fJ=None):

    # Create figure with shared y-axis and better spacing
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    # --- Z Component ---
    axs[0].plot(freqOut, ccRealZ, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[0].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    
    axs[0].set_xlim([0, 8])
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel(r"$\langle XY^*\rangle / \sqrt{\langle XX^*\rangle \langle YY^*\rangle}$")
    axs[0].set_title("Z component")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    # --- X Component ---
    axs[1].plot(freqOut, ccRealX, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[1].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    
    axs[1].set_xlim([0, 8])
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_title("X component")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    # --- Y Component ---
    axs[2].plot(freqOut, ccRealY, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[2].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    axs[2].set_xlim([0, 8])
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_title("Y component")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    return fig, axs
    
    
def plotDispSpect(outDir, xP, yP):
    """
    Plot absolute value of displacement spectra for a point (xP, yP)
    on a depth slice (3 components, H-force & V-force).
    """

    # ---- Load receiver grid ----
    data = np.load(os.path.join(outDir, "receiverGrid.npz"))
    xGrid = data["xGrid"]
    yGrid = data["yGrid"]
    freqOut = data["freqOut"]
    nFreq = data["nFreq"]
    nRec  = len(xGrid)

    # ---- Load displacement fields ----
    dispHForce = np.memmap(
        os.path.join(outDir, "dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir, "dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )

    # ---- Get receiver index ----
    rec_index, distRec = findRecIndex(xP, yP, xGrid, yGrid)

    dispH = np.abs(dispHForce[:, rec_index, :])
    dispV = np.abs(dispVForce[:, rec_index, :])

    # ---- Plotting ----
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    component_titles = ["Z component", "X component", "Y component"]

    # Loop through components
    for i in range(3):

        axs[i].semilogy(freqOut, dispH[:, i], "b", label="H-force")
        axs[i].semilogy(freqOut, dispV[:, i], "r", label="V-force")

        axs[i].set_xlim([0, 8])
        axs[i].set_xlabel("Frequency (Hz)")
        axs[i].set_title(component_titles[i])
        axs[i].grid(True)
        axs[i].legend(loc="upper right")

        if i == 0:
            axs[i].set_ylabel(r"$|\,\mathrm{disp}(f)\,|$")

    return fig, axs

def plotDispSpectAllRea(dispPointFull, freqOut):
    """
    Plot absolute value of displacement spectra for a point (after many realizations),
    showing Z, X, Y components on shared subplots.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Create improved figure layout
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    component_titles = ["Z component", "X component", "Y component"]

    for i in range(3):
        axs[i].semilogy(freqOut, np.abs(dispPointFull[:, i]), 'b', label="V+H force")

        axs[i].set_xlim([0, 8])
        axs[i].set_xlabel("Frequency (Hz)")
        axs[i].set_title(component_titles[i])
        axs[i].grid(True)
        axs[i].legend(loc="upper right")

        if i == 0:
            axs[i].set_ylabel(r"$|\,\mathrm{disp}(f)\,|$")

    return fig, axs
    
def plotDispSurf(outDir, fTarget=None):
    """
    Plot a surface map of |displacement| at a chosen frequency.
    """

    # --- Load receiver grid ---
    data = np.load(os.path.join(outDir, "receiverGrid.npz"))
    xGrid = data["xGrid"]
    yGrid = data["yGrid"]
    freqOut = data["freqOut"]
    nFreq = int(data["nFreq"])
    nRec = len(xGrid)

    # --- Open memmap files ---
    dispHForce = np.memmap(
        os.path.join(outDir, "dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir, "dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )

    # --- Optional surface maps ---
    if fTarget is not None:
        iFreq = np.argmin(np.abs(freqOut - fTarget))
        fSel = freqOut[iFreq]
        print(f"[plotDispSpectMemap] Plotting field magnitude at {fSel:.2f} Hz")

        nx = ny = int(np.sqrt(nRec))
        X = xGrid.reshape(nx, ny)
        Y = yGrid.reshape(nx, ny)

        magH = np.abs(dispHForce[iFreq, :, :]).reshape(nx, ny, 3)
        magV = np.abs(dispVForce[iFreq, :, :]).reshape(nx, ny, 3)
        comps = ["Z", "X", "Y"]

        # --- Horizontal Force field ---
        figH, axsH = plt.subplots(1, 3, figsize=(9,3))
        for i in range(3):
            im = axsH[i].pcolormesh(X, Y, magH[:, :, i], shading="auto", cmap="viridis")
            axsH[i].set_title(f"H-Force {comps[i]}")
            axsH[i].set_xlabel("x (m)")
            axsH[i].set_ylabel("y (m)")
            figH.colorbar(im, ax=axsH[i])
        figH.suptitle(f"|Displacement| (Horizontal Force) at {fSel:.2f} Hz")
        figH.tight_layout()

        # --- Vertical Force field ---
        figV, axsV = plt.subplots(1, 3, figsize=(9,3))
        for i in range(3):
            im = axsV[i].pcolormesh(X, Y, magV[:, :, i], shading="auto", cmap="viridis")
            axsV[i].set_title(f"V-Force {comps[i]}")
            axsV[i].set_xlabel("x (m)")
            axsV[i].set_ylabel("y (m)")
            figV.colorbar(im, ax=axsV[i])
        figV.suptitle(f"|Displacement| (Vertical Force) at {fSel:.2f} Hz")
        figV.tight_layout()

    plt.show()

    
def plotSrcRec(xSrc, ySrc, xGrid, yGrid, pA):
    
    fig1, axs1 = plt.subplots(1, 1, figsize=(4,4))
    axs1.plot(xSrc,ySrc,'b*',label="Source locations");

    minX = np.min(xGrid); maxX = np.max(xGrid);
    minY = np.min(yGrid); maxY = np.max(yGrid);
    simBoundX = np.array((minX,maxX,maxX,minX,minX));
    simBoundY = np.array((minY,minY,maxY,maxY,minY));

    axs1.plot(simBoundX, simBoundY, 'r', label= "Receiver boundary");
    axs1.plot(pA[0],pA[1], 'k*', label = "Point A");
    axs1.legend(loc='best');
    axs1.set_xlabel('X coordinate (m)');
    axs1.set_ylabel('Y coordinate (m)');
    axs1.set_title('Source-receiver locations');
    fig1.tight_layout();
    plt.show();
    
def findRecIndex(x1, y1, xGrid, yGrid, xVec=None, yVec=None):
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
    

if __name__ == "__main__":
    main()


# In[ ]:




