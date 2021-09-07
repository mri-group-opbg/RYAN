#!/usr/bin/env python 

from os.path import join as pjoin
import matplotlib
import seaborn
seaborn.set_style("dark")
# matplotlib.use('Agg', warn=False)
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import stabilityfuncs as sf
import CBIRobustFit as CBIRF

## Search for Spikes

def SpikeDetection(theimage):

    thold = 9
    spikesroisize = 5
    spikesxpos = int(int(spikesroisize / 2.0) + 1)
    spikesypos = int(int(spikesroisize / 2.0) + 1)
    spikesroi = sf.setroilims(spikesxpos, spikesypos, spikesroisize)
    
    numslice = theimage.shape[1]
    numvolume = theimage.shape[0]
    indexslice = np.arange(numslice)
    spikemeants = np.zeros((numvolume, numslice))
    peaks_ts = np.zeros((numvolume, numslice))
    peaks_nspk = np.zeros((numslice,1))
    onesNt = np.ones((numvolume, 1))
    
    for z in indexslice:
        spikemeants[:,z] = sf.getroimeantc(theimage, spikesroi, z) 
        m_meants = np.mean(spikemeants[:,z], axis = 0)
        if not spikemeants[:,z].any() == m_meants:
            row = spikemeants[:,z]
            row = row.reshape(-1,1)        
            m_meants = CBIRF.CBIrobustfit(onesNt,row)
            s_meants = np.sqrt(CBIRF.CBIrobustfit(onesNt,(row-m_meants)**2))
           
            #peak_ts
            soglia = m_meants + thold*s_meants
            index = (row > soglia).nonzero()[0]
            peaks_ts[index,z] = 1
            
            #peak_nspk
            peak = peaks_ts[:,z]
            peak = peak.reshape(-1,1)
            peak_ind = peak.nonzero()[0] 
            peaks_nspk[z] = peak_ind.shape
            
    #peak_slices
    peaks_slices = (indexslice+1).reshape(-1,1)

    return [spikemeants, peaks_ts, peaks_nspk, peaks_slices] 







