#!/usr/bin/env python

import os
from os.path import join as pjoin

import seaborn
seaborn.set_style("dark")
import numpy as np
import nibabel as nib
from mako.lookup import TemplateLookup
makolookup = TemplateLookup(directories=['./tpl'])

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


import shimmingfuncs as sh
import stabilityfuncs as sf
import scipy as sy
import PIL
from PIL import Image as img




def Shimming(dirname, filename, shift, radius):


    #logging.debug('dirname: {}\n'            
                  #'filename: {}\n'
                  #'shift: {}\n'
                  #'radius: {}'.format(dirname, filename, shift, radius))

    nim = nib.load(pjoin(dirname, filename))
    nim_hdr = nim.get_header()
    xdim, ydim, slicethickness,tr = nim_hdr['pixdim'][1:5]   # per gli fBirn
    #xdim, ydim, slicethickness, tr = nim_hdr['pixdim'][0:4]    # per gli ACR
    xsize, ysize, numslices = nim_hdr['dim'][1:4]

    dims = nim.get_data().shape
    if len(dims) == 4:
        # normal case
        selecteddata = nim.get_data()[:, :, :, 1:].transpose(3, 2, 1, 0)
    elif len(dims) == 3:
        # single slice
        selecteddata = nim.get_data()[:, :, 1:]
        # nifti loses the z dimension; we must reconstitute it
        sdshape = selecteddata.shape
        selecteddata = selecteddata.reshape(sdshape[0], sdshape[1], sdshape[2]).transpose(2, 1, 0)

    #numtimepoints = selecteddata.shape[0]
    #meanslice = np.mean(selecteddata, 0)
    meanslice = selecteddata

    (threshguess, threshfrac) = sf.findsepval(meanslice)
    initmask = sf.makemask(meanslice, threshfrac, 0)
    threshmean = sf.getnzfracval(initmask * meanslice, 0.02)
    objectmask = sf.makemask(meanslice, threshmean, 1)
    
    slicecenter = sf.findCOM(objectmask)
    zcenterf = slicecenter[2]
    zcenter = int(round(zcenterf))   
    slicecenter = sf.findCOM(objectmask[zcenter + shift, :, :])
    xcenterf = slicecenter[0]
    ycenterf = slicecenter[1]
    xcenter, ycenter, zcenter = [int(round(x)) for x in (xcenterf, ycenterf, zcenterf)]

    #meanimage = meanslice*objectmask
    central_slice = objectmask[zcenter + shift,:,:] 
 
    #centroid = ndimage.measurements.center_of_mass(central_slice)
    centroid = [xcenterf, ycenterf ]
    h, w = central_slice.shape[:2]

    #r=5.0
    mask = sh.createCircularMask(h, w, centroid, radius)
    Ntot = np.count_nonzero(central_slice == 1)
    Nr = np.count_nonzero(mask == 1)
    while float(Ntot-Nr)>0.1:
        ratio = float(Ntot)/float(Nr)
        radius = radius + ratio*0.02
        mask = sh.createCircularMask(h, w, centroid , radius)
        mask_product = mask * central_slice
        Nr = np.count_nonzero(mask_product == 1)
    

    diff = (mask - central_slice)
    Ndiff = np.count_nonzero(diff == 1)   

    splitfile = os.path.splitext(os.path.basename(filename))
    diff1 = (2*mask - central_slice)

    return [diff1 , Ndiff]



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calculate stability values and create the output web page.')
    parser.add_argument('dirname', help='the PROCRESULT directory ')
    parser.add_argument('filename', help='the name of the 4D NIFTI file')
    parser.add_argument('shift', help='slice shift by BOTTYLUC')
    parser.add_argument('radius', help='radius by BOTTYLUC')
    
    args = parser.parse_args()




