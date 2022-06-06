#!/usr/bin/env python -W

"""stabilityfuncs.py: provides general helper functions, especially for stabilitycalc"""
import pkg_resources
from distutils.version import LooseVersion

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ASSERTVERSION(module, minver):
    # check requirements
    if not LooseVersion(pkg_resources.get_distribution(module).version) >= LooseVersion(minver):
        raise ImportError('Module {} is too old, need at least version {}.'.format(module, minver))

ASSERTVERSION('seaborn', '0.5.1')
ASSERTVERSION('mako', '1.0.0')
ASSERTVERSION('nibabel', '2.0.0')

import csv
from collections import OrderedDict
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker

from collections import namedtuple
import configparser
import copy 
from os.path import join, exists, isdir, isfile
from os import listdir, walk, system
from pathlib import Path
from sys import platform
import subprocess


import logging
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d: %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S'
    )
logging.getLogger('matplotlib').setLevel(logging.ERROR)


Stats = namedtuple('Stats', 'mean stddev var max min ptp')

def dict_from_tsvfile(filename):
    """open a tab separated two column file, return it as a str->str dict"""

    d = {}
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            d[row[0]] = row[1]

    return d


def stabilityparms(option, section='paths'):
    """
    retrieve a value from the configuration file.
    for legacy use, assume we're looking for a path.
    """

    config = configparser.ConfigParser()

    try:
        config.read('config/stability.ini')
    except IOError:
        logging.critical('Failed to open configuration file.')
        exit(1)

    try:
        return config.get(section, option.lower())
    except (configparser.NoOptionError, configparser.NoSectionError):
        return ''


def getlimits(coil):
    # noinspection PyShadowingNames
    def get_lim_csv(coil):
        d = {}
        try:
            with open('config/{}_limits.csv'.format(coil)) as csvfile:
                reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
                for r in reader:
                    d[r['var']] = r
        except IOError:
            logging.critical("getlimits: coil not recognized!")
            exit(1)
        return d

    limitdict = get_lim_csv('coil_independent')
    limitdict.update(get_lim_csv('sample'))
    limitdict.update(get_lim_csv(coil))

    return limitdict


def getphasedarraydata(coil):
    d = OrderedDict()
    try:
        with open('config/{}_coildata.csv'.format(coil)) as csvfile:
            reader = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            for r in reader:
                d[r['element']] = r
        return d
    except IOError:
        logging.debug("getphasedarraydata: Not a phased array.")
        return False


def makemask(inputim, inputthresh, useabs):
    if useabs < 1:
        thethreshval = getfracval(inputim, inputthresh)
    else:
        thethreshval = inputthresh
    themask = sp.where(inputim > thethreshval, 1.0, 0.0)
    return themask


def vecnorm(thevec):
    return np.sqrt(np.square(thevec).sum())


def formatlimits(lim):
    return '"{}",{},{},{},{}'.format(lim['description'],
                                     lim['warn_min'],
                                     lim['good_min'],
                                     lim['good_max'],
                                     lim['warn_max'])


def limitcheck(n, lim):
    # check to see if a parameter falls within preset limits.

    # check for legacy input mode and convert if needed
    if type(lim) is tuple:
        lim = {'good_min': lim[0][0], 'good_max': lim[0][1], 'warn_min': lim[1][0], 'warn_max': lim[1][1]}

    retval = 2  # start with the assumption that the data is bad
    if (float(n) >= float(lim['warn_min'])) and (float(n) <= float(lim['warn_max'])):
        retval = 1  # number falls within the warning limits
    if (float(n) >= float(lim['good_min'])) and (float(n) <= float(lim['good_max'])):
        retval = 0  # number falls within the good limits
    return retval


def trendgen(thexvals, thefitcoffs):
    # generate the polynomial fit timecourse from the coefficients
    theshape = thefitcoffs.shape
    order = theshape[0] - 1
    thepoly = thexvals
    thefit = 0.0 * thexvals
    if order > 0:
        for i in range(1, order + 1):
            thefit = thefit + thefitcoffs[order - i] * thepoly
            thepoly = np.multiply(thepoly, thexvals)
    return thefit


def completerobust(thearray):
    # calculate the robust range of the all voxels
    themin = getfracval(thearray, 0.02)
    themax = getfracval(thearray, 0.98)
    return [themin, themax]


def nzrobust(thearray):
    # calculate the robust range of the non-zero voxels
    themin = getnzfracval(thearray, 0.02)
    themax = getnzfracval(thearray, 0.98)
    return [themin, themax]


def nzminmax(thearray):
    # calculate the min and max of the non-zero voxels
    flatarray = np.ravel(thearray)
    nzindices = np.nonzero(flatarray)
    theflatarray = flatarray[nzindices]
    themax = np.max(theflatarray)
    themin = np.min(theflatarray)
    return [themin, themax]


def nzstats(thearray):
    # calculate the stats of the non-zero voxels
    flatarray = np.ravel(thearray)
    nzindices = np.nonzero(np.ravel(thearray))
    return Stats(np.mean(flatarray[nzindices]),
                 np.std(flatarray[nzindices]),
                 np.var(flatarray[nzindices]),
                 np.max(flatarray[nzindices]),
                 np.min(flatarray[nzindices]),
                 np.ptp(flatarray[nzindices]))


def showtc(thexvals, theyvals, thelabel):
    # show an roi timecourse plot
    w, h = plt.figaspect(0.25)
    roiplot = plt.figure(figsize=(w, h))
    roisubplot = roiplot.add_subplot(111)
    roisubplot.plot(thexvals, theyvals, 'b')
    roisubplot.grid(True)
    # roisubplot.axes.Subplot.set_pad(0.1)
    for tick in roisubplot.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
    for tick in roisubplot.yaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
    roisubplot.set_title(thelabel, fontsize=30)


def showtc2(thexvals, theyvals, thefitvals, thelabel, thexlabel= None):
    # show an roi timecourse plot and a fit line
    w, h = plt.figaspect(0.25)
    roiplot = plt.figure(figsize=(w, h))
    roisubplot = roiplot.add_subplot(111)
    roisubplot.plot(thexvals, theyvals, 'b', thexvals, thefitvals, 'g')
    roisubplot.set_xlabel(thexlabel, fontsize=20)
    roisubplot.grid(True)
    # roisubplot.axes.Subplot.set_pad(0.1)
    for tick in roisubplot.xaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
    for tick in roisubplot.yaxis.get_major_ticks():
        tick.label1.set_fontsize(20)
    roisubplot.set_title(thelabel, fontsize=30)


def showweisskoff(theareas, thestddevs, theprojstddevs, thelabel):
    # initialize and show a loglog Weiskoff plot
    logging.debug("\nGenerating plot for {}".format(thelabel))
    w, h = plt.figaspect(1.0)
    roiplot = plt.figure(figsize=(w, h))
    roiplot.subplots_adjust(hspace=0.35)
    roisubplot = roiplot.add_subplot(111)
    thestddevs += 0.00000001
    roisubplot.loglog(theareas, thestddevs, 'r', theareas, theprojstddevs, 'k')
    roisubplot.grid(True)

def showweisskoff2(theareas, thestddevs, theprojstddevs, thelabel, title): 
    # initialize and show a loglog Weiskoff plot
    logging.debug("Generating plot for {}".format(thelabel))
    w, h = plt.figaspect(1.0)
    roiplot = plt.figure(figsize=(w, h))
    roiplot.subplots_adjust(hspace=0.35)
    roisubplot = roiplot.add_subplot(111)
    thestddevs += 0.00000001
    roisubplot.set_title(title)
    roisubplot.loglog(theareas, thestddevs, 'r', theareas, theprojstddevs, 'k')
    roisubplot.grid(True)


def showimage(data, thetitle, thexvals, theyvals, thelabel): 
    logging.debug("Generating plot for {}".format(thelabel))
    w, h = plt.figaspect(0.5)
    roiplot = plt.figure(figsize=(w, h))
    plt.plot(data)
    plt.title(thetitle)
    plt.xlabel(thexvals)
    plt.ylabel(theyvals)

def showimage_mod(data, thexvals, theyvals, thelabel, spikeslice): 
    logging.debug("Generating plot for {}".format(thelabel))
    w, h = plt.figaspect(0.5)
    spikeplot = plt.figure(figsize=(w, h))
    for i in range(len(spikeslice)): 
        spikesubplot = spikeplot.add_subplot(len(spikeslice),1,i+1)
        spikesubplot.plot(data[:,i])
        spikesubplot.tick_params(axis='both', which='major', labelsize=8)
        spikesubplot.set_ylim(int(min(data[:,i]))-1, int(max(data[:,i]))+2)
        spikesubplot.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=1))
        spikesubplot.set_ylabel("S %d" %spikeslice[i], fontsize=8, weight='bold')
        spikeplot.subplots_adjust(hspace = 1)
    
    spikeplot.supxlabel(thexvals)
    spikeplot.supylabel(theyvals)

def showimage2(data, thetitle, thelabel):
    logging.debug("Generating plot for {}".format(thelabel))
    w, h = plt.figaspect(1.0)
    roiplot = plt.figure(figsize=(w, h))
    plt.imshow(data)
    plt.title(thetitle)

def showimage3(data, angle, thetitle): 
    fig = plt.figure(figsize=(60, 60))
    index = np.arange(9)
    for i in index:
         plt.subplot(3,3,i+1)
         j=int(round(i*3.5))
         angolo = angle[j]
         plt.plot(data[j,:])
         s=str(angle[j])
         plt.title('Angle ' +s, fontsize=50)        
         xmax=len(data[j,:])
         xmin=0
         ymax=max(data[j,:])
         ymin=min(data[j,:])
         plt.xticks([xmin,xmax], fontsize=40)
         plt.yticks([ymax,ymax], fontsize=40, rotation = 'vertical')
    plt.title(thetitle)
    
def showimageangle(thexvals, angle, data, thefitvals, detrendingvals, i):
  
    w, h = plt.figaspect(0.75)
    roiplot = plt.figure(figsize=(w, h))
    roisubplot = roiplot.add_subplot(211)
    j=int(round(i*4))
    angolo = angle[j]
    roisubplot.plot(thexvals, data[j,:], 'b', thexvals, thefitvals[j] + detrendingvals[j,:], 'g', linewidth=0.7)    
    s=str(angle[j])
    roisubplot.set_title('Peripheral ROI at Angle ' +s)
    roisubplot.grid(True)
    roisubplot = roiplot.add_subplot(212)
    roisubplot.plot(thexvals, data[j,:]- detrendingvals[j,:], 'b', linewidth=0.7)
    roisubplot.set_title('Detrended Peripheral ROI at Angle ' +s)
    plt.subplots_adjust(hspace = 1)
    roisubplot.grid(True)

def check_zeros_corner(thevec):
     l = thevec.shape[0]
     index = np.arange(l)
     thevec_check = copy.deepcopy(thevec)
     for i in index:
         if thevec[i] == 0:
	    # if thevec(198)=0 set theveccheck=thevec(199) 
            if l<198:
               thevec_check[i] = (thevec[i-1]+ thevec[i+1])*0.5 
            else:
                thevec_check[i] = (thevec[i-1]+ thevec[i])*0.5 
         
     return thevec_check

def showslice2(thedata, thelabel, minval, maxval, colormap):
    # initialize and show a 2D slice from a dataset in greyscale

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    plt.figure(figsize=plt.figaspect(1.0))
    theshape = thedata.shape
    numslices = theshape[0]
    ysize = theshape[1]
    xsize = theshape[2]
    slicesqrt = int(np.ceil(np.sqrt(numslices)))
    theslice = np.zeros((ysize * slicesqrt, xsize * slicesqrt))
    for i in range(numslices):
        ypos = int(i / slicesqrt) * ysize
        xpos = int(i % slicesqrt) * xsize
        theslice[ypos:ypos + ysize, xpos:xpos + xsize] = thedata[i, :, :]
    if plt.isinteractive():
        plt.ioff()
    plt.axis('off')
    plt.axis('equal')
    plt.subplots_adjust(hspace=0.0)
    plt.axes([0, 0, 1, 1], frameon=False)
    if colormap == 0:
        thecmap = cm.gray
    else:
        mycmdata1 = {
            'red': ((0., 0., 0.), (0.5, 1.0, 0.0), (1., 1., 1.)),
            'green': ((0., 0., 0.), (0.5, 1.0, 1.0), (1., 0., 0.)),
            'blue': ((0., 0., 0.), (0.5, 1.0, 0.0), (1., 0., 0.))
        }
        thecmap = colors.LinearSegmentedColormap('mycm', mycmdata1)
    plt.imshow(theslice, vmin=minval, vmax=maxval, interpolation='nearest', label=thelabel, aspect='equal',
               cmap=thecmap)


def showslice3(thedata, thelabel, minval, maxval, colormap):
    # initialize and show a 2D slice from a dataset in greyscale

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)

    theshape = thedata.shape
    ysize = theshape[0]
    xsize = theshape[1]
    np.zeros((ysize, xsize))
    if plt.isinteractive():
        plt.ioff()
    plt.axis('off')
    plt.axis('equal')
    plt.subplots_adjust(hspace=0.0)
    plt.axes([0, 0, 1, 1], frameon=False)
    if colormap == 0:
        thecmap = cm.gray
    else:
        mycmdata1 = {
            'red': ((0., 0., 0.), (0.5, 1.0, 0.0), (1., 1., 1.)),
            'green': ((0., 0., 0.), (0.5, 1.0, 1.0), (1., 0., 0.)),
            'blue': ((0., 0., 0.), (0.5, 1.0, 0.0), (1., 0., 0.))
        }
        thecmap = colors.LinearSegmentedColormap('mycm', mycmdata1)
    plt.imshow(thedata, vmin=minval, vmax=maxval, interpolation='nearest', label=thelabel, aspect='equal', cmap=thecmap)


def smooth(x, window_len=11, window='hanning'):
    # this routine comes from a scipy.org Cookbook
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def findsepval(datamat):
    # Find the image intensity value that cleanly separates background from image
    numbins = 200
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    smoothhist = smooth(meanhist)
    currentpos = int(numbins * 0.05)
    for i in range(currentpos + 1, numbins):
        if smoothhist[i] < smoothhist[currentpos]:
            currentpos = i
        if smoothhist[i] > 1.2 * smoothhist[currentpos]:
            break
    cummeanhist = np.cumsum(meanhist)
    cumfrac = (1.0 * cummeanhist[currentpos]) / (1.0 * cummeanhist[numbins - 1])
    sepval = bins[currentpos]
    return [sepval, cumfrac]


def getfracval(datamat, thefrac):
    # Find the image intensity value which thefrac of the non-zero voxels in the image exceed
    numbins = 200
    themax = datamat.max()
    themin = datamat.min()
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    target = cummeanhist[numbins - 1] * thefrac
    for i in range(numbins):
        if cummeanhist[i] >= target:
            return bins[i]
    return 0.0


def getnzfracval(datamat, thefrac):
    # Find the image intensity value which thefrac of the non-zero voxels in the image exceed
    numbins = 200
    (themin, themax) = nzminmax(datamat)
    (meanhist, bins) = np.histogram(datamat, bins=numbins, range=(themin, themax))
    cummeanhist = np.cumsum(meanhist)
    target = cummeanhist[numbins - 1] * thefrac
    for i in range(numbins):
        if cummeanhist[i] >= target:
            return bins[i]
    return 0.0

# noinspection PyPep8Naming,PyUnresolvedReferences
def findCOM(datamat):
    # find the center of mass of a 2D or 3D image
    Mx = 0.0
    My = 0.0
    Mz = 0.0
    mass = 0.0
    arrdims = np.shape(datamat)

    if datamat.ndim == 2:
        for i in range(arrdims[0]):
            for j in range(arrdims[1]):
                val = datamat[i, j]
                My += (i * val)
                Mx += (j * val)
                mass += val
        COM = (Mx / mass, My / mass, 0.0)
    if datamat.ndim == 3:
        for i in range(arrdims[0]):
            for j in range(arrdims[1]):
                for k in range(arrdims[2]):
                    val = datamat[i, j, k]
                    Mz += (i * val)
                    My += (j * val)
                    Mx += (k * val)
                    mass += val
        COM = (Mx / mass, My / mass, Mz / mass)
    return COM


def markroi(theinputroi, zpos, roislice, theval):
    # given an roi and a position, mark an roi
    xstart = theinputroi[0][0]
    xend = theinputroi[1][0]
    ystart = theinputroi[0][1]
    yend = theinputroi[1][1]
    roislice[zpos, ystart:yend, xstart:xend] = theval

def setroilims(xpos, ypos, size):
    # given a location and a size, define the corners of an roi
    if (size % 2) == 0:
        halfsize = size / 2
        return (((int(round(xpos - halfsize)), int(round(ypos - halfsize))),
                 (int(round(xpos + halfsize)), int(round(ypos + halfsize)))))
    else:
        halfsize = (size - 1) / 2
        return (((int(round(xpos - halfsize)), int(round(ypos - halfsize))),
                 (int(round(xpos + halfsize + 1)), int(round(ypos + halfsize + 1)))))

def set3Droilims(xpos, ypos, zpos, size):
    # given a location and a size, define the corners of an 3D roi
    if (size % 2) == 0:
        halfsize = size / 2
        return (((int(round(xpos - halfsize)), int(round(ypos - halfsize)), int(round(zpos - halfsize))),
            (int(round(xpos + halfsize)), int(round(ypos + halfsize)), int(round(zpos + halfsize)))))
    else:
        halfsize = (size - 1) / 2
        return (((int(round(xpos - halfsize)), int(round(ypos - halfsize)), int(round(zpos - halfsize))),
            (int(round(xpos + halfsize + 1)), int(round(ypos + halfsize + 1)), int(round(zpos + halfsize + 1)))))

def getroisnr(theimage, theroi, zpos):
    # get an snr timecourse from the voxels of an roi
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    thesubreg = theimage[:, zpos, ystart:yend, xstart:xend]
    theshape = thesubreg.shape
    numtimepoints = theshape[0]
    themeans = np.zeros(numtimepoints)
    thestddevs = np.zeros(numtimepoints)
    themax = np.zeros(numtimepoints)
    themin = np.zeros(numtimepoints)
    thesnrs = np.zeros(numtimepoints)
    timeindex = np.arange(numtimepoints)
    for i in timeindex:
        themeans[i] = np.mean(np.ravel(thesubreg[i, :, :]))
        thestddevs[i] = np.std(np.ravel(thesubreg[i, :, :]))
        themax[i] = np.max(np.ravel(thesubreg[i, :, :]))
        themin[i] = np.min(np.ravel(thesubreg[i, :, :]))
        thesnrs[i] = themeans[i] / thestddevs[i]
    return thesnrs

def getroivoxels(theimage, theroi, zpos):
    # get all the voxels from an roi and return a 2d (time by space) array
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    thesubreg = theimage[:, zpos, ystart:yend, xstart:xend]
    theshape = thesubreg.shape
    numtimepoints = theshape[0]
    thevoxels = np.zeros((numtimepoints, theshape[1] * theshape[2]))
    timeindex = np.arange(numtimepoints)
    for i in timeindex:
        thevoxels[i, :] = np.ravel(thesubreg[i, :, :])
    return thevoxels

def getroistdtc(theimage, theroi, zpos):
    # get a standard deviation timecourse from the voxels of an roi
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    if zpos!= -1: #modifica
        thesubreg = theimage[:, zpos, ystart:yend, xstart:xend]
    else:
        thesubreg = theimage[:, ystart:yend, xstart:xend]
    theshape = thesubreg.shape
    numtimepoints = theshape[0]
    thestds = np.zeros(numtimepoints)
    timeindex = np.arange(numtimepoints)
    for i in timeindex:
        thestds[i] = np.std(np.ravel(thesubreg[i, :, :]))
    return thestds

def getroimeantc(theimage, theroi, zpos):
    # get an average timecourse from the voxels of an roi
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    thesubreg = theimage[:, zpos, ystart:yend, xstart:xend]
    theshape = thesubreg.shape
    numtimepoints = theshape[0]
    themeans = np.zeros(numtimepoints)
    timeindex = np.arange(numtimepoints)
    for i in timeindex:
        themeans[i] = np.mean(np.ravel(thesubreg[i, :, :]))
    return themeans

def get3Droimeantc(theimage, theroi): 
    # get an average timecourse from the voxels of an roi
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    zstart = theroi[0][2]
    zend = theroi[1][2]
    thesubreg = theimage[:, zstart:zend, ystart:yend, xstart:xend]
    theshape = thesubreg.shape
    numtimepoints = theshape[0]
    themeans = np.zeros(numtimepoints)
    timeindex = np.arange(numtimepoints)
    for i in timeindex:
        themeans[i] = np.mean(np.ravel(thesubreg[i, :, :, :]))
    return themeans

def getroival(theimage, theroi, zpos):
    # get the average value from an roi in a 3D image
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    theroival = np.mean(theimage[zpos, ystart:yend, xstart:xend])
    return theroival

def getroistd(theimage, theroi, zpos):
    # get the standard deviation of an roi in a 3D image
    xstart = theroi[0][0]
    xend = theroi[1][0]
    ystart = theroi[0][1]
    yend = theroi[1][1]
    theroistd = np.std(theimage[zpos, ystart:yend, xstart:xend])
    return theroistd

def evalweisskoff(numrois, roisizes, timepoints, xcenter, ycenter, zcenter, theimage, direction = ""): 
    # Weisskoff analysis evaluation
    '''
    This method evaluate Weiskoff RDCs for axial, coronal, sagittal directions and for 3D ROI
    
    numrois: number of ROI to try analysis (int)
    roisizes: dimensions of ROI (range)
    timepoints: temporal vector (array 1D)
    xcenter, ycenter, zcenter: centers of ROI (int, int, int)
    theimage: MRI data (array 4D)
    direction: select a (axial), c (coronal), s (sagittal) or cube (3D) (string)
    '''

    weissstddevs = np.zeros(numrois)
    weisscvs = np.zeros(numrois)
    roiareas = np.zeros(numrois)
    projstddevs = np.zeros(numrois)
    projcvs = np.zeros(numrois)

    if(direction in ["a", "c", "s"]):

        if direction == "c" or direction == "s":
            xcenter, zcenter = zcenter, xcenter

        for i in roisizes:
            roi = setroilims(xcenter, ycenter, i)
            timecourse = getroimeantc(theimage, roi, zcenter)
            weisskoffcoffs = np.polyfit(timepoints, timecourse, 2)
            fittc = trendgen(timepoints, weisskoffcoffs)
            detrendedweisskofftc = timecourse - fittc
            dtweissmean = np.mean(detrendedweisskofftc)
            weissstddevs[i - 1] = np.nan_to_num(np.std(detrendedweisskofftc))
            weisscvs[i - 1] = weissstddevs[i - 1] / dtweissmean
            roiareas[i - 1] = np.square(roisizes[i - 1]) #ROI area
            projstddevs[i - 1] = weissstddevs[0] / i
            projcvs[i - 1] = weisscvs[0] / i
        weissrdc = weisscvs[0] / weisscvs[numrois - 1]

    elif (direction == "cube"):
        for i in roisizes:
            roi = set3Droilims(xcenter, ycenter, zcenter, i)
            timecourse = get3Droimeantc(theimage, roi)
            weisskoffcoffs = np.polyfit(timepoints, timecourse, 2)
            fittc = trendgen(timepoints, weisskoffcoffs)
            detrendedweisskofftc = timecourse - fittc
            dtweissmean = np.mean(detrendedweisskofftc)
            weissstddevs[i - 1] = np.nan_to_num(np.std(detrendedweisskofftc))
            weisscvs[i - 1] = weissstddevs[i - 1] / dtweissmean
            roiareas[i - 1] = np.square(roisizes[i - 1])**3 #ROI volume
            projstddevs[i - 1] = weissstddevs[0] / i**2
            projcvs[i - 1] = weisscvs[0] / i
        weissrdc = np.sqrt(weisscvs[0] / weisscvs[numrois - 1])

    else:
        raise ValueError

    return roiareas, weissstddevs, projstddevs, weissrdc, projcvs, weisscvs, timecourse

def qualitytag(thestring, thequality):
    colors = ('00ff00', 'ffff00', 'ff0000')
    return '<FONT COLOR="{}">{}</FONT>'.format(colors[thequality], thestring)

def prepareinput(dicompath): 

    '''
    This method search for all the requested inputs file inside the acquisition folder

    input:
        dicompath: path of acquisition folder

    output:
        dicomfilename: dicom file path
        filenames: nifti first acquisition path
        shimming: nifti shimming acquisition path
        noshimming: nifti no-shimming acquisition path

    '''
    
    dicomfilename = ""
    niipath = join(dicompath, "nii")
    
    if not exists(niipath):
        raise Exception("nii folder not found!")

    for folds in listdir(dicompath):
        if "nii" in folds: continue
        dicomfilename = selectdcmfilename(dicompath, folds)
        # dcmpath = join(dicompath, folds) 
    return divideinput(niipath, dicomfilename)

def selectdcmfilename(dicompath, folds):
    dicomfilename = []
    if isfile(join(dicompath, folds)): return [join(dicompath, folds)]
    for i,pathfile in enumerate(listdir(join(dicompath, folds))):
        if folds.endswith(".ini"): continue
        if "loc" in pathfile or "Loc" in pathfile: continue
        if ("shim" not in pathfile and "SHIM" not in pathfile) or ("ap" in pathfile or "pa" in pathfile or "AP" in pathfile or "PA" in pathfile): 
            dicomtemp = join(join(dicompath, folds), listdir(join(join(dicompath, folds)))[i])
            if isdir(dicomtemp):
                dicomfilename.append(searchfile(dicomtemp))
            else: 
                dicomfilename.append(dicomtemp)
    return dicomfilename

def searchfile(path, ext = None, allfile = False): 
    
    if ext is not None:
        if listdir(path)[0].endswith(".IMA"): ext  = ".IMA"
        filelist = []
        for file in listdir(path):
            if file.endswith(ext):
                filelist.append(file)
        if len(filelist) == 0: raise Exception("file not found!")
        return filelist if allfile else filelist[0]     
    
    else:
        for file in listdir(path):
            if isdir(join(path, file)):
                return searchfile(join(join(path, file)))
            elif "DICOMDIR" not in file: 
                return join(path, file) 
    
 
def divideinput(path, dicomfilename):
    filenames = []
    shimming = ""
    noshimming = ""
    for nifti in listdir(path):
        if "loc" in nifti or "Loc" in nifti: continue
        if ("no_shimming" in nifti or "no_shiming" in nifti or "off" in nifti or "NO_" in nifti or "no_" in nifti) and ("ap" not in nifti and "pa" not in nifti and "_acquisition" not in nifti and "_con" not in nifti):
            noshimming = join(path, nifti)
        elif ("shim" in nifti or "shiming" in nifti or "shimming" in nifti or "SHIM" in nifti) and ("ap" not in nifti) and ("pa" not in nifti):
            shimming = join(path, nifti)
        else:
            filenames.append(join(path, nifti))
    return dicomfilename, filenames, shimming, noshimming

def directedfrom(filename):
    ap = ["AP", "ap", "_A."]
    pa = ["PA", "pa", "_P."]
    if any(word in filename for word in ap): return("_AP")
    elif any(word in filename for word in pa): return("_PA")
    else: return ""

def find_files(filename, search_path):
    result = ""
    for root, dir, files in walk(search_path):
        if filename in files:
            result = join(root, filename)
            break
    return result

def wkhsearch(wkh):
    #check if wkhtmltopdf is installed
    #if system is windows
    if wkh==None and platform.startswith('win'):
        home = str(Path.home())
        wkh=find_files("wkhtmltopdf.exe", home)
    elif wkh!=None and platform.startswith('win'):
        wkh=find_files("wkhtmltopdf.exe", wkh)

    #if system is linux
    elif wkh==None and platform.startswith('lin'):
        wkh=subprocess.getoutput("whereis wkhtmltopdf")
        wkh=wkh.split(" ")[1]
    elif wkh!=None and platform.startswith('lin'):
        wkh=find_files("wkhtmltopdf", wkh)
    
    if wkh=="":
        wkh=None
        logging.debug(
            "WARNING: no wkhtmltopdf installation found! This could cause problem to the PDF report creation!"
            )
    else: logging.debug("wkhtmltopdf.exe find at %s\n" %wkh)
    return(wkh)

def printLogo():
    system("clear")
    print("""
                                                
     _____     __    ___     ___        __    __  
    |   _  \   \ \  /  /    /   \      |  \  |  | 
    |  |_)  |   \ \/  /    / /_\ \     |   \ |  | 
    |      /     \   /    /  ___  \    |    \|  | 
    |  |\  \     /  /    /  /   \  \   |  |\    | 
    |__| \__\   /__/    /__/     \__\  |__| \ __|       
                                                                            

        """) 