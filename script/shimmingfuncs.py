
import pkg_resources
from distutils.version import LooseVersion

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
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
from collections import namedtuple
import configparser


def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



