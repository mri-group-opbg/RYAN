import matplotlib
import seaborn
seaborn.set_style("dark")
# matplotlib.use('Agg', warn=False)
import numpy as np
from numpy.linalg import inv
from mako.lookup import TemplateLookup
makolookup = TemplateLookup(directories=['./tpl'])
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def CBIrobustfit(X,y):
    # Initialize variables:
    #y = np.matrix([[10], [2], [32], [42]])
    #X = np.ones((4,1))
    myTolerance = 10**-4
    max_iter = 50

    w = np.ones(y.shape[0])           # weights
    b = np.zeros(X.shape[1])          # estimator
    myErr = np.zeros((max_iter,1))    # error
    
    for i in range(0,max_iter-1):
        W = np.zeros((y.shape[0], y.shape[0]))
        np.fill_diagonal(W, w)
        temp1 = np.matmul(X.transpose(), W)
        temp2 = np.matmul(temp1,X)
        temp3 = np.matmul(temp1,y)
        b = np.matmul(inv(temp2), temp3)
        e = y - X * b
        temp4 = np.matmul(e.transpose(), W)
        #print np.matmul(temp4, e)
        myErr[i,0] = np.matmul(temp4, e) 
        sigma = np.median(abs(e), axis=0) / 0.6745
        k = 4.685 * sigma
        w = np.zeros(e.shape)
        ae = abs(e)
        index_inside = (ae<=k).nonzero()
        w[index_inside,0] = np.power((1-np.power((e[index_inside,0]/k),2)),2)
    
        # Check for convergence:
        if (i > 2) and (abs(myErr[i-1,0] - myErr[i,0]) < myTolerance * myErr[i,0]):
            break
    
    #print w,b
    return b

#def spike_detection(spikes, corner1, corner2, meants)
    
    #return ts, nspk, slices, name