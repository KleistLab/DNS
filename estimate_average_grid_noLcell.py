#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:54:28 2022

@author: eric
"""

import json
import base64
import numpy as np
from scipy import optimize as opt
from scipy.spatial import ConvexHull
from scipy.spatial import distance as dist
import pandas as pd
# import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

def create_grid_from_vecs(vec1, vec2, xsteps, ysteps):
    """
    creates a grid from the two-dimensional vectors vec1 and vec2
    using integer coefficients from -x/ysteps to x/ysteps
    """
    n1, n2 = np.meshgrid(np.arange(-xsteps, xsteps+1),
                         np.arange(-ysteps, ysteps+1))
    Xgrid = (n1*vec1[0] + n2*vec2[0]).flatten()
    Ygrid = (n1*vec1[1] + n2*vec2[1]).flatten()
    return Xgrid, Ygrid

def measure_distance(vecs, Xdata, Ydata):
    """
    measures the distance between theoretical grid spanned by vec1 and vec2
    (summarized in vecs here) and the measured grid (Xdata, Ydata)

    For all points in the measured grid, the Euclidian distance to
    the closest point of the theoretical grid is considered.
    The corresponding distance is divided by the uncertainty factor dist**2
    (distance of the measured point from the focal point/original bundle)

    Here, for the no L-cell data, it was necessary to add to the objective function
    a punishment for the having too many grid points in the fitted grid ('error')

    """
    Xgrid, Ygrid = create_grid_from_vecs(vecs[:2], vecs[2:], 10, 10)
    dists = np.array([])
    dat = np.array([[x, y] for x, y in zip(Xdata, Ydata)])
    hull = ConvexHull(dat)

    path = mpltPath.Path(dat[hull.vertices])
    error = np.sum(path.contains_points(np.array([Xgrid, Ygrid]).T))\
        - np.sum(path.contains_points(np.array([Xdata, Ydata]).T))
    error = np.where(error < 0, 0, error)
    for xd, yd in dat:
        if xd+yd != 0:
            D = dist.cdist([[xd, yd]], [[x, y] for x, y in zip(Xgrid, Ygrid)], metric='euclid')
            dists = np.append(dists, D.min() / (xd**2 + yd**2))
    return np.sum(dists) + 50*error

# loop across time points
PATH = "/home/eric/axon_guidance/experimental_data/XYpositionsSmo-R4_Heels_Egemen.xlsx"
data = np.array(pd.read_excel(PATH)[1:], dtype='float')
time, xpos, ypos = data[:, 0], data[:, 1::2], data[:, 2::2]

# contains data from two brains, align the two grids for each time point
# first find the central point for each time point
xc1, yc1 = np.mean(xpos[:, :25], axis=1), np.mean(ypos[:, :25], axis=1)
xc2, yc2 = np.mean(xpos[:, 25:], axis=1), np.mean(ypos[:, 25:], axis=1)
for tt in range(len(time)):
    D1 = dist.cdist([[xc1[tt], yc1[tt]]], [[x, y] for x, y in zip(xpos[tt, :25], ypos[tt, :25])], metric='euclid')
    xpos[tt, :25] -= (xpos[tt, :25])[D1.argmin()]
    ypos[tt, :25] -= (ypos[tt, :25])[D1.argmin()]
    D2 = dist.cdist([[xc2[tt], yc2[tt]]], [[x, y] for x, y in zip(xpos[tt, 25:], ypos[tt, 25:])], metric='euclid')
    xpos[tt, 25:] -= (xpos[tt, 25:])[D2.argmin()]
    ypos[tt, 25:] -= (ypos[tt, 25:])[D2.argmin()]

vecs_opt1_dat1 = np.zeros((len(time), 2))
vecs_opt2_dat1 = np.zeros((len(time), 2))
vecs_opt1_dat2 = np.zeros((len(time), 2))
vecs_opt2_dat2 = np.zeros((len(time), 2))
for hh in range(len(time)):
    print('Data set ' + str(hh+1))
    # loop across receptors, for a given time point
    L = np.sqrt(2) * 100
    res1 = opt.minimize(measure_distance, [L/np.sqrt(2), L/np.sqrt(2), L, 0], args=(xpos[hh, :25], ypos[hh, :25])) # initial condition
    vecs_opt1_dat1[hh, :] = res1.x[:2]
    vecs_opt2_dat1[hh, :] = res1.x[2:]

    res2 = opt.minimize(measure_distance, [L/np.sqrt(2), L/np.sqrt(2), L, 0], args=(xpos[hh, 25:], ypos[hh, 25:])) # initial condition
    vecs_opt1_dat2[hh, :] = res2.x[:2]
    vecs_opt2_dat2[hh, :] = res2.x[2:]

    '''
    plt.figure()
    plt.scatter(*create_grid_from_vecs(vecs_opt1_dat1[hh, :], vecs_opt2_dat1[hh, :], 10, 10))
    plt.scatter(xpos[hh, :25], ypos[hh, :25])
    plt.axis([1.1*np.min(xpos[hh, :25]), 1.1*np.max(xpos[hh, :25]), 1.1*np.min(ypos[hh, :25]), 1.1*np.max(ypos[hh, :25])])
    plt.title(int(np.sqrt(np.dot(vecs_opt1_dat1[hh], vecs_opt1_dat1[hh]))))

    plt.figure()
    plt.scatter(*create_grid_from_vecs(vecs_opt1_dat2[hh, :], vecs_opt2_dat2[hh, :], 10, 10))
    plt.scatter(xpos[hh, 25:], ypos[hh, 25:])
    plt.axis([1.1*np.min(xpos[hh, 25:]), 1.1*np.max(xpos[hh, 25:]), 1.1*np.min(ypos[hh, 25:]), 1.1*np.max(ypos[hh, 25:])])
    plt.title(int(np.sqrt(np.dot(vecs_opt1_dat2[hh], vecs_opt1_dat2[hh]))))
    '''

final_v1 = (vecs_opt1_dat1 + vecs_opt1_dat2)/2
final_v2 = (vecs_opt2_dat1 + vecs_opt2_dat2)/2

'''
plt.figure()
plt.scatter(*final_v1.T/24)
plt.scatter(*final_v2.T/24)
plt.scatter(*np.mean(final_v1/24, 0), c='k')
plt.scatter(*np.mean(final_v2/24, 0), c='k')
plt.scatter(*np.zeros(2))

plt.figure()
plt.scatter(xpos, ypos)
'''
opt_vec = np.hstack((final_v1, final_v2))
# it seems that the changes of v1 and v2 over time are not systematic,
# so we output the average vectors for all time points
opt_vec_avg = np.ones(np.shape(opt_vec)) * np.mean(opt_vec, axis=0)
with open("optimal_grid_noLcell.json", "w", encoding='utf-8') as outfile:
    json.dump(json.dumps([str(opt_vec_avg.dtype),base64.b64encode(opt_vec_avg).decode('utf-8'), opt_vec_avg.shape]), outfile)
