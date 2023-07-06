#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:54:28 2022

@author: eric
"""

import glob
import json
import base64
from itertools import groupby
import numpy as np
from scipy import optimize as opt
from scipy.spatial import distance as dist
from pyexcel_ods import get_data
import matplotlib.pyplot as plt
from helper_functions import create_grid_from_vecs

def measure_distance(vecs, Xdat, Ydat):
    """
    measures the distance between theoretical grid spanned by vec1
    and vec2 (summarized in vecs here) and the measured grid (Xdata, Ydata)

    For all points in the measured grid, the Euclidian distance to the
    closest point of the theoretical grid is considered.
    The corresponding distance is divided by the uncertainty factor dist**2
    (distance of the measured point from the focal point/original bundle)
    """
    Xgrid, Ygrid = create_grid_from_vecs(vecs[:2], vecs[2:], 10, 10)
    dists = np.array([])
    for xd, yd in zip(Xdat, Ydat):
        if xd+yd != 0:
            D = dist.cdist([[xd, yd]], [[x, y] for x, y in zip(Xgrid, Ygrid)], metric='euclid')
            dists = np.append(dists, D.min() / (xd**2 + yd**2))
    return np.sum(dists)

def overall_distance(vecs, Xdic, Ydic):
    """
    summed distances across different grids in Xdic, Ydic
    """
    dists = np.array([])
    for X, Y in zip(Xdic.values(), Ydic.values()):
        dists = np.append(dists, measure_distance(vecs, X, Y))
    return np.sum(dists)

# loop across time points
vecs_opt1 = np.zeros((4, 2))
vecs_opt2 = np.zeros((4, 2))
for hh in range(4):
    print('Data set ' + str(hh+1))
    # loop across receptors, for a given time point
    Xdata, Ydata = {}, {}
    for rec in range(2, 7): # R1 did not come with L-cell data
        print('Receptor ' + str(rec))
        if rec in (1, 2, 5, 6):
            hours = [27, 31, 36, 41]
        elif rec == 3:
            hours = [26, 30, 35, 40]
        elif rec == 4:
            hours = [27, 31, 36, 39]
        hour = hours[hh]

        # load L-cell grid
        data = get_data("Lcell_locations_R"+str(rec)+".ods")
        Lcell_locations = {}
        Lcell_locations[str(hour)+'hAPF'] = data[str(hour)+'hAPF'][1:]
        b = [list(g) for k, g in groupby(Lcell_locations[str(hour)+'hAPF'],
                                         lambda x: str(x) != '[]') if k]
        Xorig, Yorig = np.array(b[0]).T
        if rec != 5:
            Yorig = 512 - Yorig

        # load heel locations
        if rec == 3:
            PATH_TAIL = '/home/eric/axon_guidance/experimental_data/'\
                'Sample_Growth_Cone_Pictures/20210402_data_set/R3/'\
                    'Segmented_R3/Semi_automatic_aligned_R3_pictures/'\
                    +str(hour)+'hAPF/'
            FULL_PATHS = np.sort(glob.glob(PATH_TAIL + '*.png'))
        else:
            PATH_TAIL = '/home/eric/axon_guidance/experimental_data/'\
                'Sample_Growth_Cone_Pictures/20210402_data_set/R'\
                    +str(rec)+'/'+str(hour)+'hAPF/'
            FULL_PATHS = np.sort(glob.glob(PATH_TAIL + '*Front*.tif'))
        data2 = get_data("filopodia_manual_front_R"+str(rec)+".ods")

        HEELLOCS = np.array([])
        all_tips_front_manual = {}
        all_tips_front_manual[str(hour)+'hAPF'] = data2[str(hour)+'hAPF'][1:]
        a = [list(g) for k, g in groupby(all_tips_front_manual[str(hour)+'hAPF'],
                                         lambda x: str(x) != '[]') if k]
        for i, impath in enumerate(FULL_PATHS):
            # load manually detected filopodial tips
            HEELLOC = np.array([a[i][0][0], a[i][0][1]])
            HEELLOCS = np.append(HEELLOCS, HEELLOC)
        HEELLOCS = np.reshape(HEELLOCS, (-1, 2))
        HEEL = np.mean(HEELLOCS, axis=0) # average of the 12/30 images

        # pick original bundle - smallest distance to heel of growth cone
        # - and shift grid to have original bundle at (0, 0)
        D_heel = dist.cdist([HEEL], [[x, y] for x, y in zip(Xorig, Yorig)], metric='euclid')
        Xdata['R'+str(rec)] = Xorig - Xorig[D_heel.argmin()]
        Ydata['R'+str(rec)] = Yorig - Yorig[D_heel.argmin()]

    res = opt.minimize(overall_distance, [100, 100, 200, 0], args=(Xdata, Ydata))
    vecs_opt1[hh, :] = res.x[:2]
    vecs_opt2[hh, :] = res.x[2:]

    if hh == 0:
        fig = plt.figure()
    ax = fig.add_subplot(2, 2, hh+1)
    ax.scatter(*create_grid_from_vecs(vecs_opt1[hh, :],
                                      vecs_opt2[hh, :], 4, 4), marker='+', s = 100, c='k')
    for Xd, Yd in zip(Xdata.values(), Ydata.values()):
        ax.scatter(Xd, Yd)
    if hh == 1:
        ax.legend(['opt', 'R2', 'R3', 'R4', 'R5', 'R6'])

opt_vec = np.hstack((vecs_opt1, vecs_opt2))
with open("optimal_grid.json", "w", encoding='utf-8') as outfile:
    json.dump(json.dumps([str(opt_vec.dtype),base64.b64encode(opt_vec).\
                          decode('utf-8'), opt_vec.shape]), outfile)
