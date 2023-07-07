#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:53:58 2023

@author: eric
"""

import json
import base64
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def remove_ticks_and_box(ax1):
    """
    helper function for plotting
    """
    ax1.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    ax1.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        ax1.spines[pos].set_visible(False)

def load_grid_heels_fronts():
    """
    helper function to load previously measured parameters
    """
    # load vectors of the fitted optimal grid
    with open("./data-files/optimal_grid.json", "r", encoding='utf-8') as infile:
        loaded = json.load(infile).split('\"')
    dtype = np.dtype(loaded[1])
    arr = np.frombuffer(base64.decodebytes(bytearray(loaded[3], 'utf-8')), dtype)
    vec_opt = arr.reshape([int(val) for val in loaded[4] if val.isnumeric()]).copy()

    # load front locations
    with open("./data-files/frontcenters_automatic_all.json", "r", encoding='utf-8') as infile:
        loaded = json.load(infile).split('\"')
    dtype = np.dtype(loaded[1])
    arr = np.frombuffer(base64.decodebytes(bytearray(loaded[3], 'utf-8')), dtype)
    FRONTLOCS_ALL = arr.reshape([int(val) for val in loaded[4] if val.isnumeric()]).copy()

    # load heel locations
    with open("./data-files/HEELLOCS_AVG.json", "r", encoding='utf-8') as infile:
        loaded = json.load(infile).split('\"')
    dtype = np.dtype(loaded[1])
    arr = np.frombuffer(base64.decodebytes(bytearray(loaded[3], 'utf-8')), dtype)
    HEELLOCS_AVG = arr.reshape([int(val) for val in loaded[4] if val.isnumeric()]).copy()
    return vec_opt, FRONTLOCS_ALL, HEELLOCS_AVG

def potential_parabola(loc, mu_loc, width, amp):
    """
    square parabola with support defined by width,
    different extent for x-axis and y-axis
    """
    return amp*np.maximum(-np.sum((loc[:, :, None]-mu_loc[:, None, :])**2
                              /width[:, None, None]**2, axis=0) + 1, 0)

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

def create_heatmap(data, row_labels, col_labels, xlabel, ylabel, ax1=None,
            cbarlabel="", **kwargs):
    """
    Create a heatmap for data, see matplotlib 'heatmap' documentation
    for many of the elements
    """

    if ax1 is None:
        ax1 = plt.gca()
    im1 = ax1.imshow(data, **kwargs)

    # create colorbar
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax1.figure.colorbar(im1, cax=cax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=10)

    # ticks and labels
    ax1.set_xticks(np.arange(data.shape[1]))
    ax1.set_yticks(np.arange(data.shape[0]))
    ax1.set_xticklabels(col_labels)
    ax1.set_yticklabels(row_labels)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax1.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center", va='center',
             rotation_mode="anchor")

    for _, spine in ax1.spines.items():
        spine.set_visible(False)

    ax1.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax1.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax1.tick_params(which="minor", bottom=False, left=False)
    return im1, cbar

def rotate_coord(x, y, angle):
    """
    rotate coordinates by angle, angle in rad
    """
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)

def create_starting_grid2(centerlocx, centerlocy, hour, val_inter=None, fading=0):
    """
    create grid of bundles mimicking the experimentally observed bundle shape
    ('horseshoe'), without equator
    """
    alpha = -0.23
    if isinstance(val_inter, int):
        ell_a = 32 + 2*val_inter/fading
        ell_b = 55 + 5*val_inter/fading
    else:
        if hour < 35:
            ell_a = 32
            ell_b = 55
        else:
            ell_a = 34
            ell_b = 60

    yrec = np.array([ell_a*np.cos(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi)
                     for n in np.arange(1, 7)])
    xrec = np.array([ell_b*np.sin(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi)
                     for n in np.arange(1, 7)])
    xrec_rot, yrec_rot = rotate_coord(xrec, yrec, -7*np.pi/180)

    recep_y = yrec_rot[:, None] + centerlocy[None, :]
    recep_x = xrec_rot[:, None] + centerlocx[None, :]
    return recep_x, recep_y, centerlocx, centerlocy

def create_starting_grid_noLcell(center_loc_x, center_loc_y):
    """
    create grid of bundles in the noLcell situation, elliptical bundles
    at 22 hAPF, without equator
    """
    alpha = -np.pi/3
    ell_a = 2.58 * 24 / 2
    ell_b = 2.96 * 24 / 2

    yrec = np.array([ell_a*np.cos(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])
    xrec = np.array([ell_b*np.sin(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])

    recep_y = yrec[:, None] + center_loc_y[None, :]
    recep_x = xrec[:, None] + center_loc_x[None, :]
    return recep_x, recep_y, center_loc_x, center_loc_y

def calc_closest_point_on_ellipse(a_ell, b_ell, point):
    """
    for a given point, calculate the closest point on the periphery
    of an ellipse with the two axes a_ell and b_ell
    - assume that the center of the ellipse is at (0, 0)
    """
    xr = np.sqrt(a_ell**2 * b_ell**2 / (b_ell**2 + a_ell**2*(point[:, :, 1]/point[:, :, 0])**2))
    yr = point[:, :, 1]/point[:, :, 0] * xr
    return np.sign(point[:, :, 0])*xr, np.sign(point[:, :, 1])*np.abs(yr)

def tanh_cust(x, x_half, sl):
    '''
    customized hyperbolic tangent
    '''
    return 1/2 * (1 + np.tanh(2*sl*(x - x_half)))

def index_func(x):
    """
    little helper function for create_grid_for_all_timesteps.py
    """
    return np.where(x == 0, 0, x-1)

def create_movie_from_png(moviename, remove_png_afterwards=True):
    """
    call mencoder to create a movie
    from the png files in the current folder
    """
    command = ('mencoder',
               'mf://*.png',
               '-mf',
               'type=png:w=800:h=600:fps=3',
               '-ovc',
               'lavc',
               '-lavcopts',
               'vcodec=mpeg4:vbitrate=4000',
               '-oac',
               'copy',
               '-o',
               moviename+'.avi')
    os.spawnvp(os.P_WAIT, 'mencoder', command)
    if remove_png_afterwards:
        for file in glob.glob("*.png"):
            os.remove(file)
