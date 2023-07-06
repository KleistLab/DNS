#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 18:17:02 2022

@author: eric
"""

import numpy as np
import os
import json
import base64
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from scipy.signal import fftconvolve
from scipy.spatial import distance as dist
from random_walk_potential_ito import potential_parabola
from plot_all_filopodial_tips_in_heel_density import load_grid_heels_fronts, remove_ticks_and_box, heatmap
import numexpr as ne
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from joblib import Parallel, delayed
import shapely.affinity
from shapely.geometry import Point
import descartes

def rotate_coord(x, y, angle):
    """
    angle in rad
    """
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)

scale = 1
def create_starting_grid2(center_loc_x, center_loc_y, hour, val_inter=None, fading=0):
    """
    for real L cell grid
    create grid at 22 hAPF, without equator
    """
    alpha = -0.23
    if type(val_inter) == int:
        ell_a = 32 + 2*val_inter/fading
        ell_b = 55 + 5*val_inter/fading
    else:
        if hour < 35:
            ell_a = 32#35#*np.mean(np.diff(Y)[np.diff(Y) > 0])
            ell_b = 55#60#*np.mean(np.diff(X)[np.diff(X) > 0])
        else:
            ell_a = 34#38#*np.mean(np.diff(Y)[np.diff(Y) > 0])
            ell_b = 60#65#*np.mean(np.diff(X)[np.diff(X) > 0])
            
    ell_a *= scale
    ell_b *= scale

    yrec = np.array([ell_a*np.cos(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])
    xrec = np.array([ell_b*np.sin(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])
    xrec_rot, yrec_rot = rotate_coord(xrec, yrec, -7*np.pi/180) # - 20*np.pi/180)
    
    recep_y = yrec_rot[:, None] + center_loc_y[None, :]
    recep_x = xrec_rot[:, None] + center_loc_x[None, :]
    
    ############################################################
    # perturbation testing later: rotate the entire ellipse, not just changing angles
    return recep_x, recep_y, center_loc_x, center_loc_y

def create_starting_grid_noLcell(center_loc_x, center_loc_y):
    """
    for real L cell grid
    create grid at 22 hAPF, without equator
    """
    alpha = -np.pi/3
    ell_a = 2.58 * 24 / 2 # from Egemen's measurements, Email sent June 13 2022
    ell_b = 2.96 * 24 / 2
    
    yrec = np.array([ell_a*np.cos(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])
    xrec = np.array([ell_b*np.sin(-(alpha + (n - 1)*(np.pi - 2*alpha)/5) + np.pi) for n in np.arange(1, 7)])
    
    recep_y = yrec[:, None] + center_loc_y[None, :]
    recep_x = xrec[:, None] + center_loc_x[None, :]
    
    ############################################################
    # perturbation testing later: rotate the entire ellipse, not just changing angles
    return recep_x, recep_y, center_loc_x, center_loc_y

def calc_closest_point_on_ellipse(a, b, point):
    # assume that the center of the ellipse is at (0, 0)
    # a and b are the full length of the two axes, i.e. the diameter of the circle if a=b
    xr = np.sqrt(a**2 * b**2 / (b**2 + a**2*(point[:, :, 1]/point[:, :, 0])**2))
    yr = point[:, :, 1]/point[:, :, 0] * xr
    return np.sign(point[:, :, 0])*xr, np.sign(point[:, :, 1])*np.abs(yr)

def tanh_cust(x, x_half, sl):
    return 1/2 * (1 + np.tanh(2*sl*(x - x_half)))

#@numba.njit()
def generate_indmat(xind, yind, fr_x, fr_y, r, frontang, REC_X_INTER, REC_Y_INTER, circwidth):
    rec_x, rec_y = np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)
    rec_x, rec_y = rec_x[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r**2], rec_y[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r**2]
    ind = np.zeros(np.shape(xind), dtype='bool')
    # ind1 = ((xind - fr_x)**2 + (yind - fr_y)**2 <= r**2)
    ind1 = ne.evaluate('((xind-fr_x)**2 + (yind-fr_y)**2)<=r**2')
    ind2 = (np.cos(np.arctan2(yind[ind1]-fr_y, xind[ind1]-fr_x) - frontang) > np.cos(np.pi/180*circwidth)) #0.643) #0.25882) #0.77)
    ind[yind[ind1][ind2], xind[ind1][ind2]] = True
    radius = 0.5 * CONVFAC
    ind3 = np.zeros(np.shape(xind), dtype='bool')
    for rx, ry in zip(rec_x, rec_y):
        ind3 += ne.evaluate('((xind-rx)**2 + (yind-ry)**2)<=radius**2')
    return np.logical_and(ind, ~ind3)

def sample_roi(dat2_inter, FRONTLOC, meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER):
    # xfil, yfil = np.zeros((n_fil, nr_of_rec)), np.zeros((n_fil, nr_of_rec))
    xfil, yfil = [], []
    allinds = np.zeros(1023*2047, dtype='bool') #np.zeros(np.shape(dat2_inter), dtype = 'bool')
    if single_receptor:
        frontang = np.arctan2(meanvec_old[single_receptor-1].imag, meanvec_old[single_receptor-1].real)
        # if hh == 1 and tind == 0:
        #     # frontang = startang[0]
        #     frontang = np.arctan2(meanvec_old[irec].imag, meanvec_old[irec].real)
        # else:
        #     frontang = np.arctan2(FRONTLOC[irec, 1] - exp_fronts_y[irec, 0], FRONTLOC[irec, 0] - exp_fronts_x[irec, 0])
        ind = generate_indmat(xind, yind, FRONTLOC[0], FRONTLOC[1], radii[single_receptor-1], frontang, REC_X_INTER, REC_Y_INTER, circ_width[single_receptor-1])
            
        # data = np.ravel(dat2_inter)[np.where(np.ravel(ind))[0]]
        # sortdat = np.sort(data)
        # mindat, maxdat = sortdat[0], sortdat[-1]
        # nbins = 10000
        # bins = np.linspace(mindat, maxdat, nbins+1)
        # histog = np.zeros(nbins)
        # for ii in range(nbins):
        #     histog[ii] = np.sum((data >= bins[ii])*(data <= bins[ii+1]))
        histog, bins = np.histogram(dat2_inter[ind], bins=10000)
        cs = (1 - np.cumsum(histog)/np.sum(histog))**4
        rr = np.random.rand(n_fil)
        ind_res = np.array([np.where(cs < rtemp)[0][0] for rtemp in rr])
        vals = ((bins[1:] + bins[:-1])/2)[ind_res]
        
        # pos1 = np.ravel(POS[0])[np.where(np.ravel(ind))[0]]
        # pos2 = np.ravel(POS[1])[np.where(np.ravel(ind))[0]]
        # for iv, v in enumerate(vals):
        #     minind = np.abs(v - data).argmin()
        #     xfil[iv][irec] = pos1[minind]
        #     yfil[iv][irec] = pos2[minind]
        # indflat = np.ravel(ind)
        # for i in range(len(allinds)):
        #     allinds[i] = allinds[i] + indflat[i]

        D_vals = np.abs(vals[:, None] - dat2_inter[ind][None, :])
        xfil = np.hstack((xfil, POS[0][ind][D_vals.argmin(axis=1)]))
        yfil = np.hstack((yfil, POS[1][ind][D_vals.argmin(axis=1)]))
        allinds = allinds + np.ravel(ind)
        
    else:
        for irec in range(len(FRONTLOC)):
            if region == 'circle':
                ind = (np.sqrt((xind - FRONTLOC[irec, 0])**2 + (yind - FRONTLOC[irec, 1])**2) <= r)
            elif region == 'circ_segment':
                frontang = np.arctan2(meanvec_old[irec].imag, meanvec_old[irec].real)
                # if hh == 1 and tind == 0:
                #     # frontang = startang[0]
                #     frontang = np.arctan2(meanvec_old[irec].imag, meanvec_old[irec].real)
                # else:
                #     frontang = np.arctan2(FRONTLOC[irec, 1] - exp_fronts_y[irec, 0], FRONTLOC[irec, 0] - exp_fronts_x[irec, 0])
                ind = generate_indmat(xind, yind, FRONTLOC[irec, 0], FRONTLOC[irec, 1], np.repeat(radii, nr**2)[irec], frontang, REC_X_INTER, REC_Y_INTER, np.repeat(circ_width, nr**2)[irec])
                
            # data = np.ravel(dat2_inter)[np.where(np.ravel(ind))[0]]
            # sortdat = np.sort(data)
            # mindat, maxdat = sortdat[0], sortdat[-1]
            # nbins = 10000
            # bins = np.linspace(mindat, maxdat, nbins+1)
            # histog = np.zeros(nbins)
            # for ii in range(nbins):
            #     histog[ii] = np.sum((data >= bins[ii])*(data <= bins[ii+1]))
            histog, bins = np.histogram(dat2_inter[ind], bins=10000)
            cs = (1 - np.cumsum(histog)/np.sum(histog))**4
            rr = np.random.rand(n_fil)
            ind_res = np.array([np.where(cs < rtemp)[0][0] for rtemp in rr])
            vals = ((bins[1:] + bins[:-1])/2)[ind_res]
            
            # pos1 = np.ravel(POS[0])[np.where(np.ravel(ind))[0]]
            # pos2 = np.ravel(POS[1])[np.where(np.ravel(ind))[0]]
            # for iv, v in enumerate(vals):
            #     minind = np.abs(v - data).argmin()
            #     xfil[iv][irec] = pos1[minind]
            #     yfil[iv][irec] = pos2[minind]
            # indflat = np.ravel(ind)
            # for i in range(len(allinds)):
            #     allinds[i] = allinds[i] + indflat[i]
    
            D_vals = np.abs(vals[:, None] - dat2_inter[ind][None, :])
            xfil = np.hstack((xfil, POS[0][ind][D_vals.argmin(axis=1)]))
            yfil = np.hstack((yfil, POS[1][ind][D_vals.argmin(axis=1)]))
            allinds = allinds + np.ravel(ind)
    return xfil, yfil, allinds
    
def distance_to_exp_local(firstpos, pos_eval, a, b, Xint, Yint, v1int, v2int, receptor):
    goal_loc = np.zeros((1, 2))
    D_bundles = dist.cdist([firstpos], [[x, y] for x, y in zip(Xint, Yint)], metric='euclid')
    goal_loc[0, 0] = Xint[D_bundles.argmin(axis=1)]
    goal_loc[0, 1] = Yint[D_bundles.argmin(axis=1)]
    
    if receptor == 1:
        goal_loc += -v2int[None, :]
    elif receptor == 2:
        goal_loc += v1int[None, :] - v2int[None, :]
    elif receptor == 3:
        goal_loc += 2*v1int[None, :] - 1*v2int[None, :]
    elif receptor == 4:
        goal_loc += v1int[None, :]
    elif receptor == 5:
        goal_loc += v2int[None, :]
    elif receptor == 6:
        goal_loc += -v1int[None, :] + v2int[None, :]
    
    correct1 = (pos_eval[:, 0] - goal_loc[:, 0] + 0.333*v1int[0])**2 + (pos_eval[:, 1] - goal_loc[:, 1] + 0.333*v1int[1])**2 < 1.2*b**2
    correct2 = np.sqrt((pos_eval[:, 0] - goal_loc[:, 0])**2/a**2 + (pos_eval[:, 1] - goal_loc[:, 1])**2/b**2) <= 1 #0.8

    print(correct1)
    print(correct2)
    correct = np.logical_or(correct1, correct2)
    print(correct)
    return correct    

def run_gc(c_st, sl_st, startangs, A_magnet, c_magnet = 35, sl_magnet = 0.4):
    # print('\n ' + str(c_st) + '_' + str(sl_st))
    xind, yind = np.meshgrid(np.arange(2*Nx-1), np.arange(2*Ny-1))
    TIME = np.array([])
    if all_receptors:
        meanvec_old = np.repeat(speeds, nr**2) * 2/3*np.repeat(radii, nr**2)*np.sin(np.pi/180 * np.repeat(circ_width, nr**2)/(np.pi/180 * np.repeat(circ_width, nr**2)) * np.exp(1j*np.repeat(startangs, nr**2))) #* radii[receptor-1] * np.exp(1j*startang) * sp1
    else:
        meanvec_old = speeds * 2/3*radii*np.sin(np.pi/180 * circ_width)/(np.pi/180 * circ_width) * np.exp(1j*startangs) #* radii[receptor-1] * np.exp(1j*startang) * sp1
    #meanvec_old = radii * np.exp(1j*startangs) * speeds
    distances = np.zeros((4, 6))
    outside_fil_all = np.ones((59, 6, 10), dtype='bool')
    ii = 0
    for hh, hour in enumerate(hours[1:]):
        hh += 1
        # rm_heel_int = np.linspace(rm_heels[hh-1], rm_heels[hh], nsteps)
        # rm_front_int = np.linspace(rm_fronts[0, hh-1], rm_fronts[0, hh], nsteps)
        if hour > 26:
            v1old, v2old = vec_opt[hh-2, :2], vec_opt[hh-2, 2:]
            v1new, v2new = vec_opt[hh-1, :2], vec_opt[hh-1, 2:]
        else:
            v1old, v2old = vec_opt[0, :2] - (vec_opt[1, :2] - vec_opt[0, :2]), vec_opt[0, 2:] - (vec_opt[1, 2:] - vec_opt[0, 2:])
            v1new, v2new = vec_opt[0, :2], vec_opt[0, 2:]
        
        lcellsize_int = np.linspace(LCELL_SIZE[hh-1], LCELL_SIZE[hh], nsteps)
        v1inter, v2inter = np.linspace(v1old, v1new, nsteps), np.linspace(v2old, v2new, nsteps)
        
        mini, maxi = -10, 10
        n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
        center_loc_x = (n1*v1old[0] + n2*v2old[0]).flatten()
        center_loc_y = (n1*v1old[1] + n2*v2old[1]).flatten()
        ind = (center_loc_x > -500)*(center_loc_x < 1724)*(center_loc_y > -600)*(center_loc_y < 812)
        Xold, Yold = center_loc_x[ind], center_loc_y[ind]

        center_loc_x = (n1*v1new[0] + n2*v2new[0]).flatten()
        center_loc_y = (n1*v1new[1] + n2*v2new[1]).flatten()
        ind = (center_loc_x > -500)*(center_loc_x < 1724)*(center_loc_y > -600)*(center_loc_y < 812)
        Xnew, Ynew = center_loc_x[ind], center_loc_y[ind]
        
        # calculate Xshift for the 'old' time point
        # if hour > 26:
        #     REC_X2_OLD, REC_Y2_OLD, XC2, YC2 = create_starting_grid2(Xold, Yold, hour)
        #     D_center = dist.cdist([HEELLOCS_AVG[0, hh-2, :]], [[x, y] for x, y in zip(REC_X2_OLD[0, :], REC_Y2_OLD[0, :])], metric='euclid')    
        #     Xshift_old = - REC_X2_OLD[0, D_center.argmin()] + HEELLOCS_AVG[0, hh-2, 0]
        #     Yshift_old = - REC_Y2_OLD[0, D_center.argmin()] + HEELLOCS_AVG[0, hh-2, 1]
        #     # calculate Xshift for new time point
        #     REC_X2, REC_Y2, XC2, YC2 = create_starting_grid2(Xnew, Ynew, hour)
        #     D_center = dist.cdist([HEELLOCS_AVG[0, hh-1, :]], [[x, y] for x, y in zip(REC_X2[0, :], REC_Y2[0, :])], metric='euclid')    
        #     Xshift = - REC_X2[0, D_center.argmin()] + HEELLOCS_AVG[0, hh-1, 0]
        #     Yshift = - REC_Y2[0, D_center.argmin()] + HEELLOCS_AVG[0, hh-1, 1]

        #     mini, maxi = -10, 10
        #     n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
        #     D_shift_old = dist.cdist([HEELLOCS_AVG[0, hh-2, :]], [[x, y] for x, y in zip(np.ravel(Xshift_old + n1*v1old[0] + n2*v2old[0]), np.ravel(Yshift_old + n1*v1old[1] + n2*v2old[1]))], metric='euclid')
        #     Xshift_old = np.ravel(Xshift_old + n1*v1old[0] + n2*v2old[0])[D_shift_old.argmin()]
        #     Yshift_old = np.ravel(Yshift_old + n1*v1old[1] + n2*v2old[1])[D_shift_old.argmin()]

        #     D_shift = dist.cdist([HEELLOCS_AVG[0, hh-1, :]], [[x, y] for x, y in zip(np.ravel(Xshift + n1*v1new[0] + n2*v2new[0]), np.ravel(Yshift + n1*v1new[1] + n2*v2new[1]))], metric='euclid')
        #     Xshift = np.ravel(Xshift + n1*v1new[0] + n2*v2new[0])[D_shift.argmin()]
        #     Yshift = np.ravel(Yshift + n1*v1new[1] + n2*v2new[1])[D_shift.argmin()]
        # else:
        #     REC_X2_OLD, REC_Y2_OLD, XC2, YC2 = create_starting_grid2(Xold, Yold, hour)
        #     D_center = dist.cdist([HEELLOCS_AVG[0, 0, :]], [[x, y] for x, y in zip(REC_X2_OLD[0, :], REC_Y2_OLD[0, :])], metric='euclid')    
        #     Xshift_old = - REC_X2_OLD[0, D_center.argmin()] + HEELLOCS_AVG[0, 0, 0]
        #     Yshift_old = - REC_Y2_OLD[0, D_center.argmin()] + HEELLOCS_AVG[0, 0, 1]
        #     # calculate Xshift for new time point
        #     REC_X2, REC_Y2, XC2, YC2 = create_starting_grid2(Xnew, Ynew, hour)
        #     D_center = dist.cdist([HEELLOCS_AVG[0, 0, :]], [[x, y] for x, y in zip(REC_X2[0, :], REC_Y2[0, :])], metric='euclid')
        #     Xshift = - REC_X2[0, D_center.argmin()] + HEELLOCS_AVG[0, 0, 0]
        #     Yshift = - REC_Y2[0, D_center.argmin()] + HEELLOCS_AVG[0, 0, 1]

        #     mini, maxi = -10, 10
        #     n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
        #     D_shift_old = dist.cdist([HEELLOCS_AVG[0, 0, :]], [[x, y] for x, y in zip(np.ravel(Xshift_old + n1*v1old[0] + n2*v2old[0]), np.ravel(Yshift_old + n1*v1old[1] + n2*v2old[1]))], metric='euclid')
        #     Xshift_old = np.ravel(Xshift_old + n1*v1old[0] + n2*v2old[0])[D_shift_old.argmin()]
        #     Yshift_old = np.ravel(Yshift_old + n1*v1old[1] + n2*v2old[1])[D_shift_old.argmin()]
        #     D_shift = dist.cdist([HEELLOCS_AVG[0, 0, :]], [[x, y] for x, y in zip(np.ravel(Xshift + n1*v1new[0] + n2*v2new[0]), np.ravel(Yshift + n1*v1new[1] + n2*v2new[1]))], metric='euclid')
        #     Xshift = np.ravel(Xshift + n1*v1new[0] + n2*v2new[0])[D_shift.argmin()]
        #     Yshift = np.ravel(Yshift + n1*v1new[1] + n2*v2new[1])[D_shift.argmin()]
        
        # preparing the interpolated array
        Xshift, Xshift_old = 1024, 1024
        Yshift, Yshift_old = 512, 512
        Xstart = np.linspace(Xshift_old, Xshift, nsteps)
        Ystart = np.linspace(Yshift_old, Yshift, nsteps)
                
        if hour > 26:
            FRO_INT_X = np.linspace(FRONTLOCS_ALL[:, hh-2, 0] - HEELLOCS_AVG[:, hh-2, 0], FRONTLOCS_ALL[:, hh-1, 0] - HEELLOCS_AVG[:, hh-1, 0], nsteps)
            FRO_INT_Y = np.linspace(FRONTLOCS_ALL[:, hh-2, 1] - HEELLOCS_AVG[:, hh-2, 1], FRONTLOCS_ALL[:, hh-1, 1] - HEELLOCS_AVG[:, hh-1, 1], nsteps)
            HEELLOC_INT_X = np.linspace(HEELLOCS_AVG[:, hh-2, 0], HEELLOCS_AVG[:, hh-1, 0], nsteps)
            HEELLOC_INT_Y = np.linspace(HEELLOCS_AVG[:, hh-2, 1], HEELLOCS_AVG[:, hh-1, 1], nsteps)                    
        else:
            FRO_INT_X = np.linspace(HEELLOCS_AVG[:, 0, 0], FRONTLOCS_ALL[:, 0, 0], nsteps) - HEELLOCS_AVG[:, 0, 0]
            FRO_INT_Y = np.linspace(HEELLOCS_AVG[:, 0, 1], FRONTLOCS_ALL[:, 0, 1], nsteps) - HEELLOCS_AVG[:, 0, 1]
            HEELLOC_INT_X = np.linspace(HEELLOCS_AVG[:, 0, 0], HEELLOCS_AVG[:, 0, 0], nsteps)
            HEELLOC_INT_Y = np.linspace(HEELLOCS_AVG[:, 0, 1], HEELLOCS_AVG[:, 0, 1], nsteps)
                #HEELLOC_INT_X += np.array([512]) #+ v1inter[None, :, 0]
            #HEELLOC_INT_Y += np.array([256]) #+ v1inter[None, :, 1]
        
        if hh == 1:
            # FRONTLOC = np.vstack((HEELLOC_INT_X[0, :], HEELLOC_INT_Y[0, :])).T
            if all_receptors:
                n1, n2 = np.meshgrid(np.arange(-(nr-1)/2, (nr-1)/2 + 0.1), np.arange(-(nr-1)/2, (nr-1)/2 + 0.1))
                center_loc_x = (Xstart[0] + n1*v1old[0] + n2*v2old[0]).flatten()
                center_loc_y = (Ystart[0] + n1*v1old[1] + n2*v2old[1]).flatten()                    
                if noLcell:
                    RECX, RECY, XC_, YC_ = create_starting_grid_noLcell(center_loc_x, center_loc_y)
                else:
                    RECX, RECY, XC_, YC_ = create_starting_grid2(center_loc_x, center_loc_y, hour)
            else:
                if noLcell:
                    RECX, RECY, XC_, YC_ = create_starting_grid_noLcell(np.array([Xstart[0]]), np.array([Ystart[0]]))
                else:
                    RECX, RECY, XC_, YC_ = create_starting_grid2(np.array([Xstart[0]]), np.array([Ystart[0]]), hour)
            
            if all_receptors:
                FRONTLOC = np.c_[np.ravel(RECX), np.ravel(RECY)]
                # FRONTLOC[:nr**2, :] += v1old
                # FRONTLOC[nr**2:2*nr**2, :] -= -v1old + v2old
                # FRONTLOC[2*nr**2:3*nr**2, :] -= -v1old + 2*v2old
                # FRONTLOC[3*nr**2:4*nr**2, :] -= v2old
                # FRONTLOC[4*nr**2:5*nr**2, :] -= v1old
                # FRONTLOC[5*nr**2:, :] -= v1old - v2old
                # FRONTLOC += v2old

            else:
                FRONTLOC = np.c_[RECX, RECY]
                FRONTLOC[0, :] += v1old
                FRONTLOC[1, :] -= -v1old + v2old
                FRONTLOC[2, :] -= -v1old + 2*v2old
                FRONTLOC[3, :] -= v2old
                FRONTLOC[4, :] -= v1old
                FRONTLOC[5, :] -= v1old - v2old
                FRONTLOC += v2old
        
        if False: #hour == 35:
            # calculate shortest distance to goal area at P35
            if noLcell:
                RECX, RECY, XC_, YC_ = create_starting_grid_noLcell(np.zeros(1), np.zeros(1))
            else:
                RECX, RECY, XC_, YC_ = create_starting_grid2(np.zeros(1), np.zeros(1), hour)
            goalloc = np.zeros((6, 2))
            goalloc[0, :] -= v1new
            goalloc[1, :] += -v1new + v2new
            goalloc[2, :] += -v1new + 2*v2new
            goalloc[3, :] += v2new
            goalloc[4, :] += v1new
            goalloc[5, :] += v1new - v2new
            
            clos_points = np.zeros((6, 2))
            # mpl.use('Qt5Agg') # turn on GUI for plotting

            # plt.figure()
            # plt.scatter(RECX, RECY)
            # plt.scatter(goalloc[:, 0], goalloc[:, 1])
            # plt.scatter(RECX - goalloc[:, 0] + 0.333*v2new[0], RECY - goalloc[:, 1] + 0.333*v2new[1])
            # plt.scatter(v1new[0], v1new[1])
            # plt.scatter(v2new[0] - 0.333*v2new[0], v2new[1] - 0.333*v2new[1])
            # plt.scatter(0, 0)
            # plt.plot(np.linspace(-60, 60, 120), np.sqrt((LCELL_SIZE[0, 1]*CONVFAC/2)**2 - np.linspace(-60, 60, 120)**2))
            # plt.plot(np.linspace(-60, 60, 120), -np.sqrt((LCELL_SIZE[0, 1]*CONVFAC/2)**2 - np.linspace(-60, 60, 120)**2))
            
            for rr in range(1, 5):
                clos_points[rr, :] = calc_closest_point_on_ellipse(LCELL_SIZE[hh-1, 1]*CONVFAC/2, LCELL_SIZE[hh-1, 1]*CONVFAC/2, np.array([[np.r_[RECX[rr] - goalloc[rr, 0] + 0.333*v2new[0], RECY[rr] - goalloc[rr, 1] + 0.333*v2new[1]]]]))
            for rr in [0, 5]:
                clos_points[rr, :] = calc_closest_point_on_ellipse(0.8*LCELL_SIZE[hh-1, 0]*CONVFAC/2, 0.8*LCELL_SIZE[hh-1, 1]*CONVFAC/2, np.array([[np.r_[RECX[rr] - goalloc[rr, 0], RECY[rr] - goalloc[rr, 1]]]]))
            
            distances[hh-1, :] = np.sqrt((clos_points[:, 0] + goalloc[:, 0] - 0.333*v2new[0] - RECX[:, 0])**2 + (clos_points[:, 1] + goalloc[:,1] - 0.333*v2new[1] - RECY[:, 0])**2)
            for rr in [0, 5]:
                distances[hh-1, rr] = np.sqrt((clos_points[rr, 0] + goalloc[rr, 0] - RECX[rr, 0])**2 + (clos_points[rr, 1] + goalloc[rr,1] - RECY[rr, 0])**2)
            
        print('')
        TIME_NOW = np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=False) if hour < 40 else np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=True)
        dt = np.diff(TIME_NOW)[0]
        TIME = np.hstack((TIME, TIME_NOW))
        for tind, tt in enumerate(TIME_NOW): #enumerate(np.linspace((hours[hh-1]+1+hours[hh])/2, hours[hh], fading, endpoint=False)):
            print('\r'+str(hour)+'hAPF: |' + (tind+1)*'=' + (nsteps-tind-1)*' ' + '|', end='')
            if create_movie:
                fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 20]})
                ax[0].barh([1], [41], color='w', edgecolor='k')
                ax[0].barh([1], [tt], color='k')
                ax[0].set_title('Developmental time (hAPF)')
                ax[0].tick_params(axis='y', which='both', right=False,
                                  left=False, labelleft=False)
                ax[0].set_xticks([20, 25, 30, 35, 40])
                for pos in ['right', 'top', 'bottom', 'left']:
                    ax[0].spines[pos].set_visible(False)
                
                ax[0].set_xlim([19.9, 41.1])
            
            exp_fronts_x = FRONTLOCS_ALL[:, :, 0]
            exp_fronts_y = FRONTLOCS_ALL[:, :, 1]
            heel_locs_x = HEELLOCS_AVG[:, :, 0]
            heel_locs_y = HEELLOCS_AVG[:, :, 1]

            center_loc_x_int = (Xstart[tind] + n1*v1inter[tind, 0] + n2*v2inter[tind, 0]).flatten()
            center_loc_y_int = (Ystart[tind] + n1*v1inter[tind, 1] + n2*v2inter[tind, 1]).flatten()
            ind = (center_loc_x_int > 0)*(center_loc_x_int < 2048)*(center_loc_y_int > 0)*(center_loc_y_int < 1024)
            Xint, Yint = center_loc_x_int[ind], center_loc_y_int[ind]
            
            if noLcell:
                if hh == 2:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid_noLcell(Xint, Yint)
                else:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid_noLcell(Xint, Yint)
            else:
                if hh == 2:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour, tind, nsteps)
                else:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour)

            FRO_X2_int = REC_X_INTER + FRO_INT_X[tind, None].T
            FRO_Y2_int = REC_Y_INTER + FRO_INT_Y[tind, None].T
            
            if tind == 0 and hh == 1:
                FRONTLOCS_TIME_temp = FRONTLOC.copy()
                FRONTLOCS_TIME = FRONTLOCS_TIME_temp.copy()

            xmin2, xmax2, ymin2, ymax2 = 0, 2*Nx, 0, 2*Ny
            POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx-1), np.linspace(ymin2, ymax2, 2*Ny-1))
            POS = np.array(POS) #+ np.array([512, 256])[:, None, None]#[:, 256:768, 512:1536]
            if noLcell:
                dat2_inter = np.load('./data-files/dat2_inter_R4_'+str((15*(hh-1)) + tind)+'hAPF_noLcell_noeq.npy')
            else:
                dat2_inter = np.load('./data-files/dat2_inter_R1_'+str(hours[hh])+'hAPF_'+str(tind)+'_noeq.npy')

            # create index array for the circle
            if single_receptor:
                xfil, yfil, allinds = sample_roi(dat2_inter, FRONTLOC[single_receptor-1, :], meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER)
            else:
                xfil, yfil, allinds = sample_roi(dat2_inter, FRONTLOC, meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER)                
                xfil= np.reshape(xfil, (-1, 10))
                yfil= np.reshape(yfil, (-1, 10))
            
            # how many of the filopodia are in any of the target areas
            if np.ndim(FRONTLOCS_TIME) == 3:
                a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                for rr in range(6):
                    pos_eval = np.c_[xfil[rr, :], yfil[rr, :]]
                    D_bundles = dist.cdist(pos_eval, [[x, y] for x, y in zip(Xint, Yint)], metric='euclid')
                    closest_bundle = np.zeros(np.shape(pos_eval))
                    closest_bundle[:, 0] = Xint[D_bundles.argmin(axis=1)]
                    closest_bundle[:, 1] = Yint[D_bundles.argmin(axis=1)]

                    correct1 = (pos_eval[:, 0] - closest_bundle[:, 0] + 0.333*v2inter[tind][0])**2 + (pos_eval[:, 1] - closest_bundle[:, 1] + 0.333*v2inter[tind][1])**2 < 1.2*b**2
                    
                    correct2 = np.sqrt((pos_eval[:, 0] - closest_bundle[:, 0])**2/a**2 + (pos_eval[:, 1] - closest_bundle[:, 1])**2/b**2) <= 1 #0.8
                    correct = np.logical_or(correct1, correct2)

                    outside_fil_all[ii, rr, :] = correct
                ii += 1
            
            angl = np.arctan2(yfil-FRONTLOC[:, 1][:, None], xfil-FRONTLOC[:, 0][:, None])
            radi = np.sqrt((yfil-FRONTLOC[:, 1][:, None])**2 + (xfil-FRONTLOC[:, 0][:, None])**2)
            Xs, Ys = calc_closest_point_on_ellipse(lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC, FRONTLOC[None, :, :] - np.array([[x, y] for x,y in zip(Xint, Yint)])[:, None, :])
            goal_loc = np.zeros((2, len(FRONTLOC)))
            for kk in range(len(FRONTLOC)):
                D_bundles = dist.cdist([FRONTLOC[kk, :]], [[x, y] for x, y in zip(Xint + Xs[:, kk], Yint + Ys[:, kk])], metric='euclid')
                goal_loc[0, kk] = Xint[D_bundles.argmin(axis=1)]
                goal_loc[1, kk] = Yint[D_bundles.argmin(axis=1)]
            
            # mpl.use('Qt5Agg') # turn on GUI for plotting            
            # print(FRONTLOC - goal_loc.T)
            # plt.figure()
            # plt.scatter(FRONTLOC[:, 0], FRONTLOC[:, 1])
            # plt.scatter(goal_loc[0, :], goal_loc[1, :])
            # asdasdsa
            
            if create_movie:
                len1 = 100                            
                viridis = cm.get_cmap('viridis', len1)
                rgba_vir = viridis.colors                            
                norm = plt.Normalize(np.min(dat2_inter[200:800, 1300:1600]), np.max(dat2_inter[200:800, 200:1600]))
                cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                         'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                         'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}
                newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
                ax[1].imshow(dat2_inter, origin='lower', cmap=newcmp, norm=norm)
    
                # ax[1].imshow(dat2_inter, origin='lower', vmin=np.min(dat2_inter[200:800, 1300:1600]), vmax=np.max(dat2_inter[200:800, 1300:1600]))
                
                if not all_receptors:
                    ax[1].imshow(np.reshape(allinds, (1023, 2047)), origin='lower', alpha=0.2)
                    
                # ax[1].scatter(exp_fronts_x, exp_fronts_y, c='k')
                if single_receptor:
                    ax[1].scatter(*FRONTLOC[single_receptor-1, :], c='r', zorder=10)
                    ax[1].scatter(xfil, yfil, s=5, c='w', zorder=10)
                    if np.ndim(FRONTLOCS_TIME) == 3:
                        ax[1].plot(FRONTLOCS_TIME[:, single_receptor-1, 0], FRONTLOCS_TIME[:, single_receptor-1, 1], 'r-', lw=2)
                        ax[1].scatter(FRONTLOCS_TIME[0, single_receptor-1, 0], FRONTLOCS_TIME[0, single_receptor-1, 1], c='b', zorder=5)
                    else:
                        ax[1].scatter(FRONTLOCS_TIME[single_receptor-1, 0], FRONTLOCS_TIME[single_receptor-1, 1], c='b', zorder=5)
                    
                else:
                    for irec in range(len(FRONTLOC)):
                        ax[1].scatter(*FRONTLOC[irec, :], c='r', zorder=10)
                        ax[1].scatter(xfil, yfil, s=5, c='w', zorder=10)
                        if np.ndim(FRONTLOCS_TIME) == 3:
                            ax[1].plot(FRONTLOCS_TIME[:, irec, 0], FRONTLOCS_TIME[:, irec, 1], 'r-', lw=2)
                            ax[1].scatter(FRONTLOCS_TIME[0, irec, 0], FRONTLOCS_TIME[0, irec, 1], c='b', zorder=5)
                        else:
                            ax[1].scatter(FRONTLOCS_TIME[irec, 0], FRONTLOCS_TIME[irec, 1], c='b', zorder=5)
                
                if noLcell:
                    for xx, yy in zip(Xint, Yint):
                        circle_temp = Point(xx, yy).buffer(1)  # type(circle)=polygon
                        ellipse = shapely.affinity.scale(circle_temp, 2.96*24/2, 2.58*24/2)  # type(ellipse)=polygon            
                        circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer(0.6*2.58*24)
                        uni = circle.union(ellipse)
                        ax[1].add_patch(descartes.PolygonPatch(uni, fc='g', ec='g', alpha=0.2))
                        # circle = Point(xx - 0.4*v2inter[tind, 0], yy - 0.4*v2inter[tind, 1]).buffer(0.6*lcellsize_int[tind, 1]*CONVFAC)
                        # ax[1].add_patch(descartes.PolygonPatch(circle, fc='g', ec='g', alpha=0.2))
                else:
                    for xx, yy in zip(Xint, Yint):
                        circle_temp = Point(xx, yy).buffer(1)  # type(circle)=polygon
                        ellipse = shapely.affinity.scale(circle_temp, 0.8*lcellsize_int[tind, 0]*CONVFAC/2, 0.8*lcellsize_int[tind, 1]*CONVFAC/2)  # type(ellipse)=polygon            
                        if hour == 26:
                            circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer((0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC)
                        else:
                            circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer(0.6*lcellsize_int[tind, 1]*CONVFAC)
                        
                        uni = circle.union(ellipse)
                        ax[1].add_patch(descartes.PolygonPatch(uni, fc='g', ec='g', alpha=0.2))
                        # # if xx not in Xorig and yy not in Yorig:
                        # ax[1].add_patch(Ellipse((xx, yy), 0.8*lcellsize_int[tind, 0]*CONVFAC, 0.8*lcellsize_int[tind, 1]*CONVFAC, angle=0, edgecolor = 'green', ls='-', facecolor='green', alpha=0.3))
                        # ax[1].add_patch(Circle((xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]), 0.5*lcellsize_int[tind, 1]*CONVFAC, edgecolor = 'green', ls='-', facecolor='green', alpha=0.3))
                    
                for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                    # if xx not in REC_Xorig and yy not in REC_Yorig:
                    ax[1].add_patch(Circle((xx, yy), 0.5*CONVFAC, ls = '--', edgecolor='k', facecolor='none'))
                
                if False:
                    ax[1].axis([750, 1220, 350, 620])
                    ax[1].add_patch(Rectangle((1170, 370), CONVFAC*5, 10,
                                    edgecolor = 'white',
                                    facecolor = 'white',
                                    fill=True))
                    ax[1].text(1220, 385, '5 $\mu$m', c='w')
                else:
                    ax[1].axis([900, 1500, 350, 750])
                    ax[1].add_patch(Rectangle((1350, 370), CONVFAC*5, 10,
                                    edgecolor = 'white',
                                    facecolor = 'white',
                                    fill=True))
                    ax[1].text(1400, 385, '5 $\mu$m', c='w')

                remove_ticks_and_box(ax[1])

    
                fig.tight_layout()
                inset_axes(ax[1], width='20%', height='20%', loc=2)
                plt.plot(np.arange(20, 41, 0.01), 1-tanh_cust(np.arange(20, 41, 0.01), c_st, sl_st), 'k-', lw=3)
                # xval = np.mean(np.sqrt((FRONTLOC[:, 0] - heel_locs_x[:, 0])**2 + (FRONTLOC[:, 1] - heel_locs_y[:, 0])**2))/CONVFAC
                plt.scatter(tt, 1-tanh_cust(tt, c_st, sl_st), c='r', s=40, zorder=5)
                # plt.plot(np.arange(0, 10, 0.01), 1-tanh_cust(np.arange(0, 10, 0.01)*CONVFAC, c_st, sl_st), 'k-', lw=3)
                # xval = np.mean(np.sqrt((FRONTLOC[:, 0] - heel_locs_x[:, 0])**2 + (FRONTLOC[:, 1] - heel_locs_y[:, 0])**2))/CONVFAC
                # plt.scatter(xval, 1-tanh_cust(xval*CONVFAC, c_st, sl_st), c='r', s=40, zorder=5)
                plt.ylim([-0.1, 1.1])
                plt.xlabel('Time (hour)', color='w')
                # plt.xlabel('Length of axon (um)', color='w')
                plt.ylabel('Stiffness', color='w')
                plt.gca().yaxis.set_label_position("right")
                plt.gca().yaxis.tick_right()
                plt.gca().tick_params(axis='x', colors='w')
                plt.gca().tick_params(axis='y', colors='w')
    
                inset_axes(ax[1], width='20%', height='20%', loc=1)
                if noLcell:
                    plt.plot(np.arange(20, 41, 0.01), np.zeros(len(np.arange(20, 41, 0.01))), 'k-', lw=3)
                    plt.scatter(tt, 0, c='r', s=40, zorder=5)
                else:
                    plt.plot(np.arange(20, 41, 0.01), tanh_cust(np.arange(20, 41, 0.01), c_magnet, sl_magnet), 'k-', lw=3)
                    plt.scatter(tt, tanh_cust(tt, c_magnet, sl_magnet), c='r', s=40, zorder=5)
                plt.ylim([-0.1, 1.1])
                plt.xlabel('Time (hour)', color='w')
                plt.ylabel('L-cell attraction', color='w')
                plt.gca().tick_params(axis='x', colors='w')
                plt.gca().tick_params(axis='y', colors='w')
    
                
                filename = str('%04d' % (100*hour+tind) + '.png')
                plt.savefig(filename, dpi=100)
                plt.close(fig)
                        
            #########################
            # the computational part:                
            
            if np.ndim(FRONTLOCS_TIME) == 3:
                norm = np.shape(FRONTLOCS_TIME)[0]-1
                axon_len = np.sqrt((FRONTLOC[:, 0] - FRONTLOCS_TIME[0, :, 0])**2 + (FRONTLOC[:, 1] - FRONTLOCS_TIME[0, :, 1])**2)
            else:
                norm = 1
                axon_len = np.sqrt((FRONTLOC[:, 0] - FRONTLOCS_TIME[:, 0])**2 + (FRONTLOC[:, 1] - FRONTLOCS_TIME[:, 1])**2)
            
            # stiff = 1 - tanh_cust(axon_len, c_st, sl_st)
            
            stiff = 1 - tanh_cust(tt, c_st, sl_st)
            
            if np.ndim(FRONTLOCS_TIME) == 3:
                norm = np.shape(FRONTLOCS_TIME)[0]-1
            else:
                norm = 1
            #stiff = 1
            stiffpart = stiff*meanvec_old/norm
            if novalleydyn:
                filopart = (1-stiff)*speeds*np.abs(np.mean(radi*np.exp(1j*angl), axis=1))*np.exp(1j*np.angle(stiffpart))
            else:
                if all_receptors:
                    filopart = (1-stiff)*np.repeat(speeds, nr**2)*np.mean(radi*np.exp(1j*angl), axis=1)                    
                else:
                    filopart = (1-stiff)*speeds*np.mean(radi*np.exp(1j*angl), axis=1)
            # magnetpart = A_magnet * tanh_cust(tt, c_magnet, sl_magnet) * np.exp(1j*np.arctan2(goal_loc[1]-FRONTLOC[:, 1], goal_loc[0]-FRONTLOC[:, 0]))
            magnet_weight = tanh_cust(tt, c_magnet, sl_magnet)
            
            # check whether front location is in one of the L-cell regions
            inside_goal_area = np.zeros(len(FRONTLOC), dtype='bool')
            if np.ndim(FRONTLOCS_TIME) == 3:
                a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                v1int = v2inter[tind].copy()
                v2int = v1inter[tind].copy()
                # inside_goal_area = distance_to_exp(FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2inter[tind], v1inter[tind], mode='target_area')
                for kk in range(len(FRONTLOC)):
                    for xx, yy in zip(Xint, Yint):
                        check1 = (np.sqrt((FRONTLOC[kk, 0] - xx + 0.333*v1int[0])**2 + (FRONTLOC[kk, 1] - yy + 0.333*v1int[1])**2) < (1.2*b + CONVFAC/4))
                        if include_equator and xx >= border:
                                check1 = (np.sqrt((FRONTLOC[kk, 0] - xx - 0.333*v1int[0])**2 + (FRONTLOC[kk, 1] - yy - 0.333*v1int[1])**2) < (1.2*b + CONVFAC/4))
                        
                        check2 = ((FRONTLOC[kk, 0] - xx)**2/(a+CONVFAC/4)**2 + (FRONTLOC[kk, 1] - yy)**2/(b+CONVFAC/4)**2 <= 1**2) #0.8
                        if check1 or check2:
                            inside_goal_area[kk] = True
                            # check for degree of overlap if there is overlap with multiple bundles
                            if include_equator and xx >= border:
                                goal_loc[:, kk] = np.array([xx + lcellsize_int[tind, 0]/3*CONVFAC, yy])
                            else:
                                goal_loc[:, kk] = np.array([xx  - lcellsize_int[tind, 0]/3*CONVFAC, yy])

            if all_receptors:
                magnetpart = magnet_weight * A_magnet * np.exp(1j*np.arctan2(goal_loc[1]-FRONTLOC[:, 1], goal_loc[0]-FRONTLOC[:, 0])) #-lcellsize_int[tind, 0]/2*CONVFAC
            else:
                magnetpart = magnet_weight * A_magnet * np.exp(1j*np.arctan2(goal_loc[1]-FRONTLOC[:, 1], goal_loc[0]-FRONTLOC[:, 0])) #-lcellsize_int[tind, 0]/2*CONVFAC
                print(A_magnet)
                print(magnet_weight)
                print(magnetpart)

            # print(inside_goal_area)
            if A_magnet == 0:
                meanvec = stiffpart + filopart
            else:
                meanvec = np.where(inside_goal_area, (1-magnet_weight)*(stiffpart + filopart) + magnetpart, stiffpart + filopart)
            
            # meanvec = stiffpart + filopart + magnetpart
            
            # check for hard turns and soften them
            # meanvec = np.where(np.cos(np.angle(meanvec) - np.angle(meanvec_old)) < 0.9396926207859084, (meanvec + meanvec_old/norm) / 2, meanvec)
            
            # check if axons are close to magnet goal location, if so: stop movement
            meanvec = np.where(np.sqrt((goal_loc[0]-FRONTLOC[:, 0])**2 + (goal_loc[1]-FRONTLOC[:, 1])**2) <= 5, 0+0*1j, meanvec)
            
            change = np.array([meanvec.real, meanvec.imag]).T
            
            if tind > 0:
                currmat = np.array([v1inter[tind], v2inter[tind]]).T
                prevmat = np.array([v1inter[tind-1], v2inter[tind-1]]).T
                change = (currmat@np.linalg.inv(prevmat)@change.T).T
            FRONTLOC += change * dt
            # meanvec = change[:, 0]*dt + 1j*change[:, 1]*dt
            FRONTLOCS_TIME_temp = np.vstack((FRONTLOCS_TIME_temp, FRONTLOC))
            FRONTLOCS_TIME = np.reshape(FRONTLOCS_TIME_temp.copy(), (-1, len(FRONTLOC), 2))
            meanvec_old = (FRONTLOCS_TIME[-1, :, 0] - FRONTLOCS_TIME[0, :, 0] + 1j*(FRONTLOCS_TIME[-1, :, 1] - FRONTLOCS_TIME[0, :, 1]))/dt # meanvec.copy()
            
            if tind > 0:
                for r in range(len(FRONTLOC)):
                    D_rec = dist.cdist([FRONTLOCS_TIME[0, r, :]], [[x, y] for x, y in zip(np.ravel(np.repeat(REC_X_INTER, nr**2, axis=0)[r, :]), np.ravel(np.repeat(REC_Y_INTER, nr**2, axis=0)[r, :]))], metric='euclid')
                
                    FRONTLOCS_TIME[:, r, 0] = FRONTLOCS_TIME[:, r, 0] + np.repeat(REC_X_INTER, nr**2, axis=0)[r, D_rec.argmin()] - FRONTLOCS_TIME[0, r, 0]
                    FRONTLOCS_TIME[:, r, 1] = FRONTLOCS_TIME[:, r, 1] + np.repeat(REC_Y_INTER, nr**2, axis=0)[r, D_rec.argmin()] - FRONTLOCS_TIME[0, r, 1]
            
            FRONTLOCS_TIME_temp = np.reshape(FRONTLOCS_TIME, (-1, 2))            
            FRONTLOC = FRONTLOCS_TIME[-1, :, :]
            
            if False: #np.sqrt(np.sum((FRONTLOC - FRONTLOCS_TIME[0, :, :])**2))/ > 0.5*CONVFAC:
                D_rec = dist.cdist(FRONTLOC, [[x, y] for x, y in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER))], metric='euclid')
                for r in range(6):
                    if (D_rec[r, :] < 0.5*CONVFAC).any():
                        ang = np.arctan2(FRONTLOC[r, 1] - np.ravel(REC_Y_INTER)[D_rec[r, :].argmin()], FRONTLOC[r, 0] - np.ravel(REC_X_INTER)[D_rec[r, :].argmin()])
                        FRONTLOC[r, :] = np.r_[np.ravel(REC_X_INTER)[D_rec[r, :].argmin()], np.ravel(REC_Y_INTER)[D_rec[r, :].argmin()]] + 0.5 * CONVFAC * np.r_[np.cos(ang), np.sin(ang)]
                        FRONTLOCS_TIME[-1, r, :] = FRONTLOC[r, :].copy()
    return distances, outside_fil_all

mpl.use('Agg') # do not show figures while creating movie

create_movie = True

include_equator = False

single_receptor = False #False if you want all 6 receptors in the video
all_receptors = False
if all_receptors:
    nr = 5
else:
    nr = 1

noLcell = True

# stiff = 0 # 0.7
if noLcell:
    A_magnet = 0
else:
    A_magnet = 1 #5
CONVFAC = 25.2
# parameters of the model
nsteps = 15
# radii = np.array([1, 1, 1, 0.8, 1, 1])*np.array([4.41, 3.60, 5.15, 3.58, 4.48, 4.74])*25.2
radii = np.array([4.41, 3.60, 5.15, 3.58, 4.48, 4.74])*25.2
speeds = np.array([0.093, 0.053, 0.148, 0.09, 0.052, 0.077])

# speeds = 0.12*np.ones(6)
# speeds[2] = 0.12 # different speed for R3

n_fil = 10
region = 'circ_segment'
novalleydyn = False

hours = np.array([20, 26, 30, 35, 40])

LCELL_SIZE = np.array([[3.96, 2.23], [3.96, 2.23], [3.9, 2.28], [4.71, 2.61], [4.57, 2.63]])

# circ_width = #np.array([1, 1, 1, 0.8, 1, 1]) * \
circ_width = np.array([[76.93677250002409, 66.1581562071056, 58.64359788352946, 68.19374152821266],
                        [67.10341096706019, 63.6030367287868, 64.58480307212197, 64.68382969250712],
                        [59.99951401507621, 49.70632397768759, 59.76089748808765, 62.63689465660343],
                        [65.18759340181808, 64.6663385598332, 54.128563634672744, 58.3338061658033],
                        [73.7705462792746, 73.75319815476855, 70.71260029847282, 67.54024198457094],
                        [82.29011450458808, 70.63559641125377, 69.51033837975177, 58.19852457593135]]).mean(axis=1) 

# front distances and angles from Marion's excel sheet (Cell 2015 paper)
fronts_dist_old = np.array([[2.25454545454545, 2.6247619047619, 3.91962962962963, 4.17666666666667],
                            [1.65, 1.84166666666667, 2.24222222222222, 2.27866666666667],
                            [2.76105263157895, 4.48, 6.64208333333333, 6.92636363636363],
                            [2.1776, 2.64666666666667, 3.77363636363636, 4.29125],
                            [2.26, 2.22333333333333, 2.356, 2.4675],
                            [2.02, 1.99666666666667, 3.43259259259259, 4.74222222222222]])

front_ang_old = np.array([[28.2566666666667, 27.017619047619, 26.5007407407407, 30.0891666666667],
                          [139.81, 138.655833333333, 141.716111111111, 143.435333333333],
                          [156.339473684211, 156.907333333333, 159.620416666667, 163.584242424242],
                          [179.5208, 180.847272727273, 181.234848484849, 179.0175],
                          [-154.426666666667, -146.302222222222, -145.667333333333, -144.179166666667],
                          [-26.0233333333333, -27.447619047619, -25.7514814814815, -31.4866666666667]])

# load stuff and incorporate the old front locations
vec_opt, FRONTLOCS_ALL, HEELLOCS_AVG = load_grid_heels_fronts()
#FRONTLOCS_ALL += np.array([512, 256])[None, None, :]
# FRONTLOCS_ALL[:, :, 1] = -FRONTLOCS_ALL[:, :, 1]
HEELLOCS_AVG += np.array([1024, 512])[None, None, :]
FRONTLOCS_ALL = HEELLOCS_AVG + FRONTLOCS_ALL*25.2
# rotate everything to have the grid perfectly horizontal

if noLcell:
    newtime = np.linspace(20, 45, 19, endpoint=True)
    ind_time = []
    for t in [25, 30, 35, 40]:
        ind_time.append(np.argmin(np.abs(newtime - t)))

    with open("./data-files/optimal_grid_noLcell.json", "r") as infile:
        loaded = json.load(infile).split('\"')
    dtype = np.dtype(loaded[1])
    arr = np.frombuffer(base64.decodestring(bytearray(loaded[3], 'utf-8')), dtype)
    vec_opt = arr.reshape([19, 4]).copy()[ind_time, :]
    # v1, v2 = vec_opt[8, :2], vec_opt[8, 2:]
    # v1 = rotate_coord(v1[0], v1[1], -np.arctan2(v2[1], v2[0]))
    # v2 = rotate_coord(v2[0], v2[1], -np.arctan2(v2[1], v2[0]))

rot_ang = []
for hh in range(4):
    rot_ang.append(-np.arctan2(vec_opt[hh, 3], vec_opt[hh, 2]))
    # FRONTLOCS_ALL[:, hh, 0], FRONTLOCS_ALL[:, hh, 1] = rotate_coord(FRONTLOCS_ALL[:, hh, 0], FRONTLOCS_ALL[:, hh, 1], rot_ang[-1])
    # HEELLOCS_AVG[:, hh, 0], HEELLOCS_AVG[:, hh, 1] = rotate_coord(HEELLOCS_AVG[:, hh, 0], HEELLOCS_AVG[:, hh, 1], rot_ang[-1])
    vec_opt[hh, ::2], vec_opt[hh, 1::2] = rotate_coord(vec_opt[hh, ::2], vec_opt[hh, 1::2], rot_ang[-1])

# FRONTLOCS_ALL[:, :, 0] = HEELLOCS_AVG[:, :, 0] + CONVFAC * fronts_dist_old * np.cos(np.pi/180 * front_ang_old - np.pi)
# FRONTLOCS_ALL[:, :, 1] = HEELLOCS_AVG[:, :, 1] + CONVFAC * fronts_dist_old * np.sin(np.pi/180 * front_ang_old - np.pi)


mini, maxi = -10, 10
n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
Nx, Ny = 1024, 512

# startang = np.array([np.arctan2(FRONTLOCS_ALL[rec, 0, 1] - HEELLOCS_AVG[rec, 0, 1], FRONTLOCS_ALL[rec, 0, 0] - HEELLOCS_AVG[rec, 0, 0]) for rec in range(6)])
#recx, recy, xc, yc = create_starting_grid2(np.array([0]), np.array([0]), 26)
# startang = np.arctan2(recy[:, 0], recx[:, 0])
# startang[2] = -10 * np.pi/180
#centr = [np.mean(recx), np.mean(recy)]
# startangs = np.arctan2(recy[:, 0] - centr[1], recx[:, 0] - centr[0])
# startangs[0] += np.pi/180 * 20
# startangs[1] += np.pi/180 * 20
# startangs[2] += np.pi/180 * 20

if noLcell:
    startangs = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 13.26706/2, 63.2865, 135.0751667])
else:
    startangs = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 13.26706/2, 63.2865, 135.0751667])
    
# meanvec_old = np.ones(nr_of_rec) * sp*radii[0]/2 * np.exp(1j*startang[0]) / 6 # dt approx 1/3
c_st, sl_st = 25, 0.45

dists, outside_fil = run_gc(c_st, sl_st, startangs, A_magnet)
if create_movie:
    create_movie_from_png('front_movement_model', remove_png_afterwards=False)
    mpl.use('Qt5Agg') # turn on GUI for plotting


#%%
asdsadsadsad
nrep = 100
for i in range(nrep):
    dists, outside_fil = run_gc(c_st, sl_st, startangs, A_magnet)
    np.save('outside_fil_'+str(i), outside_fil)

outside_fil_all = np.zeros((nrep, 59, 6, 10))
for i in range(nrep):
    outside_fil_all[i, :, :, :] = np.load('outside_fil_'+str(i)+'.npy')

TIME = np.array([])
for hh, hour in enumerate(hours[1:]):
    hh += 1
    TIME_NOW = np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=False) if hour < 40 else np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=True)
    TIME = np.hstack((TIME, TIME_NOW))

mpl.use('Qt5Agg') # turn on GUI for plotting
meanfil = np.mean(outside_fil_all, axis=(0, 2, 3))
semfil = np.std(outside_fil_all, axis=(0, 2, 3))/np.sqrt(nrep * 6)
plt.figure(figsize=(10, 7))
plt.plot(TIME[1:], meanfil, 'k-', lw=2)
plt.plot(TIME[1:], meanfil + semfil, 'k--')
plt.plot(TIME[1:], meanfil - semfil, 'k--')
plt.xlabel('Developmental time (hAPF)', fontsize=20)
plt.ylabel('Proportion of filopodial tips\ninside target area', fontsize=20)
plt.xticks([25, 30, 35, 40], fontsize=20)
plt.yticks([0, 0.1, 0.2, 0.3], fontsize=20)
plt.xlim([25, 40])

COL = np.array(['c', 'g', 'r', 'y', 'tab:purple', 'tab:orange'])
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
for rr in range(6):
    meanfil = np.mean(outside_fil_all[:, :, rr, :], axis=(0, 2))
    semfil = np.std(outside_fil_all[:, :, rr, :], axis=(0, 2))/np.sqrt(nrep)
    ax[rr//2, rr%2].plot(TIME[1:], meanfil, COL[rr], lw=2)
    ax[rr//2, rr%2].plot(TIME[1:], meanfil + semfil, COL[rr], ls='--')
    ax[rr//2, rr%2].plot(TIME[1:], meanfil - semfil, COL[rr], ls='--')
    ax[rr//2, rr%2].set_xticks([25, 30, 35, 40])
    ax[rr//2, rr%2].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    if rr == 4:
        ax[rr//2, rr%2].set_xlabel('Developmental time (hAPF)')
        ax[rr//2, rr%2].set_ylabel('Proportion of filopodial tips\ninside target area')
    ax[rr//2, rr%2].set_xlim([25, 40])
    # ax[rr//2, rr%2].set_title('Receptor' + str(rr+1))
    