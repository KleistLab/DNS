#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:57:50 2022

@author: eric
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle#, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance as dist
from scipy.stats import circstd
import numexpr as ne
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from joblib import Parallel, delayed
import shapely.affinity
from shapely.geometry import Point
import descartes
from helper_functions import \
load_grid_heels_fronts, remove_ticks_and_box, create_heatmap, rotate_coord,\
    create_starting_grid2, calc_closest_point_on_ellipse, create_movie_from_png

start = time.time()

def generate_indmat(xind, yind, fr_x, fr_y, r1, frontang, REC_X_INTER, REC_Y_INTER):
    """
    creates the mask for placing the filopodia in the density landscape
    """
    rec_x, rec_y = np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)
    rec_x_red = rec_x[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r1**2]
    rec_y_red = rec_y[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r1**2]
    ind = np.zeros(np.shape(xind), dtype='bool')
    ind1 = ne.evaluate('((xind-fr_x)**2 + (yind-fr_y)**2)<=r1**2')
    ind2 = (np.cos(np.arctan2(yind[ind1]-fr_y, xind[ind1]-fr_x) - frontang)
            > np.cos(np.pi/180*circ_width)) #0.643) #0.25882) #0.77)
    ind[yind[ind1][ind2], xind[ind1][ind2]] = True
    radius = 0.5 * CONVFAC
    ind3 = np.zeros(np.shape(xind), dtype='bool')
    for rx, ry in zip(rec_x_red, rec_y_red):
        ind3 += ne.evaluate('((xind-rx)**2 + (yind-ry)**2)<=radius**2')
    return np.logical_and(ind, ~ind3)

def sample_roi(dat2_inter, FRONTLOC, meanvec_old, mask, xind, yind,
               POS, REC_X_INTER, REC_Y_INTER, r1):
    """
    samples n_fil filopodial locations from the mask region created in
    generate_indmat()
    """
    xfil, yfil = [], []
    allinds = np.zeros(1023*2047, dtype='bool')
    for irec in range(nr_of_rec):
        if mask == 'circle':
            ind = (np.sqrt((xind - FRONTLOC[irec, 0])**2 + (yind - FRONTLOC[irec, 1])**2) <= r1)
        elif mask == 'circ_segment':
            frontang = np.arctan2(meanvec_old[irec].imag, meanvec_old[irec].real)
            ind = generate_indmat(xind, yind, FRONTLOC[irec, 0], FRONTLOC[irec, 1],
                                  r1, frontang, REC_X_INTER, REC_Y_INTER)

        histog, bins = np.histogram(dat2_inter[ind], bins=10000)
        cs = (1 - np.cumsum(histog)/np.sum(histog))**4
        rr = np.random.rand(n_fil)

        ind_res = np.array([np.where(cs < rtemp)[0][0] for rtemp in rr])
        vals = ((bins[1:] + bins[:-1])/2)[ind_res]

        D_vals = np.abs(vals[:, None] - dat2_inter[ind][None, :])
        xfil = np.hstack((xfil, POS[0][ind][D_vals.argmin(axis=1)]))
        yfil = np.hstack((yfil, POS[1][ind][D_vals.argmin(axis=1)]))
        allinds = allinds + np.ravel(ind)
    return xfil, yfil, allinds

def distance_to_exp(firstpos, pos_eval, a_ell, b_ell, Xint, Yint, v1int, v2int, mode='target_area'):
    """
    evaluates if pos_eval is within the correct target area/Voronoi cell
    """
    # mode can be 'voronoi' or 'target_area'
    goal_loc = np.zeros((nr_of_rec, 2))
    for kk in range(nr_of_rec):
        D_bundles = dist.cdist([firstpos[kk, :]],
                               [[x, y] for x, y in zip(Xint, Yint)], metric='euclid')
        goal_loc[kk, 0] = Xint[D_bundles.argmin(axis=1)]
        goal_loc[kk, 1] = Yint[D_bundles.argmin(axis=1)]

    if nr_of_rec == 1 or not include_equator:
        if receptor == 1:
            goal_loc += -v2int[None, :]
        elif receptor == 2:
            goal_loc += v1int[None, :] - v2int[None, :]
        elif receptor == 3:
            if r3r4swap:
                goal_loc += v1int[None, :]
            else:
                goal_loc += 2*v1int[None, :] - 1*v2int[None, :]
        elif receptor == 4:
            if r3r4swap:
                goal_loc += v1int[None, :] + v2int[None, :]
            else:
                goal_loc += v1int[None, :]
        elif receptor == 5:
            goal_loc += v2int[None, :]
        elif receptor == 6:
            goal_loc += -v1int[None, :] + v2int[None, :]
    elif nr_of_rec == 9 and include_equator:
        if receptor == 1:
            goal_loc[np.array([0, 1, 5])] += -v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += v1int[None, :] - v2int[None, :]
        elif receptor == 2:
            goal_loc[np.array([0, 1, 5])] += v1int[None, :] - v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += - v2int[None, :]
        elif receptor == 3:
            goal_loc[np.array([0, 1, 5])] += 2*v1int[None, :] - 1*v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :] - 1*v2int[None, :]
        elif receptor == 4:
            goal_loc[np.array([0, 1, 5])] += v1int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :]
        elif receptor == 5:
            goal_loc[np.array([0, 1, 5])] += v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :] + v2int[None, :]
        elif receptor == 6:
            goal_loc[np.array([0, 1, 5])] += -v1int[None, :] + v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += v2int[None, :]

    if include_equator:
        if receptor == 1:
            ind = np.array([2, 3, 4, 6, 7, 8])
        elif receptor == 2:
            ind = np.array([1, 2, 3, 4, 7, 8])
        elif receptor == 3:
            ind = np.array([0, 1, 3, 4, 5, 8])
        elif receptor == 4:
            ind = np.array([1, 3, 4, 5, 7, 8])
        elif receptor == 5:
            ind = np.array([1, 2, 3, 4, 7, 8])
        elif receptor == 6:
            ind = np.array([3, 4, 7, 8])

    if mode == 'target_area':
        correct1 = (pos_eval[:, 0] - goal_loc[:, 0] + 0.333*v1int[0])**2\
        + (pos_eval[:, 1] - goal_loc[:, 1] + 0.333*v1int[1])**2 < 1.2*b_ell**2
        if include_equator:
            correct1[ind] = (pos_eval[ind, 0] - goal_loc[ind, 0] - 0.333*v1int[0])**2\
            + (pos_eval[ind, 1] - goal_loc[ind, 1] - 0.333*v1int[1])**2 < 1.2*b_ell**2
        correct2 = np.sqrt((pos_eval[:, 0] - goal_loc[:, 0])**2/a_ell**2
                           + (pos_eval[:, 1] - goal_loc[:, 1])**2/b_ell**2) <= 1
        correct = np.logical_or(correct1, correct2)
    elif mode == 'voronoi':
        closest_bundle = np.zeros((len(goal_loc), 2))
        for kk in range(len(goal_loc)):
            Xs_ell, Ys_ell = calc_closest_point_on_ellipse(a_ell, b_ell, pos_eval[None, kk, :]
                                                           - np.array([[x, y]
                                                                       for x,y in zip(Xint, Yint)])[:, None, :])
            Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*b_ell, 1.2*b_ell, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint - 0.333*v1int[0], Yint - 0.333*v1int[1])])[:, None, :])
            if include_equator:
                if kk in ind:
                    Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*b_ell, 1.2*b_ell, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint + 0.333*v1int[0], Yint + 0.333*v1int[1])])[:, None, :])
                D_bundles_ell = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0])], metric='euclid')
                D_bundles_circ = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint - 0.333*v1int[0] + Xs_circ[:, 0], Yint - 0.333*v1int[1] + Ys_circ[:, 0])], metric='euclid')
            else:
                D_bundles_ell = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0])], metric='euclid')
                D_bundles_circ = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint - 0.333*v1int[0] + Xs_circ[:, 0], Yint - 0.333*v1int[1] + Ys_circ[:, 0])], metric='euclid')
            min_ell = D_bundles_ell.argmin(axis=1)
            min_circ = D_bundles_circ.argmin(axis=1)
            if D_bundles_circ.min(axis=1) < D_bundles_ell.min(axis=1):
                closest_bundle[kk, 0] = Xint[min_circ]
                closest_bundle[kk, 1] = Yint[min_circ]
            else:
                closest_bundle[kk, 0] = Xint[min_ell]
                closest_bundle[kk, 1] = Yint[min_ell]
        correct = np.isclose(closest_bundle, goal_loc).all(axis=1)
    return correct

def run_gc(c_st, sl_st, startang, A_magnet, circ_width, radius, c_magnet = 35, sl_magnet = 0.4):
    """
    this is the core method, simulating the growing axon(s)
    c_st  ... central point of the stiffness time course
    sl_st ... slope of the stiffness time course
    startang ... initial extension angle of the axon(s)
    A_magnet ... adhesion strength of the L-cells
    circ_width ... angular width of the 'flashlight' for filopodial exploration
    radius ... radius of the 'flashlight' for filopodial exploration
    c_magnet  ... central point of the adhesion time course
    sl_magnet ... slope of the adhesion time course
    """
    print(receptor)
    irep = 0
    cum_error_35, cum_error_40 = 0, 0
    cum_error_35_v, cum_error_40_v = 0, 0
    len_at_times = []
    len_avg = []
    # outside_fil_all = []
    xind, yind = np.meshgrid(np.arange(2*Nx-1), np.arange(2*Ny-1))
    pval_inter = np.zeros((nr_of_rec, 4))

    while irep < nrep:
        TIME = np.array([])
        meanvec_old = np.ones(nr_of_rec) * sp * 2/3*radii[receptor-1]*np.sin(np.pi/180 * circ_width)/(np.pi/180 * circ_width) * np.exp(1j*startang) #* radii[receptor-1] * np.exp(1j*startang) * sp1
        for hh, hour in enumerate(hours[1:]):
            hh += 1
            if hour == 45:
                v1old, v2old = vec_opt[-1, :2], vec_opt[-1, 2:]
                v1new, v2new = vec_opt[-1, :2], vec_opt[-1, 2:]
            elif 26 < hour < 45:
                v1old, v2old = vec_opt[hh-2, :2], vec_opt[hh-2, 2:]
                v1new, v2new = vec_opt[hh-1, :2], vec_opt[hh-1, 2:]
            else:
                v1old, v2old = vec_opt[0, :2] - (vec_opt[1, :2] - vec_opt[0, :2]), vec_opt[0, 2:] - (vec_opt[1, 2:] - vec_opt[0, 2:])
                v1new, v2new = vec_opt[0, :2], vec_opt[0, 2:]

            lcellsize_int = np.linspace(LCELL_SIZE[hh-1], LCELL_SIZE[hh], nsteps)
            v1inter, v2inter = np.linspace(v1old, v1new, nsteps), np.linspace(v2old, v2new, nsteps)

            if hour == 45:
                FRO_INT_X = np.linspace(FRONTLOCS_ALL[:, -1, 0] - HEELLOCS_AVG[:, -1, 0], FRONTLOCS_ALL[:, -1, 0] - HEELLOCS_AVG[:, -1, 0], nsteps)
                # FRO_INT_Y = np.linspace(FRONTLOCS_ALL[:, -1, 1] - HEELLOCS_AVG[:, -1, 1], FRONTLOCS_ALL[:, -1, 1] - HEELLOCS_AVG[:, -1, 1], nsteps)
            elif 26 < hour < 45:
                FRO_INT_X = np.linspace(FRONTLOCS_ALL[:, hh-2, 0] - HEELLOCS_AVG[:, hh-2, 0], FRONTLOCS_ALL[:, hh-1, 0] - HEELLOCS_AVG[:, hh-1, 0], nsteps)
                # FRO_INT_Y = np.linspace(FRONTLOCS_ALL[:, hh-2, 1] - HEELLOCS_AVG[:, hh-2, 1], FRONTLOCS_ALL[:, hh-1, 1] - HEELLOCS_AVG[:, hh-1, 1], nsteps)
            else:
                FRO_INT_X = np.linspace(HEELLOCS_AVG[:, 0, 0], FRONTLOCS_ALL[:, 0, 0], nsteps) - HEELLOCS_AVG[:, 0, 0]
                # FRO_INT_Y = np.linspace(HEELLOCS_AVG[:, 0, 1], FRONTLOCS_ALL[:, 0, 1], nsteps) - HEELLOCS_AVG[:, 0, 1]

            center_loc_x = (n1*v1new[0] + n2*v2new[0]).flatten()
            # center_loc_y = (n1*v1new[1] + n2*v2new[1]).flatten()

            # preparing the interpolated arrays
            Xshift, Xshift_old = 1024, 1024
            Yshift, Yshift_old = 512, 512
            Xstart = np.linspace(Xshift_old, Xshift, nsteps)
            Ystart = np.linspace(Yshift_old, Yshift, nsteps)

            if hh == 1:
                recx, recy, _, _ = create_starting_grid2(np.array([Xstart[0]]), np.array([Ystart[0]]), hour)
                if nr_of_rec == 1:
                    FRONTLOC = np.vstack((recx[receptor-1], recy[receptor-1])).T
                else:
                    FRONTLOC = np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2 + 0.5)[:, None] * v2old + (np.array([recx[receptor-1], recy[receptor-1]]).copy()).T
                    FRONTLOC = np.vstack((FRONTLOC, np.array([(ff - v1old) for ff in FRONTLOC if ff[0] > FRONTLOC[:, 0].min()])))

            print('')
            TIME_NOW = np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=False) if hour < 40 else np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=True)
            dt = np.diff(TIME_NOW)[0]
            TIME = np.hstack((TIME, TIME_NOW))
            for tind, tt in enumerate(TIME_NOW):
                print('\r'+str(hour)+'hAPF: |' + (tind+1)*'=' + (nsteps-tind-1)*' ' + '|', end='')
                if hh == 2:
                    recx, recy, _, _ = create_starting_grid2(np.array([Xstart[tind]]), np.array([Ystart[tind]]), hour, tind, nsteps)
                else:
                    recx, recy, _, _ = create_starting_grid2(np.array([Xstart[tind]]), np.array([Ystart[tind]]), hour)

                if include_equator:
                    heels = np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2 + 0.5)[:, None] * v2inter[tind] + (np.array([recx[receptor-1], recy[receptor-1]]).copy()).T
                    heels = np.vstack((heels, np.array([(ff - v1inter[tind]) for ff in heels if ff[0] > heels[:, 0].min()])))
                else:
                    heels = (np.array([recx[receptor-1], recy[receptor-1]]).copy()).T
                HEELLOC_INT_X = heels[:, 0]
                HEELLOC_INT_Y = heels[:, 1]

                if create_movie:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 20]})
                    ax[0].barh([1], [hours[-1]], color='w', edgecolor='k')
                    ax[0].barh([1], [tt], color='k')
                    ax[0].set_title('Developmental time (hAPF)')
                    ax[0].tick_params(axis='y', which='both', right=False,
                                      left=False, labelleft=False)
                    ax[0].set_xticks(np.arange(20, hours[-1]+1, 5))
                    for pos in ['right', 'top', 'bottom', 'left']:
                        ax[0].spines[pos].set_visible(False)
                    ax[0].set_xlim([19.9, hours[-1]+0.1])

                if nr_of_rec > 1:
                    exp_fronts_x = FRONTLOCS_ALL[receptor-1, :, 0][None, :] + (np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2+0.5)*v2inter[tind, 0])[:, None]
                    exp_fronts_x = np.vstack((exp_fronts_x, np.array([ff - v1inter[tind, 0] for ff in exp_fronts_x[1:]])))
                    #exp_fronts_x += np.array([512]) #+ v1inter[tind, 0]
                    exp_fronts_y = FRONTLOCS_ALL[receptor-1, :, 1][None, :] + (np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2+0.5)*v2inter[tind, 1])[:, None]
                    exp_fronts_y = np.vstack((exp_fronts_y, np.array([ff - v1inter[tind, 1] for ff in exp_fronts_y[1:]])))

                    heel_locs_x = HEELLOCS_AVG[receptor-1, :, 0][None, :] + (np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2+0.5)*v2inter[tind, 0])[:, None]
                    heel_locs_x = np.vstack((heel_locs_x, np.array([ff - v1inter[tind, 0] for ff in heel_locs_x[1:]])))
                    heel_locs_y = HEELLOCS_AVG[receptor-1, :, 1][None, :] + (np.arange(-((nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2+0.5)*v2inter[tind, 1])[:, None]
                    heel_locs_y = np.vstack((heel_locs_y, np.array([ff - v1inter[tind, 1] for ff in heel_locs_y[1:]])))
                else:
                    exp_fronts_x = FRONTLOCS_ALL[receptor-1, :, 0]
                    exp_fronts_y = FRONTLOCS_ALL[receptor-1, :, 1]
                    heel_locs_x = HEELLOCS_AVG[receptor-1, :, 0]
                    heel_locs_y = HEELLOCS_AVG[receptor-1, :, 1]

                center_loc_x_int = (Xstart[tind] + n1*v1inter[tind, 0] + n2*v2inter[tind, 0]).flatten()
                center_loc_y_int = (Ystart[tind] + n1*v1inter[tind, 1] + n2*v2inter[tind, 1]).flatten()
                ind = np.ones(np.shape(center_loc_x), dtype='bool')#(center_loc_x_int > 0)*(center_loc_x_int < 2048)*(center_loc_y_int > 0)*(center_loc_y_int < 1024)
                Xint, Yint = center_loc_x_int[ind], center_loc_y_int[ind]

                if hh == 2:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour, tind, nsteps)
                else:
                    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour)

                FRO_X2_int = REC_X_INTER + FRO_INT_X[tind, None].T
                # FRO_Y2_int = REC_Y_INTER + FRO_INT_Y[tind, None].T

                if include_equator:
                    if tind == 0 and hh == 1:
                        if receptor == 6:
                            border = np.mean(XC)
                        else:
                            border = np.mean(XC) - 20
                    else:
                        currmat = np.array([v1inter[tind], v2inter[tind]]).T
                        prevmat = np.array([v1inter[tind-1], v2inter[tind-1]]).T
                        border = ((currmat@np.linalg.inv(prevmat)@((border-Xstart[tind-1])*np.ones(2)).T).T)[0] + Xstart[tind]

                    Dmat = dist.cdist([[x, y] for x, y in zip(HEELLOC_INT_X, HEELLOC_INT_Y)], [[x, y] for x, y in zip(XC, YC)], metric='euclid')
                    ind2 = Dmat.argmin(axis=1)
                    if tind == 0 and hh == 1:
                        FRONTLOC[:, 0] = np.where(XC[ind2] >= border, FRONTLOC[:, 0] - 2*(FRONTLOC[:, 0] - XC[ind2]), FRONTLOC[:, 0])
                        FRONTLOCS_TIME_temp = FRONTLOC.copy()
                        FRONTLOCS_TIME = FRONTLOCS_TIME_temp.copy()
                        meanvec_old[XC[ind2] >= border] = meanvec_old[XC[ind2] >= border] - 2*(meanvec_old[XC[ind2] >= border].real)
                    exp_fronts_x[XC[ind2] >= border, :] = exp_fronts_x[XC[ind2] >= border, :] - 2*(exp_fronts_x[XC[ind2] >= border, :] - XC[ind2][XC[ind2] >= border][:, None])
                    heel_locs_x[XC[ind2] >= border, :] = heel_locs_x[XC[ind2] >= border, :] - 2*(heel_locs_x[XC[ind2] >= border, :] - XC[ind2][XC[ind2] >= border][:, None])
                    HEELLOC_INT_X = np.where(XC[ind2] >= border, HEELLOC_INT_X - 2*(HEELLOC_INT_X - XC[ind2]), HEELLOC_INT_X)
                    REC_X_INTER[:, XC >= border] = REC_X_INTER[:, XC >= border] - 2*(REC_X_INTER[:, XC >= border] - XC[XC >= border][None, :])
                    FRO_X2_int[:, XC >= border] = REC_X_INTER[:, XC >= border] - FRO_INT_X[tind, None].T
                else:
                    if tind == 0 and hh == 1:
                        FRONTLOCS_TIME_temp = FRONTLOC.copy()
                        FRONTLOCS_TIME = FRONTLOCS_TIME_temp.copy()

                xmin2, xmax2, ymin2, ymax2 = 0, 2*Nx, 0, 2*Ny
                POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx-1), np.linspace(ymin2, ymax2, 2*Ny-1))
                POS = np.array(POS)
                if run_till_p45 and hour == 45:
                    if include_equator:
                        dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(40)+'hAPF_'+str(14)+'.npy')
                        # dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'_withoutLcells.npy')
                    else:
                        # dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(40)+'hAPF_'+str(14)+'_noeq_withoutLcells.npy')
                        dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(40)+'hAPF_'+str(14)+'_noeq.npy')
                else:
                    if include_equator:
                        dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'.npy')
                        # dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'_withoutLcells.npy')
                    else:
                        # dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'_noeq_withoutLcells.npy')
                        dat2_inter = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'_noeq.npy')

                # create index array for the circle
                xfil, yfil, allinds = sample_roi(dat2_inter, FRONTLOC, meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER, radius)

                # # how many of the filopodia are in the target area
                # if nr_of_rec == 1 and np.ndim(FRONTLOCS_TIME) == 3:
                #     a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                #     outside_fil = distance_to_exp(FRONTLOCS_TIME[0, :, :], np.c_[xfil, yfil], a, b, Xint, Yint, v2inter[tind], v1inter[tind])
                #     print(np.sum(outside_fil)/len(xfil))
                #     outside_fil_all.append(outside_fil)

                xfil= np.reshape(xfil, (nr_of_rec, -1))
                yfil= np.reshape(yfil, (nr_of_rec, -1))
                angl = np.arctan2(yfil-FRONTLOC[:, 1][:, None], xfil-FRONTLOC[:, 0][:, None])
                radi = np.sqrt((yfil-FRONTLOC[:, 1][:, None])**2 + (xfil-FRONTLOC[:, 0][:, None])**2)

                goal_loc = np.zeros((2, nr_of_rec))
                for kk in range(nr_of_rec):
                    if include_equator:
                        D_bundles = dist.cdist([FRONTLOC[kk, :]], [[x, y] for x, y in zip(Xint + np.where(Xint >= border, lcellsize_int[tind, 0]/3*CONVFAC, -lcellsize_int[tind, 0]/3*CONVFAC), Yint)], metric='euclid')
                        goal_loc[0, kk] = Xint[D_bundles.argmin(axis=1)] + np.where(Xint[D_bundles.argmin(axis=1)] >= border, lcellsize_int[tind, 0]/3*CONVFAC, - lcellsize_int[tind, 0]/3*CONVFAC)
                    else:
                        D_bundles = dist.cdist([FRONTLOC[kk, :]], [[x, y] for x, y in zip(Xint - lcellsize_int[tind, 0]/3*CONVFAC, Yint)], metric='euclid')
                        goal_loc[0, kk] = Xint[D_bundles.argmin(axis=1)] - lcellsize_int[tind, 0]/3*CONVFAC
                    goal_loc[1, kk] = Yint[D_bundles.argmin(axis=1)]

                if create_movie:
                    if not movie_pure:
                        len1 = 100
                        viridis = plt.get_cmap('viridis', len1)
                        rgba_vir = viridis.colors
                        norm = plt.Normalize(np.min(dat2_inter[200:800, 1300:1600]), np.max(dat2_inter[200:800, 200:1600]))
                        cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                                 'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                                 'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}
                        newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
                        ax[1].imshow(dat2_inter, origin='lower', cmap=newcmp, norm=norm)

                        ax[1].imshow(np.reshape(allinds, (1023, 2047)), origin='lower', alpha=0.2)
                        if nr_of_rec > 1:
                            for irec in range(nr_of_rec):
                                ax[1].scatter(*FRONTLOC[irec, :], c='r')
                                ax[1].scatter(xfil, yfil, s=5, c='w')
                                if np.ndim(FRONTLOCS_TIME) == 3:
                                    ax[1].plot(FRONTLOCS_TIME[:, irec, 0], FRONTLOCS_TIME[:, irec, 1], 'r-', lw=2)
                                # ax[1].scatter(HEELLOC_INT_X[irec, tind], HEELLOC_INT_Y[irec, tind], c='b', zorder=5)
                                if np.ndim(FRONTLOCS_TIME) == 3:
                                    ax[1].scatter(FRONTLOCS_TIME[0, :, 0], FRONTLOCS_TIME[0, :, 1], c='b', zorder=5)
                                else:
                                    ax[1].scatter(FRONTLOC[:, 0], FRONTLOC[:, 1], c='b', zorder=5)
                        else:
                            ax[1].scatter(*FRONTLOC.T, c='r')
                            ax[1].scatter(xfil, yfil, s=5, c='w')
                            if np.ndim(FRONTLOCS_TIME) == 3:
                                ax[1].plot(FRONTLOCS_TIME[:, 0, 0], FRONTLOCS_TIME[:, 0, 1], 'r-', lw=2)
                            ax[1].scatter(recx[receptor-1], recy[receptor-1], c='b', zorder=5)

                        if not include_equator:
                            for xx, yy in zip(Xint, Yint):
                                circle_temp = Point(xx, yy).buffer(1)
                                ellipse = shapely.affinity.scale(circle_temp, 1.0*lcellsize_int[tind, 0]*CONVFAC/2, 1.0*lcellsize_int[tind, 1]*CONVFAC/2)  # type(ellipse)=polygon
                                if hour == 26:
                                    circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer((0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC)
                                else:
                                    circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer(0.6*lcellsize_int[tind, 1]*CONVFAC)
                                uni = circle.union(ellipse)
                                ax[1].add_patch(descartes.PolygonPatch(uni, fc='g', ec='g', alpha=0.2))
                        else:
                            for xx, yy in zip(Xint, Yint):
                                circle_temp = Point(xx, yy).buffer(1)
                                ellipse = shapely.affinity.scale(circle_temp, 1.0*lcellsize_int[tind, 0]*CONVFAC/2, 1.0*lcellsize_int[tind, 1]*CONVFAC/2)  # type(ellipse)=polygon
                                if xx >= border:
                                    if hour == 26:
                                        circle = Point(xx + 0.333*v2inter[tind, 0], yy + 0.333*v2inter[tind, 1]).buffer((0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC)
                                    else:
                                        circle = Point(xx + 0.333*v2inter[tind, 0], yy + 0.333*v2inter[tind, 1]).buffer(0.6*lcellsize_int[tind, 1]*CONVFAC)
                                else:
                                    if hour == 26:
                                        circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer((0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC)
                                    else:
                                        circle = Point(xx - 0.333*v2inter[tind, 0], yy - 0.333*v2inter[tind, 1]).buffer(0.6*lcellsize_int[tind, 1]*CONVFAC)
                                uni = circle.union(ellipse)
                                ax[1].add_patch(descartes.PolygonPatch(uni, fc='g', ec='g', alpha=0.2))
                        for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                            ax[1].add_patch(Circle((xx, yy), 0.5*CONVFAC, ls = '--', edgecolor='k', facecolor='none'))

                        if nr_of_rec == 9:
                            if receptor >= 5 :
                                ax[1].axis([450, 1550, 290, 840])
                            else:
                                ax[1].axis([450, 1550, 210, 750])
                        elif nr_of_rec == 1:
                            if receptor == 3 or (receptor == 4 and r3r4swap):
                                ax[1].axis([900, 1500, 280, 680])
                            elif receptor in (1, 2):
                                ax[1].axis([700, 1300, 300, 700])
                            else:
                                ax[1].axis([700, 1300, 350, 750])
                        remove_ticks_and_box(ax[1])
                        fig.tight_layout()
                        if show_insets:
                            inset_axes(ax[1], width='20%', height='20%', loc=2)
                            plt.plot(np.arange(20, hours[-1]+1, 0.01), 1-tanh_cust(np.arange(20, hours[-1]+1, 0.01), c_st, sl_st), 'k-', lw=3)
                            plt.scatter(tt, 1-tanh_cust(tt, c_st, sl_st), c='r', s=40, zorder=5)
                            plt.ylim([-0.1, 1.1])
                            plt.xlabel('Time (hour)', color='w')
                            plt.ylabel('Stiffness', color='w')
                            plt.gca().yaxis.set_label_position("right")
                            plt.gca().yaxis.tick_right()
                            plt.gca().tick_params(axis='x', colors='w')
                            plt.gca().tick_params(axis='y', colors='w')

                            inset_axes(ax[1], width='20%', height='20%', loc=1)
                            plt.plot(np.arange(20, hours[-1]+1, 0.01), tanh_cust(np.arange(20, hours[-1]+1, 0.01), c_magnet, sl_magnet), 'k-', lw=3)
                            plt.scatter(tt, tanh_cust(tt, c_magnet, sl_magnet), c='r', s=40, zorder=5)
                            plt.ylim([-0.1, 1.1])
                            plt.xlabel('Time (hour)', color='w')
                            plt.ylabel('L-cell attraction', color='w')
                            plt.gca().tick_params(axis='x', colors='w')
                            plt.gca().tick_params(axis='y', colors='w')

                        filename = str('%04d' % (100*hour+tind) + '.png')
                        plt.savefig(filename, dpi=100)
                        plt.close(fig)
                    else:
                        if include_equator:
                            eq_max = np.max(dat2_inter[200:800, 200:1600])
                            flat_min = np.min(dat2_inter[200:800, 1300:1600])
                            len1 = 100
                            viridis = plt.get_cmap('viridis', len1)
                            rgba_vir = viridis.colors
                            norm = plt.Normalize(flat_min, eq_max)

                            cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                                     'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                                     'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}

                            newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
                            ax[1].imshow(dat2_inter, origin='lower', cmap=newcmp, norm=norm)
                            for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                                ax[1].add_patch(Circle((xx, yy), 0.5*CONVFAC, ls = '--', edgecolor='k', facecolor='none'))

                            remove_ticks_and_box(ax[1])
                        else:
                            dat2_inter_eq = np.load('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hours[hh])+'hAPF_'+str(tind)+'.npy')
                            eq_max = np.max(dat2_inter_eq[200:800, 200:1600])
                            flat_min = np.min(dat2_inter[200:800, 1300:1600])

                            len1 = 100
                            viridis = plt.get_cmap('viridis', len1)
                            rgba_vir = viridis.colors

                            norm = plt.Normalize(flat_min, np.max(dat2_inter[200:800, 1300:1600])) #eq_max)

                            cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                                      'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                                      'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}

                            newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
                            ax[1].imshow(dat2_inter, origin='lower', cmap=newcmp, norm=norm)
                            for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                                ax[1].add_patch(Circle((xx, yy), 0.5*CONVFAC, edgecolor='yellow', facecolor='yellow'))

                        ax[1].axis([500, 1500, 300, 700])

                        remove_ticks_and_box(ax[1])
                        fig.tight_layout()
                        filename = str('%04d' % (100*hour+tind) + '.png')
                        plt.savefig(filename, dpi=100)
                        plt.close(fig)

                if nr_of_rec > 1:
                    axon_len = np.sqrt((FRONTLOC[:, 0] - heel_locs_x[:, 0])**2 + (FRONTLOC[:, 1] - heel_locs_y[:, 0])**2)
                else:
                    axon_len = np.sqrt((FRONTLOC[0, 0] - heel_locs_x[0])**2 + (FRONTLOC[0, 1] - heel_locs_y[0])**2)

                stiff = 1 - tanh_cust(tt, c_st, sl_st) #sl_magnet)
                len_avg.append(np.mean(axon_len))
                if tt in [26, 30, 35, 40]:
                    len_at_times.append(np.mean(axon_len))
                if np.ndim(FRONTLOCS_TIME) == 3:
                    norm = np.shape(FRONTLOCS_TIME)[0]-1
                else:
                    norm = 1
                stiffpart = stiff*meanvec_old/norm
                if novalleydyn:
                    filopart = (1-stiff)*sp*np.abs(np.mean(radi*np.exp(1j*angl), axis=1))*np.exp(1j*np.angle(stiffpart))
                else:
                    filopart = (1-stiff)*sp*np.mean(radi*np.exp(1j*angl), axis=1)
                magnet_weight = tanh_cust(tt, c_magnet, sl_magnet)

                # check whether front location is in one of the L-cell regions
                inside_goal_area = np.zeros(len(FRONTLOC), dtype='bool')
                if np.ndim(FRONTLOCS_TIME) == 3:
                    a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                    v1int = v2inter[tind].copy()
                    # v2int = v1inter[tind].copy()
                    for kk in range(nr_of_rec):
                        for xx, yy in zip(Xint, Yint):
                            check1 = (np.sqrt((FRONTLOC[kk, 0] - xx + 0.333*v1int[0])**2 + (FRONTLOC[kk, 1] - yy + 0.333*v1int[1])**2) < (1.2*b + CONVFAC/4))
                            if include_equator and xx >= border:
                                check1 = (np.sqrt((FRONTLOC[kk, 0] - xx - 0.333*v1int[0])**2 + (FRONTLOC[kk, 1] - yy - 0.333*v1int[1])**2) < (1.2*b + CONVFAC/4))
                            check2 = ((FRONTLOC[kk, 0] - xx)**2/(a+CONVFAC/4)**2 + (FRONTLOC[kk, 1] - yy)**2/(b+CONVFAC/4)**2 <= 1**2) #0.8
                            if check1 or check2:
                                inside_goal_area[kk] = True
                                if include_equator and xx >= border:
                                    goal_loc[:, kk] = np.array([xx + lcellsize_int[tind, 0]/3*CONVFAC, yy])
                                else:
                                    goal_loc[:, kk] = np.array([xx  - lcellsize_int[tind, 0]/3*CONVFAC, yy])

                magnetpart = magnet_weight * A_magnet * np.exp(1j*np.arctan2(goal_loc[1]-FRONTLOC[:, 1], goal_loc[0]-FRONTLOC[:, 0])) #-lcellsize_int[tind, 0]/2*CONVFAC
                meanvec = np.where(inside_goal_area, (1-magnet_weight)*(stiffpart + filopart) + magnetpart, stiffpart + filopart)

                # check if axons are close to magnet goal location, if so: stop movement
                meanvec = np.where(np.sqrt((goal_loc[0]-FRONTLOC[:, 0])**2 + (goal_loc[1]-FRONTLOC[:, 1])**2) <= 5, 0+0*1j, meanvec)
                change = np.array([meanvec.real, meanvec.imag]).T

                if tt == 30:
                    angle_at_p30 = np.angle(np.mean(radi*np.exp(1j*angl)))
                if tind > 0:
                    currmat = np.array([v1inter[tind], v2inter[tind]]).T
                    prevmat = np.array([v1inter[tind-1], v2inter[tind-1]]).T
                    change = (currmat@np.linalg.inv(prevmat)@change.T).T
                FRONTLOC += change * dt
                FRONTLOCS_TIME_temp = np.vstack((FRONTLOCS_TIME_temp, FRONTLOC))
                FRONTLOCS_TIME = np.reshape(FRONTLOCS_TIME_temp.copy(), (-1, nr_of_rec, 2))
                meanvec_old = (FRONTLOCS_TIME[-1, :, 0] - FRONTLOCS_TIME[0, :, 0] + 1j*(FRONTLOCS_TIME[-1, :, 1] - FRONTLOCS_TIME[0, :, 1]))/dt
                if tind > 0:
                    if nr_of_rec > 1:
                        FRONTLOCS_TIME[:, :, 0] = FRONTLOCS_TIME[:, :, 0] + (HEELLOC_INT_X - FRONTLOCS_TIME[0, :, 0])
                        FRONTLOCS_TIME[:, :, 1] = FRONTLOCS_TIME[:, :, 1] + (HEELLOC_INT_Y - FRONTLOCS_TIME[0, :, 1])
                    else:
                        FRONTLOCS_TIME[:, :, 0] = FRONTLOCS_TIME[:, :, 0] + (recx[receptor-1] - FRONTLOCS_TIME[0, :, 0])
                        FRONTLOCS_TIME[:, :, 1] = FRONTLOCS_TIME[:, :, 1] + (recy[receptor-1] - FRONTLOCS_TIME[0, :, 1])
                    FRONTLOCS_TIME_temp = np.reshape(FRONTLOCS_TIME, (-1, 2))
                    FRONTLOC = FRONTLOCS_TIME[-1, :, :]
                if hour == 40 and tind == 3: #hour == 35 and tind == nsteps-1:
                    a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                    cum_error_35 += np.sum(distance_to_exp(FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2inter[tind], v1inter[tind], mode='target_area'))
                    cum_error_35_v += np.sum(distance_to_exp(FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2inter[tind], v1inter[tind], mode='voronoi'))
                elif hour == 40 and tind == nsteps-1:
                    a, b = lcellsize_int[tind, 0]/2*CONVFAC, lcellsize_int[tind, 1]/2*CONVFAC
                    cum_error_40 += np.sum(distance_to_exp(FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2inter[tind], v1inter[tind], mode='target_area'))
                    cum_error_40_v += np.sum(distance_to_exp(FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2inter[tind], v1inter[tind], mode='voronoi'))
        irep += 1

    if investigate == 'paths':
        return FRONTLOCS_TIME, REC_X_INTER, REC_Y_INTER, Xint, Yint
    return cum_error_35/nrep, cum_error_40/nrep, cum_error_35_v/nrep, cum_error_40_v/nrep, np.array(len_at_times), np.array(len_avg), pval_inter, angle_at_p30# , outside_fil_all

if __name__ == "__main__":
    mpl.use('Qt5Agg')
    create_movie = True # whether or not to create a movie
    movie_pure = False # creates a movie without the axon, only the density landscape, only used if create_movie == True
    show_insets = False # whether or not to show the insets with the time course of stiffness and adhesion
    if create_movie:
        mpl.use('Agg') # do not show figures while creating movie

    include_equator = False # whether or not to simulate axons at the equator of the lamina
    r3r4swap = False # if True, swaps the R3-R4 identities
    run_till_p45 = False # if True, runs the simulations longer until 45hAPF

    receptor = 3 # what receptor to simulate, an integer from 1 to 6

    A_magnet = 10 # default value for the adhesion strength
    CONVFAC = 25.2 # scaling of the simulation, in pixels per um
    nsteps = 15 # number of time steps between two experimental time points (the experimental time points are 26, 30, 35, 40 hAPF)
    radii = np.array([4.41, 3.60, 5.15, 1.0 * 3.58, 4.48, 4.74])*CONVFAC # radius of the 'flashlight'
    circ_width_all = np.array([[76.93677250002409, 66.1581562071056, 58.64359788352946, 68.19374152821266],
                               [67.10341096706019, 63.6030367287868, 64.58480307212197, 64.68382969250712],
                               [59.99951401507621, 49.70632397768759, 59.76089748808765, 62.63689465660343],
                               [65.18759340181808, 64.6663385598332, 54.128563634672744, 58.3338061658033],
                               [73.7705462792746, 73.75319815476855, 70.71260029847282, 67.54024198457094],
                               [82.29011450458808, 70.63559641125377, 69.51033837975177, 58.19852457593135]])
    circ_width = circ_width_all.mean(axis=1)[receptor-1] # angular width of the 'flashlight'

    speeds = np.array([0.093, 0.053, 0.148, 0.09, 0.052, 0.077]) # speed scaling for the growth of the axon
    sp = speeds[receptor-1]

    r = radii[receptor-1]
    n_fil = 10 # number of filopodia in each time step

    region = 'circ_segment' # can be 'circ_segment' or 'circle'
    novalleydyn = False # turns off the density-sensing part
    investigate = 'magnet_timing' # can be 'angle', 'magnet_strength', 'magnet_timing', 'stiffness', or 'paths', only used if create_movie==False
    if include_equator:
        nr_of_rec = 9 # simulate 9 axons around the equator
    else:
        nr_of_rec = 1 # simulate 1 axon without the equator

    hours = np.array([20, 26, 30, 35, 40]) # experimental time points, added 20hAPF for which we extrapolate the simulation parameters
    LCELL_SIZE = np.array([[3.96, 2.23], [3.96, 2.23], [3.9, 2.28], [4.71, 2.61], [4.57, 2.63]]) # experimental dimensions of the L cells

    if run_till_p45:
        hours = np.append(hours, 45)
        LCELL_SIZE = np.vstack((LCELL_SIZE, np.array([[4.57, 2.63]])))

    # load the two vectors spanning the grid of bundles for the density landscape,
    # as well as the experimental front locations
    vec_opt, FRONTLOCS_ALL, HEELLOCS_AVG = load_grid_heels_fronts()
    HEELLOCS_AVG += np.array([1024, 512])[None, None, :]
    FRONTLOCS_ALL = HEELLOCS_AVG + FRONTLOCS_ALL*25.2

    # rotate everything to have the grid perfectly horizontal
    rot_ang = []
    for hh in range(4):
        rot_ang.append(-np.arctan2(vec_opt[hh, 3], vec_opt[hh, 2]))
        # FRONTLOCS_ALL[:, hh, 0], FRONTLOCS_ALL[:, hh, 1] = rotate_coord(FRONTLOCS_ALL[:, hh, 0], FRONTLOCS_ALL[:, hh, 1], rot_ang[-1])
        # HEELLOCS_AVG[:, hh, 0], HEELLOCS_AVG[:, hh, 1] = rotate_coord(HEELLOCS_AVG[:, hh, 0], HEELLOCS_AVG[:, hh, 1], rot_ang[-1])
        vec_opt[hh, ::2], vec_opt[hh, 1::2] = rotate_coord(vec_opt[hh, ::2], vec_opt[hh, 1::2], rot_ang[-1])

    # create the factors for the linear combinations of the grid-spanning vectors,
    # used in run_gc() to create the grid of bundles
    mini, maxi = -10, 10
    n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
    Nx, Ny = 1024, 512 # size of the image of the density landscape

    # initital angles, from experimental data
    startangs_all = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 13.26706/2, 63.2865, 135.0751667])
    # experimental standard deviations of the initial angles
    startangs_std = np.array([8.758102091, 10.27462811, 5.683716097, 7.21548511, 14.56159809, 5.942082166])

    startang = startangs_all[receptor-1]
    if r3r4swap:
        # modifications for the R3-R4 swap experiment
        if receptor == 3:
            sp = speeds[3]
            r = radii[3]
            startang = -startangs_all[3]
            circ_width = circ_width_all.mean(axis=1)[3]

        elif receptor == 4:
            sp = speeds[2]
            r = radii[2]
            startang = -startangs_all[2]
            circ_width = circ_width_all.mean(axis=1)[2]

    if create_movie or include_equator:
        nrep = 1
    else:
        nrep = 10

    if nrep > 1:
        print('------------------------------')
        print('nrep is larger than 1')
        print('------------------------------')

    if create_movie:
        cent_stiff = np.array([25]) # central point of the stiffness curve
        sl_stiff = np.array([0.45]) # slope of the stiffness curve
        c_magnet, sl_magnet = np.array([35]), np.array([0.45]) # parameters for the adhesion curve

        err_mat = np.zeros((1, 1))
        len_avg_all = []
        for ic, c_st in enumerate(cent_stiff):
            for isl, sl_st in enumerate(sl_stiff):
                print(isl)
                cumerr_35, cumerr_40, cumerr_35_v, cumerr_40_v, len_at_times, len_avg, pval_inter, angle_at_p30 = run_gc(c_st, sl_st, startang, A_magnet, circ_width, r, c_magnet, sl_magnet)
                err_mat[ic, isl] =  cumerr_35/nrep
                len_avg_all.append(len_avg/CONVFAC)
        create_heatmap(err_mat/CONVFAC, np.round(cent_stiff/CONVFAC, 1), np.round(sl_stiff, 1), cbarlabel='Error (um)', xlabel='Slope of stiffness', ylabel='Central point (um)', origin='lower')
    else:
        if investigate == 'angle':
            # calculates the sweeps for starting angle
            cent_stiff = np.array([30])
            sl_stiff = np.array([0.45])
            samp = 11
            if r3r4swap:
                if receptor == 4:
                    startangs = np.linspace(5, 25, samp) * np.pi/180
                elif receptor == 3:
                    startangs = np.linspace(-10, 10, samp) * np.pi/180
            else:
                startangs = startang + np.linspace(-45, 45, samp) * np.pi/180

            novalleydyn = False
            rep = 5
            data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(cent_stiff, sl_stiff, startang, A_magnet, circ_width, r) for startang in np.repeat(startangs, rep))
            cumerr_full_35 = np.array([d[0] for d in data])
            cumerr_full_35 = np.reshape(np.array(cumerr_full_35), (samp, rep))
            err_full_35 = np.mean(cumerr_full_35, axis=1)/nr_of_rec
            cumerr_full_35_v = np.array([d[2] for d in data])
            cumerr_full_35_v = np.reshape(np.array(cumerr_full_35_v), (samp, rep))
            err_full_35_v = np.mean(cumerr_full_35_v, axis=1)/nr_of_rec

            cumerr_full_40 = np.array([d[1] for d in data])
            cumerr_full_40 = np.reshape(np.array(cumerr_full_40), (samp, rep))
            err_full_40 = np.mean(cumerr_full_40, axis=1)/nr_of_rec
            cumerr_full_40_v = np.array([d[3] for d in data])
            cumerr_full_40_v = np.reshape(np.array(cumerr_full_40_v), (samp, rep))
            err_full_40_v = np.mean(cumerr_full_40_v, axis=1)/nr_of_rec

            if r3r4swap:
                plt.figure(figsize=(8, 7))
                plt.bar(180/np.pi * startangs, np.ones(len(err_full_35)), width=0.8, facecolor='w', edgecolor='b')
                plt.bar(180/np.pi * startangs, err_full_35, width=0.8, facecolor='b')
                plt.xlabel('Starting angle (deg)', fontsize=20)
                plt.ylabel('Mean performance', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                plt.figure(figsize=(8, 7))
                plt.bar(180/np.pi * startangs, np.ones(len(err_full_40)), width=0.8, facecolor='w', edgecolor='b')
                plt.bar(180/np.pi * startangs, err_full_40, width=0.8, facecolor='b')
                plt.xlabel('Starting angle (deg)', fontsize=20)
                plt.ylabel('Mean performance', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                plt.figure(figsize=(8, 7))
                plt.bar(180/np.pi * startangs, np.ones(len(err_full_35_v)), width=0.8, facecolor='w', edgecolor='b')
                plt.bar(180/np.pi * startangs, err_full_35_v, width=0.8, facecolor='b')
                plt.xlabel('Starting angle (deg)', fontsize=20)
                plt.ylabel('Mean performance', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                plt.figure(figsize=(8, 7))
                plt.bar(180/np.pi * startangs, np.ones(len(err_full_40_v)), width=0.8, facecolor='w', edgecolor='b')
                plt.bar(180/np.pi * startangs, err_full_40_v, width=0.8, facecolor='b')
                plt.xlabel('Starting angle (deg)', fontsize=20)
                plt.ylabel('Mean performance', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
            else:
                plt.figure()
                if startang < -np.pi/2:
                    startang += 2*np.pi
                    startangs += 2*np.pi
                plt.errorbar(180/np.pi * startangs, err_full_35, yerr=np.std(cumerr_full_35/nr_of_rec, axis=1)/np.sqrt(rep), lw=3, label='Full model')
                plt.scatter(180/np.pi * startang, err_full_35[int((samp-1)/2)], s=50, c='r', zorder=5)
                plt.errorbar(180/np.pi * startang, err_full_35[int((samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
                plt.xlabel('R' + str(receptor)+' starting angle')
                plt.ylabel('Mean performance')
                plt.ylim([-0.05, 1.05])
                # plt.legend()

                plt.figure()
                if startang < -np.pi/2:
                    startang += 2*np.pi
                    startangs += 2*np.pi
                plt.errorbar(180/np.pi * startangs, err_full_35_v, yerr=np.std(cumerr_full_35_v/nr_of_rec, axis=1)/np.sqrt(rep), lw=3, label='Full model')
                plt.scatter(180/np.pi * startang, err_full_35_v[int((samp-1)/2)], s=50, c='r', zorder=5)
                plt.errorbar(180/np.pi * startang, err_full_35_v[int((samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
                plt.xlabel('R' + str(receptor)+' starting angle')
                plt.ylabel('Mean performance')
                plt.ylim([-0.05, 1.05])
                # plt.legend()

                plt.figure()
                if startang < -np.pi/2:
                    startang += 2*np.pi
                    startangs += 2*np.pi
                plt.errorbar(180/np.pi * startangs, err_full_40, yerr=np.std(cumerr_full_40/nr_of_rec, axis=1)/np.sqrt(rep), lw=3, label='Full model')
                plt.scatter(180/np.pi * startang, err_full_40[int((samp-1)/2)], s=50, c='r', zorder=5)
                plt.errorbar(180/np.pi * startang, err_full_40[int((samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
                plt.xlabel('R' + str(receptor)+' starting angle')
                plt.ylabel('Mean performance')
                plt.ylim([-0.05, 1.05])
                # plt.legend()

                plt.figure()
                if startang < -np.pi/2:
                    startang += 2*np.pi
                    startangs += 2*np.pi
                plt.errorbar(180/np.pi * startangs, err_full_40_v, yerr=np.std(cumerr_full_40_v/nr_of_rec, axis=1)/np.sqrt(rep), lw=3, label='Full model')
                plt.scatter(180/np.pi * startang, err_full_40_v[int((samp-1)/2)], s=50, c='r', zorder=5)
                plt.errorbar(180/np.pi * startang, err_full_40_v[int((samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
                plt.xlabel('R' + str(receptor)+' starting angle')
                plt.ylabel('Mean performance')
                plt.ylim([-0.05, 1.05])
                # plt.legend()

                if include_equator:
                    np.save('starting_angle_R' + str(receptor) + '_WT_35_centstiff' + str(cent_stiff[0]) + '_eq', err_full_35)
                    np.save('starting_angle_R' + str(receptor) + '_WT_35_centstiff' + str(cent_stiff[0]) + 'voronoi_eq', err_full_35_v)
                    np.save('starting_angle_R' + str(receptor) + '_WT_40_centstiff' + str(cent_stiff[0]) + '_eq', err_full_40)
                    np.save('starting_angle_R' + str(receptor) + '_WT_40_centstiff' + str(cent_stiff[0]) + 'voronoi_eq', err_full_40_v)
                else:
                    np.save('starting_angle_R' + str(receptor) + '_WT_35_centstiff' + str(cent_stiff[0]), err_full_35)
                    np.save('starting_angle_R' + str(receptor) + '_WT_35_centstiff' + str(cent_stiff[0]) + 'voronoi', err_full_35_v)
                    np.save('starting_angle_R' + str(receptor) + '_WT_40_centstiff' + str(cent_stiff[0]), err_full_40)
                    np.save('starting_angle_R' + str(receptor) + '_WT_40_centstiff' + str(cent_stiff[0]) + 'voronoi', err_full_40_v)

                # restrict the angular range to +-20 or so, check the performance curves
                startangs = startang + np.linspace(-45, 45, samp) * np.pi/180
                angle_at_p30 = np.array([d[7] for d in data]).reshape((samp, rep))
                mean_ang = np.angle(np.mean(np.exp(1j*angle_at_p30), axis=1))[:-1]
                std_ang = (circstd(angle_at_p30, axis=1)/np.sqrt(rep))[:-1]
                startangs = startangs[:-1]
                if receptor in (1, 6):
                    mean_ang = np.where(mean_ang < 0, mean_ang + 2*np.pi, mean_ang)
                plt.figure()
                plt.scatter(180/np.pi * startangs, 180/np.pi * mean_ang, s=50, zorder=5)
                plt.errorbar(180/np.pi * startangs, 180/np.pi * mean_ang, yerr=180/np.pi * std_ang)
                minang, maxang = np.min(180/np.pi * startangs) - 40, np.max(180/np.pi * startangs) + 40
                plt.axis([minang, maxang, minang, maxang])
                plt.gca().set_aspect('equal')
                plt.plot([minang, maxang], [minang, maxang], 'k', ls='dotted')
                plt.xlabel('R' + str(receptor)+' starting angle (deg)')
                plt.ylabel('R' + str(receptor)+' angle at P30 (deg)')

        elif investigate == 'stiffness':
            # calculate the stiffness parameter sweeps
            nsamp = 10
            cent_stiff = np.linspace(25, 32, nsamp)
            sl_stiff = np.linspace(0.1, 2, nsamp)

            data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(c_st, sl_st, startang, A_magnet, circ_width, r) for c_st in cent_stiff for sl_st in sl_stiff)
            cumerr_35 = np.array([d[0] for d in data])
            cumerr_35 = np.reshape(np.array(cumerr_35), (nsamp, nsamp))
            cumerr_40 = np.array([d[1] for d in data])
            cumerr_40 = np.reshape(np.array(cumerr_40), (nsamp, nsamp))
            cumerr_35_v = np.array([d[2] for d in data])
            cumerr_35_v = np.reshape(np.array(cumerr_35_v), (nsamp, nsamp))
            cumerr_40_v = np.array([d[3] for d in data])
            cumerr_40_v = np.reshape(np.array(cumerr_40_v), (nsamp, nsamp))
            len_at_times = np.array([d[4] for d in data])
            len_at_times = np.mean(len_at_times, axis=0)

            fig35, ax35 = plt.subplots(1, 1, figsize=(8, 8))
            create_heatmap(cumerr_35/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 1, np.round(sl_stiff, 2), np.round(sl_stiff, 2)), cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax35, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig40, ax40 = plt.subplots(1, 1, figsize=(8, 8))
            create_heatmap(cumerr_40/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 1, np.round(sl_stiff, 2), np.round(sl_stiff, 2)), cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax40, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig35_v, ax35_v = plt.subplots(1, 1, figsize=(8, 8))
            create_heatmap(cumerr_35_v/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 1, np.round(sl_stiff, 2), np.round(sl_stiff, 2)), cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax35_v, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig40_v, ax40_v = plt.subplots(1, 1, figsize=(8, 8))
            create_heatmap(cumerr_40_v/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 1, np.round(sl_stiff, 2), np.round(sl_stiff, 2)), cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax40_v, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            np.save('stiffness_sweep_R'+str(receptor)+'_WT_36hAPF_eq', cumerr_35)
            np.save('stiffness_sweep_R'+str(receptor)+'_WT_40hAPF_eq', cumerr_40)
            np.save('stiffness_sweep_R'+str(receptor)+'_WT_36hAPF_eq_voronoi', cumerr_35_v)
            np.save('stiffness_sweep_R'+str(receptor)+'_WT_40hAPF_eq_voronoi', cumerr_40_v)

            ########
            # to load:
            ########
            # cumerr_35 = np.load('stiffness_sweep_R'+str(receptor)+'_WT_36hAPF_eq.npy')
            # cumerr_40 = np.load('stiffness_sweep_R'+str(receptor)+'_WT_40hAPF_eq.npy')
            # cumerr_35_v = np.load('stiffness_sweep_R'+str(receptor)+'_WT_36hAPF_eq_voronoi.npy')
            # cumerr_40_v = np.load('stiffness_sweep_R'+str(receptor)+'_WT_40hAPF_eq_voronoi.npy')

        elif investigate == 'magnet_strength':
            # calculates the parameter sweep for adhesion strength A_L
            cent_stiff = np.array([25])
            sl_stiff = np.array([0.4])
            samp = 11
            As = np.linspace(0, 10, samp)

            novalleydyn = False
            rep = 1
            data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(cent_stiff, sl_stiff, startang, A_magnet, circ_width, r) for A_magnet in np.repeat(As, rep))
            cumerr = np.array([d[1] for d in data])
            cumerr = np.reshape(np.array(cumerr), (samp, rep))
            err = np.mean(cumerr, axis=1)/nr_of_rec

            plt.figure()
            plt.plot(As, err, lw=3, label='Full model')
            plt.xlabel('R' + str(receptor)+' magnet strength')
            plt.ylabel('Mean performance')
            # plt.legend()
            plt.ylim([0, 1.05])

        elif investigate == 'magnet_timing':
            # calculates the parameter sweep for adhesion timing
            nsamp = 10
            cent_magnet = np.linspace(25, 38, nsamp)
            slopes_magnet = np.linspace(0.1, 2, nsamp)

            c_st = 32
            sl_st = 0.45
            data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(c_st, sl_st, startang, A_magnet, circ_width, r, c_magnet, sl_magnet) for c_magnet in cent_magnet for sl_magnet in slopes_magnet)
            cumerr_35 = np.array([d[0] for d in data])
            cumerr_35 = np.reshape(np.array(cumerr_35), (nsamp, nsamp))
            cumerr_40 = np.array([d[1] for d in data])
            cumerr_40 = np.reshape(np.array(cumerr_40), (nsamp, nsamp))
            cumerr_35_v = np.array([d[2] for d in data])
            cumerr_35_v = np.reshape(np.array(cumerr_35_v), (nsamp, nsamp))
            cumerr_40_v = np.array([d[3] for d in data])
            cumerr_40_v = np.reshape(np.array(cumerr_40_v), (nsamp, nsamp))
            len_at_times = np.array([d[4] for d in data])
            len_at_times = np.mean(len_at_times, axis=0)

            fig_35, ax_35 = plt.subplots(1, 1, figsize=(10, 6))
            create_heatmap(cumerr_35/nr_of_rec, np.round(cent_magnet, 1), np.where(slopes_magnet < 2, np.round(slopes_magnet, 3), np.array(slopes_magnet, dtype='int')), cbarlabel='Mean performance', xlabel='Slope', ylabel='Central point (hAPF)', ax1=ax_35, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig_40, ax_40 = plt.subplots(1, 1, figsize=(10, 6))
            create_heatmap(cumerr_40/nr_of_rec, np.round(cent_magnet, 1), np.where(slopes_magnet < 2, np.round(slopes_magnet, 3), np.array(slopes_magnet, dtype='int')), cbarlabel='Mean performance', xlabel='Slope', ylabel='Central point (hAPF)', ax1=ax_40, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig_35_v, ax_35_v = plt.subplots(1, 1, figsize=(10, 6))
            create_heatmap(cumerr_35_v/nr_of_rec, np.round(cent_magnet, 1), np.where(slopes_magnet < 2, np.round(slopes_magnet, 3), np.array(slopes_magnet, dtype='int')), cbarlabel='Mean performance', xlabel='Slope', ylabel='Central point (hAPF)', ax1=ax_35_v, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

            fig_40_v, ax_40_v = plt.subplots(1, 1, figsize=(10, 6))
            create_heatmap(cumerr_40_v/nr_of_rec, np.round(cent_magnet, 1), np.where(slopes_magnet < 2, np.round(slopes_magnet, 3), np.array(slopes_magnet, dtype='int')), cbarlabel='Mean performance', xlabel='Slope', ylabel='Central point (hAPF)', ax1=ax_40_v, vmin=0, vmax=1, origin='lower')
            plt.tight_layout()

        elif investigate == 'paths':
            # investigate the shape of the paths for changes of the torch light
            include_equator = False
            c_st = 25
            sl_st = 0.45
            nsamp = 16
            nrep = 5
            radius_all = r * np.linspace(0.5, 2, nsamp)
            data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(c_st, sl_st, startang, A_magnet, circ_width, radius) for radius in np.repeat(radius_all, nrep))

            FRONTLOCS_TIME = np.array([d[0] for d in data]).reshape((nsamp, nrep, -1, 1, 2))
            paths = np.mean(FRONTLOCS_TIME, axis=1)[:, :, 0, :]
            REC_X = np.array([d[1] for d in data])[0, :, :]
            REC_Y = np.array([d[2] for d in data])[0, :, :]
            Xint = np.array([d[3] for d in data])[0, :]
            Yint = np.array([d[4] for d in data])[0, :]

            plt.figure()
            cmap = plt.get_cmap('Purples', nsamp)
            for i in range(nsamp):
                plt.plot(paths[i, :, 0], paths[i, :, 1], lw=2, c=cmap(i))
            norm = mpl.colors.Normalize(vmin=0.5, vmax=2)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, ticks=np.linspace(0.5, 2, nsamp))
            cb.ax.tick_params(labelsize=10)
            cb.set_label('Relative radius of ROI')
            plt.scatter(REC_X, REC_Y)

            for xx, yy in zip(Xint, Yint):
                circle_temp = Point(xx, yy).buffer(1)  # type(circle)=polygon
                ellipse = shapely.affinity.scale(circle_temp, 1.0*LCELL_SIZE[-1, 0]*CONVFAC/2, 1.0*LCELL_SIZE[-1, 1]*CONVFAC/2)  # type(ellipse)=polygon
                circle = Point(xx - 0.333*vec_opt[-1, 2], yy).buffer(0.6*LCELL_SIZE[-1, 1]*CONVFAC)
                uni = circle.union(ellipse)
                plt.gca().add_patch(descartes.PolygonPatch(uni, fc='g', ec='g', alpha=0.2))

            plt.axis([1000, 1300, 400, 600])

    if create_movie:
        create_movie_from_png('front_movement_model', False)
        mpl.use('Qt5Agg')
    print('\nTotal time: ' + str(np.round(time.time()-start, 2)) + ' seconds')
