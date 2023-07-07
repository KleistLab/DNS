#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:57:50 2022

@author: eric
"""
import os
import json
import base64
import time
import descartes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle # Ellipse
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance as dist
from scipy.interpolate import splrep, splev
from scipy.stats import circstd
from scipy.ndimage import uniform_filter1d
import numexpr as ne
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from joblib import Parallel, delayed
import shapely.affinity
from shapely.geometry import Point
from helper_functions import remove_ticks_and_box, create_heatmap, rotate_coord,\
    calc_closest_point_on_ellipse, tanh_cust, create_starting_grid_noLcell,\
        create_movie_from_png

start = time.time()

def generate_indmat(xind, yind, fr_x, fr_y, r, frontang, circ_wid, REC_X_INTER, REC_Y_INTER):
    """
    creates the mask for placing the filopodia in the density landscape,
    included circ_wid as a parameter here, in contrast to the WT version
    of this function
    """
    rec_x, rec_y = np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)
    rec_x, rec_y = rec_x[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r **
                         2], rec_y[(rec_x - fr_x)**2 + (rec_y - fr_y)**2 <= 1.2*r**2]
    ind = np.zeros(np.shape(xind), dtype='bool')
    # ind1 = ((xind - fr_x)**2 + (yind - fr_y)**2 <= r**2)
    ind1 = ne.evaluate('((xind-fr_x)**2 + (yind-fr_y)**2)<=r**2')
    # 0.77) #0.25882) #0.77)
    ind2 = (np.cos(np.arctan2(yind[ind1]-fr_y,
            xind[ind1]-fr_x) - frontang) > circ_wid)
    ind[yind[ind1][ind2], xind[ind1][ind2]] = True
    radius = 0.5 * CONVFAC
    ind3 = np.zeros(np.shape(xind), dtype='bool')
    for rx, ry in zip(rec_x, rec_y):
        ind3 += ne.evaluate('((xind-rx)**2 + (yind-ry)**2)<=radius**2')
    return np.logical_and(ind, ~ind3)

def sample_roi(dat2_inter, FRONTLOC, circ_wid, meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER):
    """
    samples n_fil filopodial locations from the mask region created in
    generate_indmat()
    """
    xfil, yfil = [], []
    allinds = np.zeros(1023*2047, dtype='bool')
    for irec in range(nr_of_rec):
        if region == 'circle':
            ind = (
                np.sqrt((xind - FRONTLOC[irec, 0])**2 + (yind - FRONTLOC[irec, 1])**2) <= r)
        elif region == 'circ_segment':
            frontang = np.arctan2(
                meanvec_old[irec].imag, meanvec_old[irec].real)
            ind = generate_indmat(
                xind, yind, FRONTLOC[irec, 0], FRONTLOC[irec, 1], r, frontang, circ_wid, REC_X_INTER, REC_Y_INTER)
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

def distance_to_exp(firstpos, pos_eval, a, b, Xint, Yint, v1int, v2int, border, mode='target_area'):
    """
    evaluates if pos_eval is within the correct target area/Voronoi cell,
    slightly different from the WT version of this function, especially at the equator
    """
    goal_loc = np.zeros((nr_of_rec, 2))
    for kk in range(nr_of_rec):
        D_bundles = dist.cdist([firstpos[kk, :]], [[x, y] for x, y in zip(Xint, Yint)], metric='euclid')
        goal_loc[kk, 0] = Xint[D_bundles.argmin(axis=1)]
        goal_loc[kk, 1] = Yint[D_bundles.argmin(axis=1)]

    if nr_of_rec == 1 or not include_equator:
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
    elif nr_of_rec == 9 and include_equator:
        if receptor == 1:
            goal_loc[np.array([0, 1, 5, 6])] += -v2int[None, :]
            goal_loc[np.array([2, 3, 4, 7, 8])] += v1int[None, :] - v2int[None, :]
        elif receptor == 2:
            goal_loc[np.array([0, 1, 5, 6])] += v1int[None, :] - v2int[None, :]
            goal_loc[np.array([2, 3, 4, 7, 8])] += - v2int[None, :]
        elif receptor == 3:
            goal_loc[np.array([0, 1, 5])] += 2*v1int[None, :] - 1*v2int[None, :]
            goal_loc[np.array([2, 3, 4, 6, 7, 8])] += -v1int[None, :] - 1*v2int[None, :]
        elif receptor == 4:
            goal_loc[np.array([0, 1, 5, 6])] += v1int[None, :]
            goal_loc[np.array([2, 3, 4, 7, 8])] += -v1int[None, :]
        elif receptor == 5:
            goal_loc[np.array([0, 1, 5, 6])] += v2int[None, :]
            goal_loc[np.array([2, 3, 4, 7, 8])] += -v1int[None, :] + v2int[None, :]
        elif receptor == 6:
            goal_loc[np.array([0, 1, 5, 6])] += -v1int[None, :] + v2int[None, :]
            goal_loc[np.array([2, 3, 4, 7, 8])] += v2int[None, :]

    if include_equator:
        if receptor == 1:
            ind = np.array([2, 3, 4, 7, 8])
        elif receptor == 2:
            ind = np.array([3, 4, 6, 7, 8])
        elif receptor == 3:
            ind = np.array([1, 4, 5, 6, 8])
        elif receptor == 4:
            ind = np.array([1, 3, 4, 6, 8])
        elif receptor == 5:
            ind = np.array([2, 3, 4, 6, 7, 8])
        elif receptor == 6:
            ind = np.array([2, 3, 4, 7, 8])

    if mode == 'target_area':
        # print(b)
        # print(v1int)
        correct1 = (pos_eval[:, 0] - goal_loc[:, 0] + 0.333*v1int[0])**2 + (pos_eval[:, 1] - goal_loc[:, 1] + 0.333*v1int[1])**2 < 1.2*b**2
        if include_equator:
            correct1[ind] = (pos_eval[ind, 0] - goal_loc[ind, 0] - 0.333*v1int[0])**2 + (pos_eval[ind, 1] - goal_loc[ind, 1] - 0.333*v1int[1])**2 < 1.2*b**2
        correct2 = np.sqrt((pos_eval[:, 0] - goal_loc[:, 0])**2/a**2 + (pos_eval[:, 1] - goal_loc[:, 1])**2/b**2) <= 1 #0.8
        correct = np.logical_or(correct1, correct2)
    elif mode == 'voronoi':
        closest_bundle = np.zeros((len(goal_loc), 2))
        for kk in range(len(goal_loc)):
            Xs_ell, Ys_ell = calc_closest_point_on_ellipse(a, b, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint, Yint)])[:, None, :])
            if include_equator:
                Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*b, 1.2*b, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(np.where(Xint < border, Xint - 0.333*v1int[0], Xint + 0.333*v1int[0]), Yint)])[:, None, :])
                D_bundles_ell = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(Xint + Xs_ell[:, 0], Yint + Ys_ell[:, 0])], metric='euclid')
                D_bundles_circ = dist.cdist([pos_eval[kk, :]], [[x, y] for x, y in zip(np.where(Xint < border, Xint - 0.333*v1int[0] + Xs_circ[:, 0], Xint + 0.333*v1int[0] + Xs_circ[:, 0]), Yint + Ys_circ[:, 0])], metric='euclid')
            else:
                Xs_circ, Ys_circ = calc_closest_point_on_ellipse(1.2*b, 1.2*b, pos_eval[None, kk, :] - np.array([[x, y] for x,y in zip(Xint - 0.333*v1int[0], Yint)])[:, None, :])
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

def run_gc(c_st, sl_st, startang, A_magnet=0, c_magnet=35, sl_magnet=0.4):
    """
    this is the core method, simulating the growing axon(s)
    c_st  ... central point of the stiffness time course
    sl_st ... slope of the stiffness time course
    startang ... initial extension angle of the axon(s)
    A_magnet ... adhesion strength of the L-cells, just for playing, usually set to zero
    c_magnet  ... central point of the adhesion time course, usually not needed
    sl_magnet ... slope of the adhesion time course, usually not needed
    """
    print(receptor)
    irep = 0
    cum_error_35, cum_error_40, cum_error_45 = 0, 0, 0
    cum_error_35_v, cum_error_40_v = 0, 0
    len_avg = []
    outside_fil_all = []
    len_at_times = []
    xind, yind = np.meshgrid(np.arange(2*Nx-1), np.arange(2*Ny-1))
    pval_inter = np.zeros((nr_of_rec, 4))
    while irep < nrep:
        # initialize the growth vector with the theoretical expectation for
        # random uniformly placed points in a circular segment of given radius and angular width
        meanvec_old = np.ones(nr_of_rec) * sp * 2/3*radii[receptor-1]*np.sin(np.pi/180 * circ_width)/(
            np.pi/180 * circ_width) * np.exp(1j*startang)
        for hh, hour in enumerate(newtime):
            print(hour)
            v1, v2 = vec_opt_ext[hh, :2], vec_opt_ext[hh, 2:]
            Xstart, Ystart = 1024, 512

            center_loc_x_int = (Xstart + n1*v1[0] + n2*v2[0]).flatten()
            center_loc_y_int = (Ystart + n1*v1[1] + n2*v2[1]).flatten()
            ind = (center_loc_x_int > 0)*(center_loc_x_int < 2048) * \
                (center_loc_y_int > 0)*(center_loc_y_int < 1024)
            Xint, Yint = center_loc_x_int[ind], center_loc_y_int[ind]

            REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid_noLcell(
                Xint, Yint)

            REC_X_SHIFT, REC_Y_SHIFT, _, _ = create_starting_grid_noLcell(
                np.array([Xstart]), np.array([Ystart]))
            Xstart_rec, Ystart_rec = REC_X_SHIFT[receptor -
                                                 1, 0], REC_Y_SHIFT[receptor-1, 0]

            if hh == 0:
                if nr_of_rec > 1:
                    FRONTLOC = np.array([Xstart_rec, Ystart_rec]) + np.arange(-(
                        (nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2 + 0.5)[:, None] * v2
                    FRONTLOC = np.vstack(
                        (FRONTLOC, np.array([ff - v1 for ff in FRONTLOC[1:]])))
                    Xstart_rec_all, Ystart_rec_all = FRONTLOC[:, 0].copy(
                    ), FRONTLOC[:, 1].copy()
                else:
                    FRONTLOC = np.array([[Xstart_rec, Ystart_rec]])
                    print(np.shape(FRONTLOC))
                    Xstart_rec_all, Ystart_rec_all = FRONTLOC[:, 0].copy(
                    ), FRONTLOC[:, 1].copy()

            if create_movie and hour <= 40:
                fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={
                                       'height_ratios': [1, 20]})
                ax[0].barh([1], [40], color='w', edgecolor='k')
                ax[0].barh([1], [hour], color='k')
                ax[0].set_title('Developmental time (hAPF)')
                ax[0].tick_params(axis='y', which='both', right=False,
                                  left=False, labelleft=False)
                ax[0].set_xticks([20, 25, 30, 35, 40])
                for pos in ['right', 'top', 'bottom', 'left']:
                    ax[0].spines[pos].set_visible(False)

                ax[0].set_xlim([19.9, 40.1])

            # mirror heel and front locations at the equator, if needed
            if include_equator:
                if hh == 0:
                    border = np.mean(XC) - 10
                else:
                    currmat = np.array([v1, v2]).T
                    prevmat = np.array(
                        [vec_opt_ext[hh-1, :2], vec_opt_ext[hh-1, 2:]]).T
                    border = ((currmat@np.linalg.inv(prevmat) @
                              ((border-Xstart)*np.ones(2)).T).T)[0] + Xstart
                Dmat = dist.cdist([[x, y] for x, y in zip(Xstart_rec_all, Ystart_rec_all)], [
                                  [x, y] for x, y in zip(XC, YC)], metric='euclid')
                ind2 = Dmat.argmin(axis=1)
                if hh == 0:
                    FRONTLOC[:, 0] = np.where(
                        XC[ind2] >= border, FRONTLOC[:, 0] - 2*(FRONTLOC[:, 0] - XC[ind2]), FRONTLOC[:, 0])
                    FRONTLOCS_TIME_temp = FRONTLOC.copy()
                    FRONTLOCS_TIME = FRONTLOCS_TIME_temp.copy()
                    meanvec_old[XC[ind2] >= border] = meanvec_old[XC[ind2]
                                                                  >= border] - 2*(meanvec_old[XC[ind2] >= border].real)
                REC_X_INTER[:, XC >= border] = REC_X_INTER[:, XC >= border] - \
                    2*(REC_X_INTER[:, XC >= border] -
                       XC[XC >= border][None, :])
            else:
                border = 0
                if hh == 0:
                    FRONTLOCS_TIME_temp = FRONTLOC.copy()
                    FRONTLOCS_TIME = FRONTLOCS_TIME_temp.copy()

            xmin2, xmax2, ymin2, ymax2 = 0, 2*Nx, 0, 2*Ny
            POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx-1),
                              np.linspace(ymin2, ymax2, 2*Ny-1))
            POS = np.array(POS)

            # load files created by create_grid_for_all_timesteps_noLcell.py
            if include_equator:
                dat2_inter = np.load(
                    './data-files/dat2_inter_R'+str(receptor)+'_'+str(hh)+'hAPF_noLcell.npy')
            else:
                dat2_inter = np.load(
                    './data-files/dat2_inter_R'+str(receptor)+'_'+str(hh)+'hAPF_noLcell_noeq.npy')

            # create index array for the circle segment
            xfil, yfil, allinds = sample_roi(dat2_inter, FRONTLOC, np.cos(
                np.pi/180*circ_width), meanvec_old, region, xind, yind, POS, REC_X_INTER, REC_Y_INTER)

            # how many of the filopodia are in the target area?
            if calc_inside_filop:
                a, b = LCELL_SIZE[0]/2*CONVFAC, LCELL_SIZE[1]/2*CONVFAC
                outside_fil = distance_to_exp(FRONTLOCS_TIME[0, :, :], np.c_[
                                              xfil, yfil], a, b, Xint, Yint, v2, v1, border)
                print(np.sum(outside_fil)/len(xfil))
                outside_fil_all.append(outside_fil)

            xfil = np.reshape(xfil, (nr_of_rec, -1))
            yfil = np.reshape(yfil, (nr_of_rec, -1))
            angl = np.arctan2(
                yfil-FRONTLOC[:, 1][:, None], xfil-FRONTLOC[:, 0][:, None])
            radi = np.sqrt(
                (yfil-FRONTLOC[:, 1][:, None])**2 + (xfil-FRONTLOC[:, 0][:, None])**2)
            goal_loc = np.zeros((2, nr_of_rec))
            for kk in range(nr_of_rec):
                D_bundles = dist.cdist([FRONTLOC[kk, :]], [[x, y] for x, y in zip(
                    Xint - LCELL_SIZE[0]/2*CONVFAC, Yint)], metric='euclid')
                goal_loc[0, kk] = Xint[D_bundles.argmin(
                    axis=1)] - LCELL_SIZE[0]/2*CONVFAC
                goal_loc[1, kk] = Yint[D_bundles.argmin(axis=1)]
            if create_movie and hour <= 40:
                if not movie_pure:
                    len1 = 100
                    viridis = cm.get_cmap('viridis', len1)
                    rgba_vir = viridis.colors
                    norm = plt.Normalize(np.min(dat2_inter[200:800, 1300:1600]), np.max(
                        dat2_inter[200:800, 200:1600]))
                    cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                             'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                             'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}
                    newcmp = LinearSegmentedColormap(
                        'testCmap', segmentdata=cdict, N=256)
                    ax[1].imshow(dat2_inter, origin='lower',
                                 cmap=newcmp, norm=norm)
                    ax[1].imshow(np.reshape(allinds, (1023, 2047)),
                                 origin='lower', alpha=0.2)
                    for irec in range(nr_of_rec):
                        ax[1].scatter(*FRONTLOC[irec, :], c='r')
                        ax[1].scatter(xfil, yfil, s=5, c='w')
                        if np.ndim(FRONTLOCS_TIME) == 3:
                            ax[1].plot(FRONTLOCS_TIME[:, irec, 0],
                                       FRONTLOCS_TIME[:, irec, 1], 'r-', lw=2)
                            ax[1].scatter(FRONTLOCS_TIME[0, irec, 0], FRONTLOCS_TIME[0, irec, 1], c='b', zorder=5)
                        else:
                            ax[1].scatter(*FRONTLOC[irec, :], c='b', zorder=5)
                    for xx, yy in zip(Xint, Yint):
                        circle_temp = Point(xx, yy).buffer(1)
                        ellipse = shapely.affinity.scale(
                            circle_temp, 1.0*LCELL_SIZE[0]*CONVFAC/2, 1.0*LCELL_SIZE[1]*CONVFAC/2)  # type(ellipse)=polygon
                        if include_equator and xx >= border:
                            circle = Point(
                                xx + 0.333*v2[0], yy + 0.333*v2[1]).buffer(0.6*LCELL_SIZE[1]*CONVFAC)
                        else:
                            circle = Point(
                                xx - 0.333*v2[0], yy - 0.333*v2[1]).buffer(0.6*LCELL_SIZE[1]*CONVFAC)
                        uni = circle.union(ellipse)
                        ax[1].add_patch(descartes.PolygonPatch(
                            uni, fc='g', ec='g', alpha=0.2))
                    for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                        ax[1].add_patch(
                            Circle((xx, yy), 0.5*CONVFAC, ls='--', edgecolor='k', facecolor='none'))

                    if nr_of_rec == 9:
                        ax[1].axis([400, 1600, 200, 800])
                    elif nr_of_rec == 1:
                        if receptor == 3:
                            ax[1].axis([900, 1500, 315, 715])
                        elif receptor < 3:
                            ax[1].axis([700, 1300, 315, 715])
                        else:
                            ax[1].axis([700, 1300, 350, 750])

                    remove_ticks_and_box(ax[1])
                    fig.tight_layout()

                    if show_insets:
                        inset_axes(ax[1], width='20%', height='20%', loc=2)
                        plt.plot(np.arange(
                            20, 40, 0.01), 1-tanh_cust(np.arange(20, 40, 0.01), c_st, sl_st), 'k-', lw=3)
                        plt.scatter(hour, 1-tanh_cust(hour, c_st,
                                    sl_st), c='r', s=40, zorder=5)
                        plt.ylim([-0.1, 1.1])
                        plt.xlabel('Time (hour)', color='w')
                        plt.ylabel('Stiffness', color='w')
                        plt.gca().tick_params(axis='x', colors='w')
                        plt.gca().tick_params(axis='y', colors='w')
                        plt.gca().yaxis.set_label_position("right")
                        plt.gca().yaxis.tick_right()

                        inset_axes(ax[1], width='20%', height='20%', loc=1)
                        plt.plot(np.arange(20, 40, 0.01), np.zeros(
                            len(np.arange(20, 40, 0.01))), 'k-', lw=3)
                        plt.scatter(hour, 0, c='r', s=40, zorder=5)
                        plt.ylim([-0.1, 1.1])
                        plt.xlabel('Time (hour)', color='w')
                        plt.ylabel('L-cell attraction', color='w')
                        plt.gca().tick_params(axis='x', colors='w')
                        plt.gca().tick_params(axis='y', colors='w')

                    if hour == newtime[ind_time[2]]:
                        tt = 0
                        while tt < 30:
                            filename = str('%05d' % (100*hh + tt) + '.png')
                            plt.savefig(save_folder + '/' + filename, dpi=100)
                            tt += 1
                    else:
                        filename = str('%05d' % (100*hh) + '.png')
                        plt.savefig(save_folder + '/' + filename, dpi=100)
                    plt.close(fig)
                else:
                    if include_equator:
                        eq_max = np.max(dat2_inter[200:800, 200:1600])
                        flat_min = np.min(dat2_inter[200:800, 1300:1600])

                        len1 = 100
                        viridis = cm.get_cmap('viridis', len1)
                        rgba_vir = viridis.colors

                        norm = plt.Normalize(flat_min, eq_max)

                        cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                                 'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                                 'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}

                        newcmp = LinearSegmentedColormap(
                            'testCmap', segmentdata=cdict, N=256)
                        ax[1].imshow(dat2_inter, origin='lower',
                                     cmap=newcmp, norm=norm)
                        for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                            ax[1].add_patch(
                                Circle((xx, yy), 0.5*CONVFAC, ls='--', edgecolor='k', facecolor='none'))

                        remove_ticks_and_box(ax[1])
                    else:
                        len1 = 100
                        viridis = cm.get_cmap('viridis', len1)
                        rgba_vir = viridis.colors
                        norm = plt.Normalize(np.min(dat2_inter[200:800, 1300:1600]), np.max(
                            dat2_inter[200:800, 200:1600]))
                        cdict = {'red':   np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 0].reshape(-1, 1), rgba_vir[:, 0].reshape(-1, 1))),
                                 'green': np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 1].reshape(-1, 1), rgba_vir[:, 1].reshape(-1, 1))),
                                 'blue':  np.hstack(((np.linspace(0, 1, len1, endpoint=True)**2).reshape((-1, 1)), rgba_vir[:, 2].reshape(-1, 1), rgba_vir[:, 2].reshape(-1, 1)))}
                        newcmp = LinearSegmentedColormap(
                            'testCmap', segmentdata=cdict, N=256)
                        ax[1].imshow(dat2_inter, origin='lower',
                                     cmap=newcmp, norm=norm)
                        for xx, yy in zip(np.ravel(REC_X_INTER), np.ravel(REC_Y_INTER)):
                            ax[1].add_patch(
                                Circle((xx, yy), 0.5*CONVFAC, ls='--', edgecolor='k', facecolor='none'))
                    ax[1].axis([400, 1600, 200, 800])

                    remove_ticks_and_box(ax[1])
                    fig.tight_layout()
                    filename = str('%04d' % (hh) + '.png')
                    plt.savefig(filename, dpi=100)
                    plt.close(fig)

            if np.ndim(FRONTLOCS_TIME) == 3:
                axon_len = np.sqrt((FRONTLOC[:, 0] - FRONTLOCS_TIME[0, :, 0])
                                   ** 2 + (FRONTLOC[:, 1] - FRONTLOCS_TIME[0, :, 1])**2)
            else:
                axon_len = np.sqrt(
                    (FRONTLOC[:, 0] - FRONTLOCS_TIME[:, 0])**2 + (FRONTLOC[:, 1] - FRONTLOCS_TIME[:, 1])**2)

            len_avg.append(np.mean(axon_len))
            stiff = 1 - tanh_cust(hour, c_st, sl_st)
            if hour in newtime[ind_time]:
                len_at_times.append(np.mean(axon_len))
            if np.ndim(FRONTLOCS_TIME) == 3:
                norm = np.shape(FRONTLOCS_TIME)[0]-1
            else:
                norm = 1
            stiffpart = stiff*meanvec_old/norm
            if novalleydyn:
                filopart = (1-stiff)*sp*np.abs(np.mean(radi *
                                                       np.exp(1j*angl), axis=1))*np.exp(1j*np.angle(stiffpart))
            else:
                filopart = (1-stiff)*sp*np.mean(radi*np.exp(1j*angl), axis=1)
            magnetpart = A_magnet * tanh_cust(hour, c_magnet, sl_magnet) * np.exp(1j*np.arctan2(
                goal_loc[1]-FRONTLOC[:, 1], goal_loc[0]-FRONTLOC[:, 0]))  # -lcellsize_int[tind, 0]/2*CONVFAC
            meanvec = stiffpart + filopart + magnetpart
            change = np.array([meanvec.real, meanvec.imag]).T

            if hh == 30: # corresponding to hour approx 25
                angle_at_p30 = np.angle(np.mean(radi*np.exp(1j*angl)))

            if hh > 0:
                currmat = np.array([v1, v2]).T
                prevmat = np.array(
                    [vec_opt_ext[hh-1, :2], vec_opt_ext[hh-1, 2:]]).T
                change = (currmat@np.linalg.inv(prevmat)@change.T).T
            FRONTLOC += change * dt
            FRONTLOCS_TIME_temp = np.vstack((FRONTLOCS_TIME_temp, FRONTLOC))
            FRONTLOCS_TIME = np.reshape(
                FRONTLOCS_TIME_temp.copy(), (-1, nr_of_rec, 2))
            meanvec_old = (FRONTLOCS_TIME[-1, :, 0] - FRONTLOCS_TIME[0, :, 0] + 1j*(
                FRONTLOCS_TIME[-1, :, 1] - FRONTLOCS_TIME[0, :, 1]))/dt

            if hh > 0:
                if nr_of_rec > 1:
                    FIRSTPOS_NEWCOORD = np.array([Xstart_rec, Ystart_rec]) + np.arange(-(
                        (nr_of_rec+1)/2-1)/2, ((nr_of_rec+1)/2-1)/2 + 0.5)[:, None] * v2
                    FIRSTPOS_NEWCOORD = np.vstack(
                        (FIRSTPOS_NEWCOORD, np.array([ff - v1 for ff in FIRSTPOS_NEWCOORD[1:]])))
                else:
                    FIRSTPOS_NEWCOORD = np.array([[Xstart_rec, Ystart_rec]])
                if include_equator:
                    FIRSTPOS_NEWCOORD[:, 0] = np.where(XC[ind2] >= border, FIRSTPOS_NEWCOORD[:, 0] - 2*(
                        FIRSTPOS_NEWCOORD[:, 0] - XC[ind2]), FIRSTPOS_NEWCOORD[:, 0])

                FRONTLOCS_TIME[:, :, 0] = FRONTLOCS_TIME[:, :, 0] + \
                    (FIRSTPOS_NEWCOORD[:, 0] - FRONTLOCS_TIME[0, :, 0])
                FRONTLOCS_TIME[:, :, 1] = FRONTLOCS_TIME[:, :, 1] + \
                    (FIRSTPOS_NEWCOORD[:, 1] - FRONTLOCS_TIME[0, :, 1])

                FRONTLOCS_TIME_temp = np.reshape(FRONTLOCS_TIME, (-1, 2))
                FRONTLOC = FRONTLOCS_TIME[-1, :, :]
            if hour == newtime[ind_time[2]]:
                a, b = LCELL_SIZE[0]/2*CONVFAC, LCELL_SIZE[1]/2*CONVFAC
                cum_error_35 += np.sum(distance_to_exp(
                    FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2, v1, border, mode='target_area'))
                cum_error_35_v += np.sum(distance_to_exp(
                    FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2, v1, border, mode='voronoi'))
            elif hour == newtime[ind_time[3]]:
                a, b = LCELL_SIZE[0]/2*CONVFAC, LCELL_SIZE[1]/2*CONVFAC
                cum_error_40 += np.sum(distance_to_exp(
                    FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2, v1, border, mode='target_area'))
                cum_error_40_v += np.sum(distance_to_exp(
                    FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2, v1, border, mode='voronoi'))
            elif hour == newtime[ind_time[4]]:
                a, b = LCELL_SIZE[0]/2*CONVFAC, LCELL_SIZE[1]/2*CONVFAC
                cum_error_45 += np.sum(distance_to_exp(
                    FRONTLOCS_TIME[0, :, :], FRONTLOCS_TIME[-1, :, :], a, b, Xint, Yint, v2, v1, border, mode='target_area'))
        irep += 1
    return cum_error_35/nrep, cum_error_40/nrep, cum_error_45/nrep, cum_error_35_v/nrep, cum_error_40_v/nrep, np.array(len_at_times), np.array(len_avg), pval_inter, outside_fil_all, angle_at_p30


mpl.use('Qt5Agg')  # turn on GUI for plotting

create_movie = True # whether or not to create a movie, otherwise parameter sweeps are calculated
show_insets = True # show insets with stiffness curve and adhesion curve in the movie
movie_pure = False # show only the density landscape
calc_inside_filop = False # whether or not to calculate the proportion of filopodia in the target area
if create_movie:
    mpl.use('Agg')  # do not show figures while creating movie

include_equator = False # whether or not to simulate the equator
receptor = 3 # receptor subtype to simulate

# parameters of the model
A_magnet = 0 # usually kept 0 here because we simulate L-cell ablation experiment
CONVFAC = 24
nsteps = 3
# like in WT
radii = np.array([1*4.41, 3.60, 5.15, 3.58, 4.48, 4.74])*CONVFAC
r = radii[receptor-1]
# like in WT
speeds = np.array([0.093/1, 0.053, 0.148, 0.09, 0.052, 0.077])
sp = speeds[receptor-1]
n_fil = 10
region = 'circ_segment'
novalleydyn = False
investigate = 'angle'  # can be 'angle' or 'stiffness'
if include_equator:
    nr_of_rec = 9
    save_folder = '/home/eric/axon_guidance/videos_and_individual_images_noLcells/R'+str(receptor)+'/9rec_eq'
else:
    nr_of_rec = 1
    save_folder = '/home/eric/axon_guidance/videos_and_individual_images_noLcells/R'+str(receptor)+'/1rec_noeq'

if not show_insets:
    save_folder = save_folder + '_noinsets'

hours = np.array([26, 30, 35, 40, 45])

LCELL_SIZE = np.array([2.96, 2.58])

# like in WT
circ_width = np.array([[76.93677250002409, 66.1581562071056, 58.64359788352946, 68.19374152821266],
                       [67.10341096706019, 63.6030367287868,
                           64.58480307212197, 64.68382969250712],
                       [59.99951401507621, 49.70632397768759,
                           59.76089748808765, 62.63689465660343],
                       [65.18759340181808, 64.6663385598332,
                           54.128563634672744, 58.3338061658033],
                       [73.7705462792746, 73.75319815476855,
                           70.71260029847282, 67.54024198457094],
                       [82.29011450458808, 70.63559641125377, 69.51033837975177, 58.19852457593135]]).mean(axis=1)[receptor-1]

# load fitted grid vectors for noLcell condition
with open("./data-files/optimal_grid_noLcell.json", "r", encoding='utf-8') as infile:
    loaded = json.load(infile).split('\"')
dtype = np.dtype(loaded[1])
arr = np.frombuffer(base64.decodebytes(bytearray(loaded[3], 'utf-8')), dtype)
vec_opt = arr.reshape((19, 4)).copy()

# rotate everything to have the grid perfectly horizontal
rot_ang = []
for hh in range(19):
    rot_ang.append(-np.arctan2(vec_opt[hh, 3], vec_opt[hh, 2]))
    vec_opt[hh, ::2], vec_opt[hh, 1::2] = rotate_coord(
        vec_opt[hh, ::2], vec_opt[hh, 1::2], rot_ang[-1])

# create the time stamps for the simulation
newtime = np.linspace(20, 45, 75, endpoint=True)
ind_time = []
for t in [25, 30, 35, 40, 45]:
    ind_time.append(np.argmin(np.abs(newtime - t)))

dt = np.diff(newtime)[0]

# interpolate the grid vectors for the simulation time, not really needed anymore
# as they are kept constant in the current version of noLcell experiments
vec_opt_ext = np.zeros((75, 4))
for i in range(4):
    y = uniform_filter1d(vec_opt[:, i], size=10)
    spl1 = splrep(
        np.hstack((np.array([20]), np.arange(27, 46))), np.hstack((np.mean(y), y)))
    vec_opt_ext[:, i] = splev(newtime, spl1)

mini, maxi = -10, 10
n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))
Nx, Ny = 1024, 512

# starting angles, from WT data
startangs_all = np.pi/180 * np.array([-140.6786, -64.3245, -17.25796667, 13.26706/2, 63.2865, 135.0751667])
startangs_std = np.array([8.758102091, 10.27462811, 5.683716097, 7.21548511, 14.56159809, 5.942082166])
startang = startangs_all[receptor-1]

if create_movie or include_equator:
    nrep = 1
else:
    nrep = 10

if nrep > 1:
    print('------------------------------')
    print('nrep is larger than 1')
    print('------------------------------')

if create_movie:
    cent_stiff = np.array([25])
    sl_stiff = np.array([0.45])
    err_mat = np.zeros((1, 1))
    len_avg_all = []
    for ic, c_st in enumerate(cent_stiff):
        for isl, sl_st in enumerate(sl_stiff):
            print(isl)
            cumerr_35, cumerr_40, cumerr_45, cumerr_35_v, cumerr_40_v, len_at_times, len_avg, pval_inter, outside_fil_all, angle_at_p30 = run_gc(
                c_st, sl_st, startang, A_magnet)
            err_mat[ic, isl] = cumerr_40/nrep
    len_avg_all.append(len_avg/CONVFAC)
    create_heatmap(err_mat/CONVFAC, np.round(cent_stiff/CONVFAC, 1), np.round(sl_stiff, 1),
                   cbarlabel='Error (um)', xlabel='Slope of stiffness', ylabel='Central point (um)', origin='lower')
    np.save('len_avg_R'+str(receptor)+'_noLcell',
            np.array(len_avg_all).reshape((nrep, -1)))
    if calc_inside_filop:
        print(str(np.round(np.mean(np.array(outside_fil_all)) * 100, 1)) +
              '% of filopodial tips are in the target area')
else:
    if investigate == 'angle':
        # cent_stiff = 25.2*np.array([2.2])
        # sl_stiff = np.array([0.02])
        cent_stiff = np.array([30]) #np.array([10]) #np.array([32])
        sl_stiff = np.array([0.45])
        samp = 11
        startangs = startang + np.linspace(-45, 45, samp) * np.pi/180

        novalleydyn = False
        rep = 5
        data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(cent_stiff,
                                                                sl_stiff, startang, A_magnet) for startang in np.repeat(startangs, rep))
        cumerr_full_35 = np.array([d[0] for d in data])
        cumerr_full_40 = np.array([d[1] for d in data])
        cumerr_full_35 = np.reshape(np.array(cumerr_full_35), (samp, rep))
        cumerr_full_40 = np.reshape(np.array(cumerr_full_40), (samp, rep))
        err_full_35 = np.mean(cumerr_full_35, axis=1)/nr_of_rec
        err_full_40 = np.mean(cumerr_full_40, axis=1)/nr_of_rec
        cumerr_full_35_v = np.array([d[3] for d in data])
        cumerr_full_40_v = np.array([d[4] for d in data])
        cumerr_full_35_v = np.reshape(np.array(cumerr_full_35_v), (samp, rep))
        cumerr_full_40_v = np.reshape(np.array(cumerr_full_40_v), (samp, rep))
        err_full_35_v = np.mean(cumerr_full_35_v, axis=1)/nr_of_rec
        err_full_40_v = np.mean(cumerr_full_40_v, axis=1)/nr_of_rec

        plt.figure()
        plt.errorbar(180/np.pi * startangs, err_full_35, yerr=0, lw=3, label='Full model')
        plt.scatter(180/np.pi * startang,
                    err_full_35[int((samp-1)/2)], s=50, c='r', zorder=5)
        plt.errorbar(180/np.pi * startang, err_full_35[int(
            (samp-1)/2)], xerr=startangs_std[receptor-1], ecolor='r', elinewidth=4, zorder=5)
        plt.xlabel('R' + str(receptor)+' starting angle')
        plt.ylabel('Mean performance')
        plt.ylim([-0.05, 1.05])

        plt.figure()
        plt.errorbar(180/np.pi * startangs, err_full_40, yerr=0, lw=3, label='Full model')
        plt.scatter(180/np.pi * startang,
                    err_full_40[int((samp-1)/2)], s=50, c='r', zorder=5)
        plt.errorbar(180/np.pi * startang, err_full_40[int(
            (samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
        plt.xlabel('R' + str(receptor)+' starting angle')
        plt.ylabel('Mean performance')
        plt.ylim([-0.05, 1.05])

        plt.figure()
        plt.errorbar(180/np.pi * startangs, err_full_35_v, yerr=0, lw=3, label='Full model')
        plt.scatter(180/np.pi * startang,
                    err_full_35_v[int((samp-1)/2)], s=50, c='r', zorder=5)
        plt.errorbar(180/np.pi * startang, err_full_35_v[int(
            (samp-1)/2)], xerr=startangs_std[receptor-1], ecolor='r', elinewidth=4, zorder=5)
        plt.xlabel('R' + str(receptor)+' starting angle')
        plt.ylabel('Mean performance')
        plt.ylim([-0.05, 1.05])

        plt.figure()
        plt.errorbar(180/np.pi * startangs, err_full_40_v, yerr=0, lw=3, label='Full model')
        plt.scatter(180/np.pi * startang,
                    err_full_40_v[int((samp-1)/2)], s=50, c='r', zorder=5)
        plt.errorbar(180/np.pi * startang, err_full_40_v[int(
            (samp-1)/2)], xerr=startangs_std[receptor-1], elinewidth=4, ecolor='r', zorder=5)
        plt.xlabel('R' + str(receptor)+' starting angle')
        plt.ylabel('Mean performance')
        plt.ylim([-0.05, 1.05])

        if include_equator:
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_35_centstiff'+ str(cent_stiff[0]) + '_eq', err_full_35)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_40_centstiff'+ str(cent_stiff[0]) + '_eq', err_full_40)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_35_centstiff'+ str(cent_stiff[0]) + '_voronoi_eq', err_full_35_v)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_40_centstiff'+ str(cent_stiff[0]) + '_voronoi_eq', err_full_40_v)
        else:
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_35_centstiff'+ str(cent_stiff[0]), err_full_35)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_40_centstiff'+ str(cent_stiff[0]), err_full_40)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_35_centstiff'+ str(cent_stiff[0]) + '_voronoi', err_full_35_v)
            np.save('starting_angle_10rep_R' +
                    str(receptor)+'_noLcell_40_centstiff'+ str(cent_stiff[0]) + '_voronoi', err_full_40_v)

        samp = 11
        startangs = startang + np.linspace(-45, 45, samp) * np.pi/180
        angle_at_p30 = np.array([d[9] for d in data]).reshape((samp, rep))
        mean_ang = np.angle(np.mean(np.exp(1j*angle_at_p30), axis=1))#[:9]
        std_ang = circstd(angle_at_p30, axis=1)/np.sqrt(rep)#[:9]
        if receptor == 3 or receptor == 4 or receptor == 5 or receptor == 2:
            plt.figure()
            plt.scatter(180/np.pi * startangs, 180/np.pi * mean_ang, s=50, zorder=5)
            plt.errorbar(180/np.pi * startangs, 180/np.pi * mean_ang, yerr=180/np.pi * std_ang)
            minang, maxang = np.min(180/np.pi * startangs) - 60, np.max(180/np.pi * startangs) + 60
        else:
            plt.figure()
            plt.scatter(180/np.pi * np.mod(startangs, 2*np.pi), 180/np.pi * np.mod(mean_ang, 2*np.pi), s=50, zorder=5)
            plt.errorbar(180/np.pi * np.mod(startangs, 2*np.pi), 180/np.pi * np.mod(mean_ang, 2*np.pi), yerr=180/np.pi * std_ang)
            minang, maxang = np.min(180/np.pi * np.mod(startangs, 2*np.pi)) - 60, np.max(180/np.pi * np.mod(startangs, 2*np.pi)) + 60
        plt.axis([minang, maxang, minang, maxang])
        plt.gca().set_aspect('equal')
        plt.plot([minang, maxang], [minang, maxang], 'k', ls='dotted')
        plt.xlabel('R' + str(receptor)+' starting angle (deg)')
        plt.ylabel('R' + str(receptor)+' angle at P25 (deg)')

    elif investigate == 'stiffness':
        nsamp = 10
        cent_stiff = np.linspace(25, 32, nsamp)
        sl_stiff = np.linspace(0.1, 2, nsamp)  # np.logspace(-2, 0, nsamp)
        data = Parallel(n_jobs=-1, verbose=100)(delayed(run_gc)(c_st, sl_st,
                                                                startang, A_magnet) for c_st in cent_stiff for sl_st in sl_stiff)

        cumerr_35 = np.array([d[0] for d in data])
        cumerr_40 = np.array([d[1] for d in data])
        cumerr_45 = np.array([d[2] for d in data])
        cumerr_35 = np.reshape(np.array(cumerr_35), (nsamp, nsamp))
        cumerr_40 = np.reshape(np.array(cumerr_40), (nsamp, nsamp))
        cumerr_45 = np.reshape(np.array(cumerr_45), (nsamp, nsamp))
        cumerr_35_v = np.array([d[3] for d in data])
        cumerr_40_v = np.array([d[4] for d in data])
        cumerr_35_v = np.reshape(np.array(cumerr_35_v), (nsamp, nsamp))
        cumerr_40_v = np.reshape(np.array(cumerr_40_v), (nsamp, nsamp))
        len_at_times = np.array([d[5] for d in data])
        len_at_times = np.mean(len_at_times, axis=0)

        fig35, ax35 = plt.subplots(1, 1, figsize=(8, 8))
        create_heatmap(cumerr_35/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 2, np.round(sl_stiff, 3), np.array(sl_stiff, dtype='int')),
                cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax35, vmin=0, vmax=1, origin='lower')
        plt.tight_layout()

        fig40, ax40 = plt.subplots(1, 1, figsize=(8, 8))
        create_heatmap(cumerr_40/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 2, np.round(sl_stiff, 3), np.array(sl_stiff, dtype='int')),
                cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax40, vmin=0, vmax=1, origin='lower')
        plt.tight_layout()

        fig35_v, ax35_v = plt.subplots(1, 1, figsize=(8, 8))
        create_heatmap(cumerr_35_v/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 2, np.round(sl_stiff, 3), np.array(sl_stiff, dtype='int')),
                cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax35_v, vmin=0, vmax=1, origin='lower')
        plt.tight_layout()

        fig40_v, ax40_v = plt.subplots(1, 1, figsize=(8, 8))
        create_heatmap(cumerr_40_v/nr_of_rec, np.round(cent_stiff, 1), np.where(sl_stiff < 2, np.round(sl_stiff, 3), np.array(sl_stiff, dtype='int')),
                cbarlabel='Mean performance', xlabel='Slope of stiffness', ylabel='Central point (hAPF)', ax1=ax40_v, vmin=0, vmax=1, origin='lower')
        plt.tight_layout()

if create_movie:
    os.chdir(save_folder)
    create_movie_from_png('front_movement_model_noLcell', remove_png_afterwards=False)
    mpl.use('Qt5Agg')  # turn on GUI for plotting

print('\nTotal time: ' + str(np.round(time.time()-start, 2)) + ' seconds')
