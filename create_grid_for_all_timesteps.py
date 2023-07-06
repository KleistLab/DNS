#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:56:38 2022

@author: eric
"""
import numpy as np
from scipy.signal import fftconvolve
from helper_functions import load_grid_heels_fronts, potential_parabola,\
    rotate_coord, create_starting_grid2, index_func

nsteps = 15 # number of time steps between two experimental measurements
hours = np.array([20, 26, 30, 35, 40])
CONVFAC = 25.2 # pixels per micrometer

# measured size of heels
rm_heels = np.array([[3.10231051831059, 2.91492280980449, 2.04300797855269, 2.16799853146643],
                     [3.36613563287586, 3.6474344212202, 2.59092160570733, 2.49818435400807],
                     [3.26, 3.45, 2.15, 2.1],
                     [2.5838454966756, 3.26122733330694, 2.43382971588824, 1.79884449690457],
                     [2.73134911063699, 3.95170014933712, 3.89010177959303, 3.75649529229474],
                     [3.97548396049507, 3.37776310930197, 3.23239134019713, 3.02306813633871]])

# measured size of fronts
rm_fronts = np.array([[3.39768913, 4.6265753 , 4.61201485, 4.17743499],
                      [3.11686657, 3.37643334, 3.71268422, 2.80596312],
                      [3.63754606, 4.92912784, 3.42151975, 3.4512155 ],
                      [3.01362226, 3.0008909 , 3.62965469, 3.61263049],
                      [3.4521001 , 3.91277842, 4.38606931, 4.57895422],
                      [3.99243196, 4.05693107, 4.49508072, 4.3103579 ]])

# measured L cell sizes
LCELL_SIZE = np.array([[3.96, 2.23], [3.96, 2.23], [3.9, 2.28], [4.71, 2.61], [4.57, 2.63]])

Nx, Ny = 1024, 512 # size of image frame
# load some parameters
vec_opt, FRONTLOCS_ALL, HEELLOCS_AVG = load_grid_heels_fronts()
HEELLOCS_AVG += np.array([1024, 512])[None, None, :]
FRONTLOCS_ALL = HEELLOCS_AVG + FRONTLOCS_ALL*25.2

# rotate grid vectors to obtain horizontally aligned grid
for hh in range(4):
    vec_opt[hh, ::2], vec_opt[hh, 1::2] = rotate_coord(vec_opt[hh, ::2],
                                                       vec_opt[hh, 1::2],
                                                       -np.arctan2(vec_opt[hh, 3],
                                                                   vec_opt[hh, 2]))

receptor = 5 # choose receptor subtype here
include_equator = False # whether or not to simulate the equator region
include_Lcells = False # whether or not to include the L cells in the density landscape

for hh, hour in enumerate(hours[1:]):
    print(hour)
    hh += 1
    # interpolate heel and front sizes
    rm_heel_int = np.linspace(rm_heels[:, index_func(hh-1)], rm_heels[:, index_func(hh)], nsteps)
    rm_front_int = np.linspace(rm_fronts[:, index_func(hh-1)], rm_fronts[:, index_func(hh)], nsteps)
    if hour > 26:
        v1old, v2old = vec_opt[hh-2, :2], vec_opt[hh-2, 2:]
        v1new, v2new = vec_opt[hh-1, :2], vec_opt[hh-1, 2:]
    else:
        v1old, v2old = vec_opt[0, :2] - (vec_opt[1, :2] - vec_opt[0, :2]),\
        vec_opt[0, 2:] - (vec_opt[1, 2:] - vec_opt[0, 2:])
        v1new, v2new = vec_opt[0, :2], vec_opt[0, 2:]

    # interpolate L cell size and grid-spanning vectors
    lcellsize_int = np.linspace(LCELL_SIZE[hh-1], LCELL_SIZE[hh], nsteps)
    v1inter, v2inter = np.linspace(v1old, v1new, nsteps), np.linspace(v2old, v2new, nsteps)

    # create bundle locations
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

    # grid offset
    Xshift, Xshift_old = 1024, 1024
    Yshift, Yshift_old = 512, 512
    Xstart = np.linspace(Xshift_old, Xshift, nsteps)
    Ystart = np.linspace(Yshift_old, Yshift, nsteps)

    # interpolate front locations, relative to heel locations
    if hour > 26:
        FRO_INT_X = np.linspace(FRONTLOCS_ALL[:, hh-2, 0] - HEELLOCS_AVG[:, hh-2, 0],
                                FRONTLOCS_ALL[:, hh-1, 0] - HEELLOCS_AVG[:, hh-1, 0], nsteps)
        FRO_INT_Y = np.linspace(FRONTLOCS_ALL[:, hh-2, 1] - HEELLOCS_AVG[:, hh-2, 1],
                                FRONTLOCS_ALL[:, hh-1, 1] - HEELLOCS_AVG[:, hh-1, 1], nsteps)
    else:
        FRO_INT_X = np.linspace(HEELLOCS_AVG[:, 0, 0], FRONTLOCS_ALL[:, 0, 0], nsteps)\
            - HEELLOCS_AVG[:, 0, 0]
        FRO_INT_Y = np.linspace(HEELLOCS_AVG[:, 0, 1], FRONTLOCS_ALL[:, 0, 1], nsteps)\
            - HEELLOCS_AVG[:, 0, 1]

    print('')
    for tind, tt in enumerate(np.linspace(hours[hh-1], hours[hh], nsteps, endpoint=True)):
        print('\r'+str(hour)+'hAPF: |' + (tind+1)*'=' + (nsteps-tind-1)*' ' + '|', end='')

        center_loc_x_int = (Xstart[tind] + n1*v1inter[tind, 0] + n2*v2inter[tind, 0]).flatten()
        center_loc_y_int = (Ystart[tind] + n1*v1inter[tind, 1] + n2*v2inter[tind, 1]).flatten()
        ind = np.ones(np.shape(center_loc_x), dtype='bool')
        Xint, Yint = center_loc_x_int[ind], center_loc_y_int[ind]

        if hh == 2:
            REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour, tind, nsteps)
        else:
            REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid2(Xint, Yint, hour)

        FRO_X2_int = REC_X_INTER + FRO_INT_X[tind, None].T
        FRO_Y2_int = REC_Y_INTER + FRO_INT_Y[tind, None].T

        # create equator if needed
        if include_equator:
            if tind == 0 and hh == 1:
                if receptor == 3:
                    border = np.mean(XC) - 10
                elif receptor == 6:
                    border = np.mean(XC)
                else:
                    border = np.mean(XC) - 20
            else:
                currmat = np.array([v1inter[tind], v2inter[tind]]).T
                prevmat = np.array([v1inter[tind-1], v2inter[tind-1]]).T
                border = ((currmat@np.linalg.inv(prevmat)@((border-Xstart[tind-1])*
                                                           np.ones(2)).T).T)[0] + Xstart[tind]

            REC_X_INTER[:, XC >= border] = REC_X_INTER[:, XC >= border] -\
            2*(REC_X_INTER[:, XC >= border] - XC[XC >= border][None, :])
            FRO_X2_int[:, XC >= border] = REC_X_INTER[:, XC >= border] - FRO_INT_X[tind, None].T

        # location arrays for the convolution
        xmin2, xmax2, ymin2, ymax2 = 0, 2*Nx, 0, 2*Ny
        POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx), np.linspace(ymin2, ymax2, 2*Ny))
        POS2 = np.reshape(np.meshgrid(np.linspace(xmin2, xmax2, Nx),
                                      np.linspace(ymin2, ymax2, Ny)), (2, -1))

        # parabola kernel to represent an individual heel, subtype-specific
        kernel_heel = np.zeros((512, 1024, 6))
        kernel_front = np.zeros((512, 1024, 6))
        hist_rec_temp_real = np.zeros((1023, 2047, 6))
        hist_rec_temp_real_front = np.zeros((1023, 2047, 6))
        # 2D histograms of heel and front locations
        for rr in range(6):
            kernel_heel[:, :, rr] = potential_parabola(POS2, np.array([[(xmin2+xmax2)/2], [(ymin2+ymax2)/2]]),
                                             rm_heel_int[tind, rr]*np.ones(2)*CONVFAC*2, 1).reshape((Ny, Nx))
            kernel_front[:, :, rr] = 0.5 * potential_parabola(POS2, np.array([[(xmin2+xmax2)/2], [(ymin2+ymax2)/2]]),
                                                              rm_front_int[tind, rr] * np.ones(2)*CONVFAC*2, 1).reshape((Ny, Nx))
            hist_rec_temp_real[:, :, rr] = np.histogram2d(np.ravel(REC_X_INTER[rr, :]), np.ravel(REC_Y_INTER[rr, :]),
                                                  bins=[np.linspace(xmin2, xmax2, 2*Nx),
                                                        np.linspace(ymin2, ymax2, 2*Ny)])[0].T
            hist_rec_temp_real_front[:, :, rr] = np.histogram2d(np.ravel(FRO_X2_int[rr, :]), np.ravel(FRO_Y2_int[rr, :]),
                                                                bins=[np.linspace(xmin2, xmax2, 2*Nx),
                                                                      np.linspace(ymin2, ymax2, 2*Ny)])[0].T
        hist_rec_temp_real_front[:, :, rr] = np.array(hist_rec_temp_real_front[:, :, rr], dtype = 'bool')
        hist_rec_temp_real[:, :, rr] = np.array(hist_rec_temp_real[:, :, rr], dtype = 'bool')
        # the histrogram of receptor locations is convolved with the parabola kernel
        dat_temp_real = np.zeros((1023, 2047, 6))
        dat_temp_real_front = np.zeros((1023, 2047, 6))
        # convolve with heel and front kernel
        for rr in range(6):
            dat_temp_real[:, :, rr] = fftconvolve(hist_rec_temp_real[:, :, rr], kernel_heel[:, :, rr], mode='same')
            dat_temp_real_front[:, :, rr] = fftconvolve(hist_rec_temp_real_front[:, :, rr], kernel_front[:, :, rr], mode='same')
        dat_heelfront = np.sum(dat_temp_real, axis=2) + np.sum(dat_temp_real_front, axis=2)

        # add L-cell density
        if include_Lcells:
            POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx-1), np.linspace(ymin2, ymax2, 2*Ny-1))
            POS2 = np.reshape(POS, (2, -1))
            index = (Xint > 400)*(Xint < 1600)*(Yint > 200) * (Yint < 900)
            if include_equator:
                if hour == 26:
                    add_lcell_circ_1 = potential_parabola(POS2, np.c_[Xint[index*(Xint > border)] + 0.333*v2inter[tind, 0], Yint[index*(Xint > border)] + 0.333*v2inter[tind, 1]].T, (0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                    add_lcell_circ_2 = potential_parabola(POS2, np.c_[Xint[index*(Xint <= border)] - 0.333*v2inter[tind, 0], Yint[index*(Xint <= border)] - 0.333*v2inter[tind, 1]].T, (0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                else:
                    add_lcell_circ_1 = potential_parabola(POS2, np.c_[Xint[index*(Xint > border)] + 0.333*v2inter[tind, 0], Yint[index*(Xint > border)] + 0.333*v2inter[tind, 1]].T, 0.6*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                    add_lcell_circ_2 = potential_parabola(POS2, np.c_[Xint[index*(Xint <= border)] - 0.333*v2inter[tind, 0], Yint[index*(Xint <= border)] - 0.333*v2inter[tind, 1]].T, 0.6*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                dat_lcell_circ = (np.sum(add_lcell_circ_1, axis=1) + np.sum(add_lcell_circ_2, axis=1)).reshape((2*Ny-1, 2*Nx-1))
            else:
                if hour == 26:
                    add_lcell_circ = potential_parabola(POS2, np.c_[Xint[index] - 0.333*v2inter[tind, 0], Yint[index] - 0.333*v2inter[tind, 1]].T, (0.4+0.2*tind/nsteps)*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                else:
                    add_lcell_circ = potential_parabola(POS2, np.c_[Xint[index] - 0.333*v2inter[tind, 0], Yint[index] - 0.333*v2inter[tind, 1]].T, 0.6*lcellsize_int[tind, 1]*CONVFAC*np.ones(2), 10)
                dat_lcell_circ = np.sum(add_lcell_circ, axis=1).reshape((2*Ny-1, 2*Nx-1))

            add_lcell_ell = potential_parabola(POS2, np.c_[Xint[index], Yint[index]].T, 0.8*lcellsize_int[tind, :]/2*CONVFAC, 10)
            dat_lcell_ell = np.sum(add_lcell_ell, axis=1).reshape((2*Ny-1, 2*Nx-1))
        else:
            dat_lcell_ell = np.zeros(np.shape(dat_heelfront))
            dat_lcell_circ = np.zeros(np.shape(dat_heelfront))

        # final density landscape
        dat2_inter = dat_heelfront + dat_lcell_circ + dat_lcell_ell

        # write to file, these files will be loaded by randomwalk_growthcone_density.py
        if include_equator:
            if include_Lcells:
                np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hour)+'hAPF_'+str(tind), dat2_inter)
            else:
                np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hour)+'hAPF_'+str(tind)+'_withoutLcells', dat2_inter)
        else:
            if include_Lcells:
                np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hour)+'hAPF_'+str(tind)+'_noeq', dat2_inter)
            else:
                np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hour)+'hAPF_'+str(tind)+'_noeq_withoutLcells', dat2_inter)
