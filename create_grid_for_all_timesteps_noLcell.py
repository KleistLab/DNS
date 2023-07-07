#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:56:38 2022

@author: eric
"""
import json
import base64
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import splrep, splev
from scipy.ndimage.filters import uniform_filter1d
from helper_functions import rotate_coord, load_grid_heels_fronts,\
potential_parabola, create_starting_grid_noLcell

hours = np.array([26, 30, 35, 40])
CONVFAC = 45.8 # conversion factor, pixels/micrometer

rm_heels = np.array([[3.10231051831059, 2.91492280980449, 2.04300797855269, 2.16799853146643],
                     [3.36613563287586, 3.6474344212202, 2.59092160570733, 2.49818435400807],
                     [3.26, 3.45, 2.15, 2.1],
                     [2.5838454966756, 3.26122733330694, 2.43382971588824, 1.79884449690457],
                     [2.73134911063699, 3.95170014933712, 3.89010177959303, 3.75649529229474],
                     [3.97548396049507, 3.37776310930197, 3.23239134019713, 3.02306813633871]])

rm_fronts = np.array([[3.39768913, 4.6265753 , 4.61201485, 4.17743499],
                      [3.11686657, 3.37643334, 3.71268422, 2.80596312],
                      [3.63754606, 4.92912784, 3.42151975, 3.4512155 ],
                      [3.01362226, 3.0008909 , 3.62965469, 3.61263049],
                      [3.4521001 , 3.91277842, 4.38606931, 4.57895422],
                      [3.99243196, 4.05693107, 4.49508072, 4.3103579 ]])

Nx, Ny = 1024, 512

# load the WT heel and front locations
_, FRONTLOCS_ALL, HEELLOCS_AVG = load_grid_heels_fronts()
FRONTLOCS_ALL[:, :, 1] = -FRONTLOCS_ALL[:, :, 1]
FRONTLOCS_ALL = HEELLOCS_AVG + 0.8 * FRONTLOCS_ALL*25.2

# load the noLcell grid vectors
with open("./data-files/optimal_grid_noLcell.json", "r", encoding='utf-8') as infile:
    loaded = json.load(infile).split('\"')
dtype = np.dtype(loaded[1])
arr = np.frombuffer(base64.decodebytes(bytearray(loaded[3], 'utf-8')), dtype)
vec_opt = arr.reshape((19, 4)).copy()

# horizontally align the grid vectors
for hh in range(19):
    vec_opt[hh, ::2], vec_opt[hh, 1::2] = rotate_coord(vec_opt[hh, ::2], vec_opt[hh, 1::2], -np.arctan2(vec_opt[hh, 3], vec_opt[hh, 2]))

# time array for simulation, specific to the noLcell condition
newtime = np.linspace(20, 45, 75, endpoint=True)
dt = np.diff(newtime)[0]

# interpolate the grid vectors at the simulation time points
vec_opt_ext = np.zeros((75, 4))
for i in range(4):
    y = uniform_filter1d(vec_opt[:, i], size=10)
    spl1 = splrep(np.hstack((np.array([20]), np.arange(27, 46))), np.hstack((np.mean(y), y)))
    vec_opt_ext[:, i] = splev(newtime, spl1)

# interpolate (cubic) rm_heel and rm_front at the simulation time points
rm_heel_int = np.zeros((6, 75))
rm_front_int = np.zeros((6, 75))
FRO_INT_X = np.zeros((6, 75))
FRO_INT_Y = np.zeros((6, 75))
for i in range(6):
    spl1 = splrep(hours, rm_heels[i, :])
    rm_heel_int[i, :] = splev(newtime, spl1)
    spl1 = splrep(hours, rm_fronts[i, :])
    rm_front_int[i, :] = splev(newtime, spl1)
    spl1 = splrep(np.hstack((np.array([20]), hours)), np.hstack((np.array([0]), FRONTLOCS_ALL[i, :, 0]-HEELLOCS_AVG[i, :, 0])))
    FRO_INT_X[i, :] = splev(newtime, spl1)
    spl1 = splrep(np.hstack((np.array([20]), hours)), np.hstack((np.array([0]), FRONTLOCS_ALL[i, :, 1]-HEELLOCS_AVG[i, :, 1])))
    FRO_INT_Y[i, :] = splev(newtime, spl1)

# linearly extrapolate to past and future simulation time points
# beyond the experimentally measured time points
rm_heel_int[:, newtime < 26] = np.linspace(rm_heels[:, 0] - (rm_heels[:, 1] - rm_heels[:, 0]), rm_heel_int[:, newtime < 26][:, -1], 18).T
rm_front_int[:, newtime < 26] = np.linspace(rm_fronts[:, 0] - (rm_fronts[:, 1] - rm_fronts[:, 0]), rm_front_int[:, newtime < 26][:, -1], 18).T
rm_heel_int[:, newtime > 40] = np.linspace(rm_heel_int[:, newtime > 40][:, 0], rm_heels[:, -1] - (rm_heels[:, -1] - rm_heels[:, -2]), 15).T
rm_front_int[:, newtime > 40] = np.linspace(rm_front_int[:, newtime > 40][:, 0], rm_fronts[:, -1] - (rm_fronts[:, -1] - rm_fronts[:, -2]), 15).T
FRO_INT_X[:, newtime > 40] = (FRONTLOCS_ALL[:, -1, 0]-HEELLOCS_AVG[:, -1, 0])[:, None]
FRO_INT_Y[:, newtime > 40] = (FRONTLOCS_ALL[:, -1, 1]-HEELLOCS_AVG[:, -1, 1])[:, None]

mini, maxi = -10, 10
n1, n2 = np.meshgrid(np.arange(mini, maxi), np.arange(mini, maxi))

include_equator = False # whether or not to include the equator
# choose receptor here, this receptor subtype is in the center of the density landscape
receptor = 1
for hh, hour in enumerate(newtime):
    print(hour)
    v1, v2 = vec_opt_ext[hh, :2], vec_opt_ext[hh, 2:]
    v1 = v1 / 24 * CONVFAC
    v2 = v2 / 24 * CONVFAC

    Xstart, Ystart = 1024, 512

    # span the grid
    center_loc_x_int = (Xstart + n1*v1[0] + n2*v2[0]).flatten()
    center_loc_y_int = (Ystart + n1*v1[1] + n2*v2[1]).flatten()
    ind = (center_loc_x_int > 0)*(center_loc_x_int < 2048)*(center_loc_y_int > 0)*(center_loc_y_int < 1024)
    Xint, Yint = center_loc_x_int[ind], center_loc_y_int[ind]

    # put the bundle shape around every grid position
    REC_X_INTER, REC_Y_INTER, XC, YC = create_starting_grid_noLcell(Xint, Yint)
    # calculate the resulting front positions
    FRO_X2_int = REC_X_INTER + FRO_INT_X[:, hh, None]
    FRO_Y2_int = REC_Y_INTER + FRO_INT_Y[:, hh, None]

    # flip heel and front locations at the equator if needed
    if include_equator:
        if hh == 0:
            border = np.mean(XC)
        else:
            currmat = np.array([v1, v2]).T
            prevmat = np.array([vec_opt_ext[hh-1, :2], vec_opt_ext[hh-1, 2:]]).T
            border = ((currmat@np.linalg.inv(prevmat)@((border-Xstart)*np.ones(2)).T).T)[0] + Xstart

        REC_X_INTER[:, XC >= border] = REC_X_INTER[:, XC >= border] - 2*(REC_X_INTER[:, XC >= border] - XC[XC >= border][None, :])
        FRO_X2_int[:, XC >= border] = REC_X_INTER[:, XC >= border] - FRO_INT_X[:, hh, None]

    # create the position vectors for the convolution
    xmin2, xmax2, ymin2, ymax2 = 0, 2*Nx, 0, 2*Ny
    POS = np.meshgrid(np.linspace(xmin2, xmax2, 2*Nx), np.linspace(ymin2, ymax2, 2*Ny))
    POS2 = np.reshape(np.meshgrid(np.linspace(xmin2, xmax2, Nx),
                                  np.linspace(ymin2, ymax2, Ny)), (2, -1))
    # parabola kernel to represent an individual heel
    kernel_heel = np.zeros((512, 1024, 6))
    kernel_front = np.zeros((512, 1024, 6))
    hist_rec_temp_real = np.zeros((1023, 2047, 6))
    hist_rec_temp_real_front = np.zeros((1023, 2047, 6))
    for rr in range(6):
        kernel_heel[:, :, rr] = potential_parabola(POS2, np.array([[(xmin2+xmax2)/2], [(ymin2+ymax2)/2]]),
                                         rm_heel_int[rr, hh]*np.ones(2)*CONVFAC*2, 1).reshape((Ny, Nx))
        kernel_front[:, :, rr] = 0.5 * potential_parabola(POS2, np.array([[(xmin2+xmax2)/2], [(ymin2+ymax2)/2]]),
                                                          rm_front_int[rr, hh] * np.ones(2)*CONVFAC*2, 1).reshape((Ny, Nx))
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
    for rr in range(6):
        dat_temp_real[:, :, rr] = fftconvolve(hist_rec_temp_real[:, :, rr], kernel_heel[:, :, rr], mode='same')
        dat_temp_real_front[:, :, rr] = fftconvolve(hist_rec_temp_real_front[:, :, rr], kernel_front[:, :, rr], mode='same')
    # the final density landscape
    dat2_inter = np.sum(dat_temp_real, axis=2) + np.sum(dat_temp_real_front, axis=2)

    # save to file
    if include_equator:
        np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hh)+'hAPF_noLcell', dat2_inter)
    else:
        np.save('./data-files/dat2_inter_R'+str(receptor)+'_'+str(hh)+'hAPF_noLcell_noeq_smaller', dat2_inter)
