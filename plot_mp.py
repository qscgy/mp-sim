import numpy as np

# Copyright (c) 2018 Sam Ehrenstein. The full copyright notice is at the bottom of this file.


# Plots a motion profile. This is mostly a Python port of Noah Gleason's drawMP R script.
def plot_mp(left, right, dt):
    wheelbase_dia = 26.0/12  # Effective wheelbase; see whitepaper
    startingCenter = (0, 0)  # can be arbitrary for now

    out = np.zeros((left.shape[0], 5))
    out[0] = np.array([0, startingCenter[1], startingCenter[0]+wheelbase_dia/2,startingCenter[1],startingCenter[1]-wheelbase_dia/2])
    for i in range(1, out.shape[0]):
        perpendicular = angle_between(out[i-1, 1], out[i-1, 2], out[i-1, 3], out[i-1, 4])-np.pi/2
        out[i, 0] = out[i-1, 0] + dt
        delta_left = left[i]-left[i-1]
        delta_right = right[i]-right[i-1]

        theta = (delta_left-delta_right)/wheelbase_dia
        if theta == 0:
            out[i, 1] = out[i-1, 1] + delta_left*np.cos(perpendicular)
            out[i, 2] = out[i-1, 2] + delta_left*np.sin(perpendicular)
            out[i, 3] = out[i-1, 3] + delta_right*np.cos(perpendicular)
            out[i, 4] = out[i-1, 4] + delta_right*np.sin(perpendicular)
        else:
            rightR = (wheelbase_dia/2)*(delta_left+delta_right)/(delta_left-delta_right)-wheelbase_dia/2
            leftR = rightR + wheelbase_dia
            vectorTheta = (np.pi-theta)/2 - (np.pi/2-perpendicular)
            vectorDistanceWithoutR = np.sin(theta)/np.sin((np.pi-theta)/2)

            out[i, 1] = out[i-1, 1] + vectorDistanceWithoutR*leftR*np.cos(vectorTheta)
            out[i, 2] = out[i-1, 2] + vectorDistanceWithoutR*leftR*np.sin(vectorTheta)
            out[i, 3] = out[i-1, 3] + vectorDistanceWithoutR*rightR*np.cos(vectorTheta)
            out[i, 4] = out[i-1, 4] + vectorDistanceWithoutR*rightR*np.sin(vectorTheta)

    return out


# Helper method
def angle_between(lx, ly, rx, ry):
    delta_x = lx-rx
    delta_y = ly-ry
    angle = 0
    if delta_x == 0:
        angle = np.pi/2
    else:
        angle = np.arctan(delta_y/delta_x)

    if delta_y>0:
        if delta_x>0:
            return angle
        else:
            return np.pi-angle
    else:
        if delta_x>0:
            return -angle
        else:
            return angle-np.pi


# Compute the deviation of the actual trajectory from the profile trajectory at each time step
# We define deviation as the Euclidean distance between the predicted and actual positions
def deviation(predicted, actual):
    pred_T = np.transpose(predicted)
    act_T = np.transpose(actual)
    return np.array([dist(pred_T[1], act_T[1], pred_T[2], act_T[2]), dist(pred_T[3], act_T[3], pred_T[4], act_T[4])])


def dist(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)


# This file is part of MP-Sim.
#
# This program is free software; you can redistribute it and / or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this
# program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
# Fifth Floor, Boston, MA 02110 - 1301 USA
