import numpy as np
import pandas as pd
from plot_mp import plot_mp, deviation
import bokeh.plotting as bp
from bokeh.layouts import gridplot, widgetbox, column
from bokeh.models.widgets import Panel, Tabs, TextInput, Button, Div
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
import matplotlib.pyplot as plt

# Copyright (c) 2018-2019 Sam Ehrenstein. The full copyright notice is at the bottom of this file.

def prepare_profile(filename):
    prof = pd.read_csv(filename, skiprows=[0], header=None).values
    prof = prof[:, 0:5]
    return prof

# Expands the profile to be the length of t such that u(t) has the last value that would have been sent
# This is used instead of interpolating because the setpoint is updated slower than the robot's response time
def staircase(profile, t, dt, dt_prof):
    u = np.zeros((t.shape[0], 5))
    ratio = dt_prof/dt
    for i in range(profile.shape[0]):
        u[int(i*ratio):int((i+1)*ratio)] = profile[i]
    return u

def next_motor(vc, kp, ki, kd, ka, kv, last_pos, last_err,last_vel, ig, xset, vset, dt):
    c2 = ka/kv*(vc/kv-last_vel)
    c1 = -c2 + last_pos
    pos = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
    vel= vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
    err = xset-pos
    dedt = (err-last_err)/dt
    ig += err*dt
    vc = kp*err + ki*ig + kd*dedt + kv*vset
    if vc > 9:
        vc = 9
    return np.array([pos, err, vel]), vc, ig
def simulate_tank(kp, ki, kd, lka, lkv, rka, rkv, lprofile, rprofile, t, wb):
    pose = np.zeros((t.shape[0], 6))
    l_ig = 0
    r_ig = 0
    theta = np.zeros_like(t)
    lvc = 9
    rvc = 9
    for i in range(1, t.shape[0]):
        dt = t[i]-t[i-1]
        pose[i, 0:3], lvc, l_ig = next_motor(lvc, kp, ki, kd, lka, lkv, pose[i-1,0], 
            pose[i-1,1], pose[i-1,2], l_ig, lprofile[i-1,0], lprofile[i-1,1], dt)
        pose[i, 3:6], rvc, r_ig = next_motor(rvc, kp, ki, kd, rka, rkv, pose[i-1,3], 
            pose[i-1,4], pose[i-1,5], r_ig, rprofile[i-1,0], rprofile[i-1,1], dt)
        delta_l = pose[i,0-pose[i-1],0]
        delta_r = pose[i,3]-pose[i-1,3]
        theta[i] = theta[i-1] + (delta_l-delta_r)/wb
        e_theta = theta[i]-np.deg2rad(lprofile[i,4])
    return pose

    # pos = np.zeros_like(t)
    # vel = np.zeros_like(t)
    # err = np.zeros_like(t)
    # integral = 0
    # vc = 9
    # for i in range(1, t.shape[0]):
    #     dt = t[i]-t[i-1]
    #     c2 = ka/kv*(vc/kv-vel[i-1])
    #     c1 = -c2 + pos[i-1]
    #     pos[i] = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
    #     vel[i] = vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
    #     err[i] = profile[i, 0]-pos[i]
    #     dedt = (err[i]-err[i-1])/dt
    #     integral += err[i]*dt
    #     vc = kp*err[i] + ki*integral + kd*dedt + kv*profile[i, 1]
    #     if vc > 9:
    #         vc = 9
    # return pos, err, vel

def simulate(args):
    kv_l = args['kv_l']
    ka_l = args['ka_l']
    kv_r = args['kv_r']
    ka_r = args['ka_r']
    kp = args['kp']
    ki = args['ki']
    kd = args['kd']
    left_file = args['leftprof']
    right_file = args['rightprof']

    dt = 0.001
    dt_prof = 0.02

    left_profile = prepare_profile(left_file)
    right_profile = prepare_profile(right_file)
    # print('left profile', left_profile.shape)
    t = np.arange(0, left_profile.shape[0]*dt_prof-dt, dt)
    u_left = staircase(left_profile, t, dt, dt_prof)
    u_right = staircase(right_profile, t, dt, dt_prof)

    pose = simulate_tank(kp, ki, kd, ka_l, kv_l, ka_r, kv_r, u_left, u_right, t, 26/12)
    pos_l = pose[:,0]
    err_l = pose[:,1]
    vel_l = pose[:,2]
    pos_r = pose[:,3]
    err_l = pose[:,4]
    vel_l = pose[:,5]

    trajectories = plot_mp(pos_l, pos_r, dt)
    trajectories_prof = plot_mp(u_left[:,0], u_right[:,0], dt)

    plt.figure()
    plt.plot(trajectories_prof[:,1], trajectories_prof[:,2], label='intended left')
    plt.plot(trajectories_prof[:,3], trajectories_prof[:,4], label='intended right')
    plt.plot(trajectories[:,1], trajectories[:,2], label='actual left')
    plt.plot(trajectories[:,3], trajectories[:,4], label='actual rigt')
    plt.title('Intended vs. Actual Trajectories')
    plt.xlabel('x displacement (ft)')
    plt.ylabel('y displacement (ft')
    plt.legend()

    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.plot(t, u_left[:,0], label='left profile')
    plt.plot(t, pos_l, label='actual left distance')
    plt.title('Left side path')
    plt.xlabel('time (s)')
    plt.ylabel('distance (ft)')
    plt.legend()
    plt.subplot(212)
    plt.plot(t, u_right[:,0], label='right profile')
    plt.plot(t, pos_r, label='actual right distance')
    plt.title('Right side path')
    plt.xlabel('time (s)')
    plt.ylabel('distance (ft)')
    plt.legend()

    plt.show()

k_args = {'kp':7,
'ki':2,
'kd':0.3,
'kv_l':0.83,
'ka_l':0.1,
'kv_r':0.85,
'ka_r':0.11,
'leftprof':'demoLeft2.csv',
'rightprof':'demoRight2.csv'
}
simulate(k_args)

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