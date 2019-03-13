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
    """
    Given a profile with some time step, returns a new profile with the same value
    at any given time, but with a different time step. In other words, a zeroth-order
    interpolation of the profile with a new time step.
    
    Args:
        profile (np.array): the profile to staircase
        t (np.array): the array times at which the returned profile should have values
        dt (float): the time step of t;  should divide into dt_prof
        dt_prof (float): the time step of profile

    Returns:
        np.array: the new profile
    """
    u = np.zeros((t.shape[0], 5))
    ratio = dt_prof/dt
    for i in range(profile.shape[0]):
        u[int(i*ratio):int((i+1)*ratio)] = profile[i]
    return u

def next_motor(vc, kp, ki, kd, ka, kv, last_pos, last_vel, dt):
    """
    Given the constants for a system moved by a motor under PID control and 
    information about its state at the start of a time slice, returns its 
    state after some time `dt` has passed.

    Args:
        vc (float): the voltage at the start of the time slice
        kp (float): the P constant in the PID loop
        ki (float): the I constant in the PID loop
        kd (float): the D constant in the PID loop
        ka (float): the kA for the motor, see README
        kv (float): the kV for the motor, see README
        last_pos (float): distance traveled by the system at the start of the time slice
        last_vel (float): velocity of the system at the start of the time slice
        dt (float): the length of the time slice

    Returns:
        np.array: an array of the form [position, error, velocity]
        float: the voltage at the end of 
    """    
    c2 = ka/kv*(vc/kv-last_vel)
    c1 = -c2 + last_pos
    pos = c1 + vc/kv*dt + c2*np.exp(-kv/ka*dt)
    vel= vc/kv-c2*kv/ka*np.exp(-kv/ka*dt)
    return pos, vel

def next_pid(kp, ki, kd, kv, pos, vel, last_err, ig, xset, vset, dt):
    """
    Computes the next output value of a motor-system under position PID control
    with a velocity feedforward.

    Args:
        kp (float): P value in the PID
        ki (float): I value in the PID
        kd (float): D value in the PID
        kv (float): kV of the system, see README
        pos (float): the current linear position
        vel (float): the current linear velocity
        last_err (float): the last value of the position error
        ig (float): the integral of the position error
        xset (float): the current position setpoint
        vset (float): the current velocity setpoint
        dt (float): the time step of the simulation

    Returns:
        float: the next output
        float: the current position error
        float: the current integral of the error
    """    
    err = xset-pos
    dedt = (err-last_err)/dt
    ig += err*dt
    vc = kp*err + ki*ig + kd*dedt + kv*vset
    if vc > 9:
        vc = 9
    return vc, err, ig

def simulate_tank(kp, ki, kd, lka, lkv, rka, rkv, lprofile, rprofile, t, wb, kpa, kia, kda):
    """
    Simulates a tank-drive robot following motion profiles under position PID contro
    with a velocity feedforward, cascaded with an outer PID loop on angular heading.

    Args:
        kp (float): P value in the position PID
        ki (float): I value in the position PID
        kd (float): D value in the position PID
        lka (float): left side kA, see README
        lkv (float): left side kV, see README
        rka (float): right side kA, see README
        rkv (float): right side kV, see README
        lprofile (np.array): left side profile, columns are position, velocity, acceleration, dt, heading
        rprofile (np.array): right side profile, same format as left side
        t (np.array): 1D array of the times at which the profile is evaluated
        wb (float): effective wheelbase diameter
        kpa (float): P in angular PID
        kia (float): I in angular PID
        kda (float): D in angular PID

    Returns:
        np.array: an array with as many rows as t and 6 columns containing the pose,
        in the form left pos, left err, left vel, right pos, right err, right vel
        np.array : an array with the same shape as t, containing the heading at each time
    """    
    pose = np.zeros((t.shape[0], 6))
    l_ig = 0
    r_ig = 0
    theta = np.zeros_like(t)
    theta_ig = 0
    e_theta = np.zeros_like(t)
    corr = 0
    lvc = 9
    rvc = 9
    for i in range(1, t.shape[0]):
        dt = t[i]-t[i-1]
        pose[i, 0], pose[i, 2] = next_motor(lvc, kp, ki, kd, lka, lkv, pose[i-1,0]-corr,
            pose[i-1,2], dt)
        lvc, pose[i, 1], l_ig = next_pid(kp, ki, kd, lkv, pose[i, 0], pose[i,2], pose[i-1,1],l_ig,
            lprofile[i,0], lprofile[i,1], dt)
        pose[i, 3], pose[i,5] = next_motor(rvc, kp, ki, kd, rka, rkv, pose[i-1,3]+corr, 
            pose[i-1,5], dt)
        rvc, pose[i, 4], r_ig = next_pid(kp, ki, kd, rkv, pose[i, 3], pose[i,5],pose[i-1,4], r_ig,
            rprofile[i,0], rprofile[i,1], dt)
        delta_l = pose[i,0]-pose[i-1,0]
        delta_r = pose[i,3]-pose[i-1,3]
        tp = (delta_r-delta_l)/wb
        theta[i] = theta[i-1] + tp
        corr, e_theta[i], theta_ig = next_pid(kpa, kia, kda, 0, theta[i], 0, e_theta[i-1],
            theta_ig, np.deg2rad(lprofile[i, 4]), 0, dt)

    return pose, theta

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
    kpa = args['kpa']
    kia = args['kia']
    kda = args['kda']
    left_file = args['leftprof']
    right_file = args['rightprof']

    dt = 0.001
    dt_prof = 0.02

    left_profile = prepare_profile(left_file)
    right_profile = prepare_profile(right_file)
    t = np.arange(0, left_profile.shape[0]*dt_prof-dt, dt)
    u_left = staircase(left_profile, t, dt, dt_prof)
    u_right = staircase(right_profile, t, dt, dt_prof)

    pose, theta = simulate_tank(kp, ki, kd, ka_l, kv_l, ka_r, kv_r, u_left, u_right, t, 26/12,
        kpa, kia, kda)
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
    plt.plot(trajectories[:,3], trajectories[:,4], label='actual right')
    plt.title('Intended vs. Actual Trajectories')
    plt.xlabel('x displacement (ft)')
    plt.ylabel('y displacement (ft')
    plt.legend()

    plt.figure(figsize=(6, 6))
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

    plt.figure()
    plt.plot(t, u_left[:,4], label='intended theta')
    plt.plot(t, theta*180/np.pi, label='actual theta')
    plt.legend()
    plt.title('Intended vs. Actual Theta')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')

    plt.show()

k_args = {'kp':7,
'ki':0,
'kd':0.3,
'kv_l':0.83,
'ka_l':0.1,
'kv_r':0.85,
'ka_r':0.11,
'kpa':0.2,
'kia':0,
'kda':0,
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